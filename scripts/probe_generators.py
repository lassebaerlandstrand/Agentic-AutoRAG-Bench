"""Profile generator models on a fixed exam: speed, accuracy, capability.

Runs one model or the whole ``search_space.generator`` pool of a project
config over a fixed 30-question exam, **closed-book** (the question is sent
straight to the model, no retrieval). Each call is timed; answers are scored
exact-match then LLM-judge. Prints a table ranked slowest-first so the models
worth pruning from the search space stand out, and writes a JSON report.

Closed-book isolates the generator's own speed and capability and needs no
corpus. Because the prompts are short, the per-call latency UNDERESTIMATES real
RAG generation (which prepends retrieved context) — use it for relative
ranking, not as a production-latency number.

Usage:
    uv run python scripts/probe_generators.py configs/hotpot_paper_project.yaml
    uv run python scripts/probe_generators.py configs/hotpot_paper_project.yaml --model azure/gpt-4o-mini
    uv run python scripts/probe_generators.py configs/hotpot_paper_project.yaml --no-judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from agentic_autorag.benchmark_eval.scoring import best_em, llm_judge
from agentic_autorag.config.loader import load_config
from agentic_autorag.litellm_runtime import (
    acompletion_with_cost,
    configure_litellm_runtime,
    install_model_aliases,
    resolve_model,
)
from dotenv import load_dotenv

DEFAULT_EXAM = Path(__file__).parent / "model_probe_exam.json"
DEFAULT_ENV = Path(__file__).resolve().parents[2] / "Agentic-AutoRAG" / ".env"
DEFAULT_OUT = Path("results_model_probe/probe.json")
DEFAULT_TIMEOUT_S = 120.0
INSTRUCTION = "Answer the question. Be concise.\n\nQuestion: {q}\nAnswer:"


def _load_exam(path: Path) -> list[dict]:
    exam = json.loads(path.read_text(encoding="utf-8"))
    if not exam:
        raise ValueError(f"exam file {path} is empty")
    return exam


async def _run_question(
    q: dict,
    model: str,
    sem: asyncio.Semaphore,
    judge_model: str | None,
    reasoning_effort: str | None,
    timeout_s: float,
) -> dict:
    """Answer one question (timed), then score it. One semaphore slot covers
    both the generation and the judge call.

    No token cap — matches ``RAGPipeline.generate`` (the real generator call
    path sets none) so reasoning models aren't truncated mid-thought; the
    per-call ``timeout`` is the ceiling, as it is in the repo.
    """
    async with sem:
        rec: dict = {"id": q["id"]}
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": INSTRUCTION.format(q=q["question"])}],
            "num_retries": 0,
            "timeout": timeout_s,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        start = time.perf_counter()
        try:
            response, usage = await acompletion_with_cost(cost_category="probe", **kwargs)
        except Exception as e:  # noqa: BLE001 — any failure marks the model unusable
            rec.update(
                ok=False,
                latency_s=time.perf_counter() - start,
                pred="",
                correct=False,
                refused=False,
                prompt_tokens=0,
                completion_tokens=0,
                usd=0.0,
                error=str(e),
            )
            return rec

        rec["latency_s"] = time.perf_counter() - start
        pred = (response.choices[0].message.content or "").strip()
        rec.update(
            ok=True,
            pred=pred,
            error=None,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            usd=usage["usd"],
        )

        gold = q["gold_answers"]
        if best_em(pred, gold) > 0:
            rec.update(correct=True, refused=False)
        elif not pred:
            rec.update(correct=False, refused=False)
        elif judge_model:
            verdict = await llm_judge(judge_model, q["question"], pred, gold)
            rec.update(correct=verdict == 1, refused=verdict == -1)
        else:
            rec.update(correct=False, refused=False)
        return rec


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list (non-empty)."""
    rank = max(1, int(-(-pct * len(values) // 1)))  # ceil(pct * n)
    return values[rank - 1]


async def _probe_model(
    model: str,
    exam: list[dict],
    concurrency: int,
    judge_model: str | None,
    reasoning_effort: str | None,
    timeout_s: float,
) -> dict:
    target, _ = resolve_model(model)
    sem = asyncio.Semaphore(concurrency)
    start = time.perf_counter()
    records = await asyncio.gather(
        *(_run_question(q, model, sem, judge_model, reasoning_effort, timeout_s) for q in exam)
    )
    wall_s = time.perf_counter() - start

    ok = [r for r in records if r["ok"]]
    lat = sorted(r["latency_s"] for r in ok)
    n = len(records)
    latency = None
    if lat:
        latency = {
            "mean": statistics.fmean(lat),
            "median": statistics.median(lat),
            "p90": _percentile(lat, 0.90),
            "max": lat[-1],
        }
    return {
        "model": model,
        "target": target,
        "n": n,
        "n_correct": sum(1 for r in records if r["correct"]),
        "n_refused": sum(1 for r in records if r["refused"]),
        "n_error": sum(1 for r in records if not r["ok"]),
        "accuracy": sum(1 for r in records if r["correct"]) / n,
        "latency_s": latency,
        "total_wall_s": wall_s,
        "mean_prompt_tokens": statistics.fmean(r["prompt_tokens"] for r in ok) if ok else 0.0,
        "mean_completion_tokens": statistics.fmean(r["completion_tokens"] for r in ok) if ok else 0.0,
        "total_usd": sum(r["usd"] for r in records),
        "missed": [r["id"] for r in records if not r["correct"]],
        "questions": records,
    }


def _sort_key(m: dict) -> float:
    # Slowest first; models that never returned (no latency) sort to the top —
    # unreachable/timed-out is the strongest exclusion signal.
    return m["latency_s"]["median"] if m["latency_s"] else float("inf")


def _fmt_s(x: float | None) -> str:
    return f"{x:6.1f}s" if x is not None else "   n/a"


def _print_report(summaries: list[dict]) -> None:
    header = (
        f"{'model':45s} {'acc':>5} {'ref':>3} {'err':>3} "
        f"{'median':>7} {'p90':>7} {'max':>7} {'in/out tok':>11} {'wall':>7} {'usd':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for m in summaries:
        lat = m["latency_s"] or {}
        name = m["model"] if len(m["model"]) <= 45 else m["model"][:44] + "…"
        print(
            f"{name:45s} {m['accuracy']:>5.0%} {m['n_refused']:>3} {m['n_error']:>3} "
            f"{_fmt_s(lat.get('median'))} {_fmt_s(lat.get('p90'))} {_fmt_s(lat.get('max'))} "
            f"{m['mean_prompt_tokens']:>4.0f}/{m['mean_completion_tokens']:<6.0f} "
            f"{m['total_wall_s']:6.1f}s ${m['total_usd']:>8.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="project YAML (source of generator pool, aliases, judge model)")
    parser.add_argument("--model", action="append", default=None, help="probe only this model (repeatable)")
    parser.add_argument("--exam", type=Path, default=DEFAULT_EXAM, help="exam JSON (default model_probe_exam.json)")
    parser.add_argument("--judge-model", type=str, default=None, help="override agent.judge_model")
    parser.add_argument("--no-judge", action="store_true", help="score exact-match only, no LLM judge")
    parser.add_argument("--reasoning-effort", type=str, default=None, help="default: config value; 'none' omits it")
    parser.add_argument("--concurrency", type=int, default=None, help="parallel calls/model (default from config)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-call timeout seconds")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON report path")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="path to .env (default sibling repo .env)")
    args = parser.parse_args()

    if args.env.exists():
        load_dotenv(args.env, override=False)
    else:
        print(f"WARN: env file {args.env} not found; relying on shell environment")
    configure_litellm_runtime()

    config = load_config(args.config)
    install_model_aliases(config.model_aliases)

    exam = _load_exam(args.exam)
    models = args.model if args.model else config.search_space.generator.models
    judge_model = None if args.no_judge else (args.judge_model or config.agent.judge_model)
    reasoning_effort = args.reasoning_effort
    if reasoning_effort is None:
        reasoning_effort = config.search_space.generator.reasoning_effort
    if reasoning_effort and reasoning_effort.lower() == "none":
        reasoning_effort = None
    concurrency = args.concurrency or config.agent.concurrency

    print(f"Probing {len(models)} model(s) on {len(exam)} questions from {args.exam.name} (closed-book)")
    print(f"config={args.config}  judge={judge_model or 'EM-only'}")
    print(f"reasoning_effort={reasoning_effort or 'off'}  concurrency={concurrency}")
    print("Latency is closed-book per-call wall time — for relative ranking, not production RAG latency.")
    print("=" * 72)

    summaries: list[dict] = []
    for model in models:
        print(f"  probing {model} ...", flush=True)
        summary = asyncio.run(
            _probe_model(model, exam, concurrency, judge_model, reasoning_effort, args.timeout)
        )
        summaries.append(summary)

    summaries.sort(key=_sort_key, reverse=True)
    _print_report(summaries)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "exam": str(args.exam),
        "n_questions": len(exam),
        "closed_book": True,
        "judge_model": judge_model,
        "reasoning_effort": reasoning_effort,
        "concurrency": concurrency,
        "models": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Report written to {args.out}  (total cost ${sum(m['total_usd'] for m in summaries):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
