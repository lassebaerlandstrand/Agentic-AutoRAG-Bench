#!/usr/bin/env python3
"""Experiment-1 matrix scheduler — a 2-worker, seed-major, dependency-gated
supervisor for the paper's accuracy headline (3 QA datasets x 5 methods x 3 seeds
+ kb_greedy reference).

It drives one ``(dataset, method, seed)`` unit per subprocess via the bench CLI
(``agentic-autorag-bench run -c <cfg> -m <method> --seeds <n> --resume``), keeping
at most 2 running at once (the DeepSeek-endpoint-safe ceiling). Finalization per
dataset is a single ``analyze`` render (no ``replay-holdout`` — variance comes from
the 3 seeds).

Design notes (all verified against the bench source):
  * SUCCESS IS DECIDED BY DISK, NOT EXIT CODE. ``run``'s matrix loop swallows
    per-unit exceptions with ``continue`` (run.py:977/985), so a single-unit run
    exits 0 even when it crashed internally or a ``motpe_warm`` cell was silently
    skipped for a missing transfer prior. A unit is DONE only when its
    ``benchmark_results.json`` (+ declared @k checkpoints) exists on disk —
    exactly ``run._is_method_seed_complete``, which we import.
  * HARD WARM GATE. ``motpe_warm/(ds,seed)`` reads
    ``<root>/random/seed_<n>/details/history.jsonl``; if random isn't complete it
    is silently skipped. So warm units depend on the corresponding random unit
    being DONE on disk before they launch.
  * RESUMABLE. Every unit runs with ``--resume``; the scheduler re-stats disk on
    (re)start and skips finished units, so a killed scheduler can be relaunched
    with the same command and continues where it stopped.

Launch detached (survives an interactive session dying), from the bench repo root:

    mkdir -p experiment-1/logs
    setsid nohup uv run python scripts/run_experiment1.py --include-kb-greedy \\
        > experiment-1/logs/nohup.out 2>&1 &
    echo $! > experiment-1/logs/scheduler.pid

Monitor:  tail -f experiment-1/logs/scheduler.log   ;   cat experiment-1/logs/STATUS.json
Dry run:  uv run python scripts/run_experiment1.py --dry-run   (prints the DAG, runs nothing)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# --- repo layout -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FOR = {
    "hotpot": "configs/hotpot_paper.yaml",
    "musique": "configs/musique_paper.yaml",
    "multihop": "configs/multihop_rag_paper.yaml",
}
DATASET_ORDER = {"hotpot": 0, "musique": 1, "multihop": 2}
# Tie-break within a seed: random first (warms the shared cache + unblocks warm),
# then the rest; kb_greedy last so long searches get the worker first.
METHOD_ORDER = {
    "random": 0,
    "agentic_score": 1,
    "agentic_nokb_nodiag": 2,
    "motpe": 3,
    "motpe_warm": 4,
    "kb_greedy": 5,
}
LOG_DIR = REPO_ROOT / "experiment-1" / "logs"
STATUS_PATH = LOG_DIR / "STATUS.json"
MASTER_LOG = LOG_DIR / "scheduler.log"

STATUS_FLUSH_S = 15.0  # loop heartbeat: status refresh + shutdown/backoff wake

# --- import the bench completion oracle (single source of truth) -------------
# Falls back to a self-contained reimplementation if the private symbol moves.
try:
    from agentic_autorag_bench.run import BenchConfig
    from agentic_autorag_bench.run import _is_method_seed_complete as _bench_complete
except Exception as exc:  # pragma: no cover - import guard
    print(f"FATAL: cannot import agentic_autorag_bench.run ({exc}). "
          f"Run under the bench venv (uv run python ...).", file=sys.stderr)
    raise


def _search_complete(output_root: Path, method: str, seed: int, checkpoints: list[int]) -> bool:
    seed_label = f"seed_{seed}"
    if _bench_complete is not None:
        return bool(_bench_complete(output_root, method, seed_label, checkpoints))
    # fallback (kept behaviourally identical to run._is_method_seed_complete)
    base = output_root / method / seed_label
    if not (base / "benchmark_results.json").exists():
        return False
    n_done: int | None = None
    meta = base / "optimizer_meta.json"
    if meta.exists():
        try:
            n_done = int(json.loads(meta.read_text(encoding="utf-8")).get("n_trials_completed"))
        except (ValueError, TypeError, json.JSONDecodeError):
            n_done = None
    for k in checkpoints:
        if n_done is not None and k >= n_done:
            continue
        if not (output_root / f"{method}@{k}" / seed_label / "benchmark_results.json").exists():
            return False
    return True


# --- data model --------------------------------------------------------------
class UnitKind(Enum):
    SEARCH = "search"
    KB_GREEDY = "kb_greedy"
    ANALYZE = "analyze"


class UnitState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"  # a hard dependency failed permanently


TERMINAL = {UnitState.DONE, UnitState.FAILED, UnitState.BLOCKED}


@dataclass
class Dataset:
    id: str
    config_path: Path
    output_root: Path
    methods: list[str]
    seeds: list[int]
    checkpoints: dict[str, list[int]]


@dataclass
class Unit:
    kind: UnitKind
    dataset: str
    key: str
    method: str | None = None
    seed: int | None = None
    deps: list[str] = field(default_factory=list)  # keys that must be DONE first
    state: UnitState = UnitState.PENDING
    attempts: int = 0
    not_before: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    last_error: str = ""
    log_path: Path | None = None


# --- runtime state -----------------------------------------------------------
class SchedulerState:
    def __init__(self) -> None:
        self.shutdown = False
        self.durations: list[float] = []  # completed unit wall-times, for ETA


STATE = SchedulerState()
PROC_REG: dict[str, subprocess.Popen] = {}
REG_LOCK = threading.Lock()


def now() -> float:
    return time.time()


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def master_log(event: str, unit: Unit | None = None, **kw: object) -> None:
    parts = [ts(), event]
    if unit is not None:
        parts.append(unit.key)
    for k, v in kw.items():
        parts.append(f"{k}={v}")
    line = " ".join(str(p) for p in parts)
    try:
        with open(MASTER_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


# --- config / DAG ------------------------------------------------------------
def load_datasets(selected_ids: list[str], selected_seeds: list[int] | None) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {}
    for ds_id in selected_ids:
        cfg = REPO_ROOT / CONFIG_FOR[ds_id]
        if not cfg.exists():
            raise SystemExit(f"config not found: {cfg}")
        bench = BenchConfig.load(cfg)
        seeds = list(bench.seeds)
        if selected_seeds is not None:
            seeds = [s for s in seeds if s in set(selected_seeds)]
        datasets[ds_id] = Dataset(
            id=ds_id,
            config_path=cfg,
            output_root=bench.output_root,
            methods=list(bench.methods),
            seeds=seeds,
            checkpoints=dict(bench.checkpoints),
        )
    return datasets


def build_units(datasets: dict[str, Dataset], include_kb: bool) -> dict[str, Unit]:
    units: dict[str, Unit] = {}
    for ds in datasets.values():
        member_keys: list[str] = []  # search + kb keys the analyze unit waits on
        for seed in ds.seeds:
            for method in ds.methods:
                key = f"{ds.id}/{method}/seed_{seed}"
                deps = []
                if method == "motpe_warm":
                    deps = [f"{ds.id}/random/seed_{seed}"]
                units[key] = Unit(UnitKind.SEARCH, ds.id, key, method=method, seed=seed, deps=deps)
                member_keys.append(key)
            if include_kb:
                key = f"{ds.id}/kb_greedy/seed_{seed}"
                units[key] = Unit(UnitKind.KB_GREEDY, ds.id, key, method="kb_greedy", seed=seed)
                member_keys.append(key)
        akey = f"{ds.id}/analyze"
        units[akey] = Unit(UnitKind.ANALYZE, ds.id, akey, deps=list(member_keys))
    return units


# --- completion / dependency predicates (disk-truth) -------------------------
def is_unit_complete(ds: Dataset, u: Unit) -> bool:
    if u.kind is UnitKind.SEARCH:
        return _search_complete(ds.output_root, u.method, u.seed, ds.checkpoints.get(u.method, []))
    if u.kind is UnitKind.KB_GREEDY:
        return (ds.output_root / "kb_greedy" / f"seed_{u.seed}" / "benchmark_results.json").exists()
    if u.kind is UnitKind.ANALYZE:
        return (ds.output_root / ".scheduler_analyze_done").exists()
    return False


def eligibility(u: Unit, units: dict[str, Unit], datasets: dict[str, Dataset]) -> str:
    """'ready' | 'waiting' | 'blocked' — resolves deps by unit STATE.

    The analyze unit uses a relaxed gate: it becomes ready once every member
    (search+kb) unit of its dataset is terminal (DONE/FAILED/BLOCKED), so one
    permanently-failed cell can't wedge the dataset (a warning is logged when it
    runs with holes).
    """
    if u.kind is UnitKind.ANALYZE:
        members = [units[k] for k in u.deps]
        if all(m.state in TERMINAL for m in members):
            return "ready"
        return "waiting"
    for dk in u.deps:  # hard deps (motpe_warm -> random)
        d = units[dk]
        if d.state in (UnitState.FAILED, UnitState.BLOCKED):
            return "blocked"
        if d.state is not UnitState.DONE:
            return "waiting"
    return "ready"


def any_search_done(units: dict[str, Unit], ds_id: str) -> bool:
    return any(
        u.dataset == ds_id and u.kind is UnitKind.SEARCH and u.state is UnitState.DONE
        for u in units.values()
    )


# --- selection ---------------------------------------------------------------
def select_next(
    units: dict[str, Unit],
    datasets: dict[str, Dataset],
    busy_datasets: set[str],
    warmup: bool,
    seed_barrier: bool,
) -> Unit | None:
    t = now()
    candidates: list[Unit] = []
    for u in units.values():
        if u.state is not UnitState.PENDING:
            continue
        if u.not_before > t:  # in retry backoff
            continue
        elig = eligibility(u, units, datasets)
        if elig == "blocked":
            u.state = UnitState.BLOCKED
            master_log("BLOCKED", u, reason="dependency-failed")
            continue
        if elig == "waiting":
            continue
        # fast-path: already complete on disk (resume / sibling produced it)
        if is_unit_complete(datasets[u.dataset], u):
            u.state = UnitState.DONE
            continue
        # warmup: don't start a 2nd unit of a fresh dataset until one search
        # unit landed, so the exam/corpus/embed cache is built by one writer.
        if (warmup and u.kind in (UnitKind.SEARCH, UnitKind.KB_GREEDY)
                and u.dataset in busy_datasets and not any_search_done(units, u.dataset)):
            continue
        candidates.append(u)

    if not candidates:
        return None

    def srank(u: Unit) -> int:
        return u.seed if u.seed is not None else 10 ** 6  # analyze sorts last

    if seed_barrier:
        open_level = min((srank(u) for u in units.values() if u.state not in TERMINAL), default=10 ** 6)
        held = [c for c in candidates if srank(c) <= open_level]
        candidates = held or candidates

    def key(u: Unit):
        return (
            srank(u),                                      # 1. seed-major
            0 if u.method == "random" else 1,              # 2. random early
            0 if u.dataset not in busy_datasets else 1,    # 3. dataset affinity
            DATASET_ORDER.get(u.dataset, 9),               # 4. stable
            METHOD_ORDER.get(u.method, 50),
            u.seed or 0,
        )

    return min(candidates, key=key)


# --- execution ---------------------------------------------------------------
def build_argv(ds: Dataset, u: Unit) -> list[str]:
    base = [sys.executable, "-m", "agentic_autorag_bench.cli"]
    cfg = str(ds.config_path)
    if u.kind is UnitKind.SEARCH:
        return base + ["run", "-c", cfg, "-m", u.method, "--seeds", str(u.seed), "--resume"]
    if u.kind is UnitKind.KB_GREEDY:
        return base + ["kb-greedy", "-c", cfg, "--seed", str(u.seed)]
    if u.kind is UnitKind.ANALYZE:
        return base + ["analyze", "--results-dir", str(ds.output_root)]
    raise ValueError(u.kind)


def run_unit_blocking(u: Unit, argv: list[str]) -> int:
    u.log_path = LOG_DIR / (u.key.replace("/", "__") + ".log")
    with open(u.log_path, "ab", buffering=0) as fh:
        header = f"\n==== attempt {u.attempts} {ts()} :: {' '.join(argv)}\n"
        fh.write(header.encode())
        proc = subprocess.Popen(
            argv, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group -> clean group kill, no signal bounce
        )
        with REG_LOCK:
            PROC_REG[u.key] = proc
        try:
            rc = proc.wait()
        finally:
            with REG_LOCK:
                PROC_REG.pop(u.key, None)
        return rc


def backoff_seconds(attempt: int, base: float, cap: float) -> float:
    # deterministic exponential backoff (no RNG — keep the process resume-safe);
    # attempt is the count just completed (1-based).
    return min(base * (2 ** (attempt - 1)), cap)


def record_failure_or_retry(u: Unit, reason: str, max_attempts: int, base: float, cap: float) -> None:
    u.last_error = reason
    if u.attempts >= max_attempts:
        u.state = UnitState.FAILED
        master_log("FAIL", u, reason=reason, attempts=u.attempts)
    else:
        wait_s = backoff_seconds(u.attempts, base, cap)
        u.not_before = now() + wait_s
        u.state = UnitState.PENDING
        master_log("RETRY", u, reason=reason, next_attempt=u.attempts + 1, wait_s=int(wait_s))


def handle_completion(u: Unit, fut: Future, datasets: dict[str, Dataset],
                      units: dict[str, Unit], max_attempts: int, base: float, cap: float) -> None:
    u.finished_at = now()
    ds = datasets[u.dataset]
    exc = fut.exception()
    if exc is not None:
        record_failure_or_retry(u, f"thread:{exc}", max_attempts, base, cap)
        return
    u.returncode = fut.result()

    if u.kind is UnitKind.ANALYZE:
        # analyze is a pure render; unlike `run` it does not swallow-and-continue,
        # so exit 0 == success. Mark with a scheduler-owned sentinel.
        if u.returncode == 0:
            try:
                (ds.output_root / ".scheduler_analyze_done").write_text(ts() + "\n", encoding="utf-8")
            except OSError:
                pass
            missing = [k for k in u.deps if units[k].state is not UnitState.DONE]
            if missing:
                master_log("INCOMPLETE_MATRIX", u, missing=",".join(missing))
            u.state = UnitState.DONE
            if u.started_at:
                STATE.durations.append(u.finished_at - u.started_at)
            master_log("DONE", u, rc=0)
        else:
            record_failure_or_retry(u, f"exit{u.returncode}", max_attempts, base, cap)
        return

    # SEARCH / KB_GREEDY: disk state is authoritative (exit code is not).
    if is_unit_complete(ds, u):
        u.state = UnitState.DONE
        if u.started_at:
            STATE.durations.append(u.finished_at - u.started_at)
        master_log("DONE", u, rc=u.returncode)
    else:
        reason = "incomplete-after-exit0" if u.returncode == 0 else f"exit{u.returncode}"
        record_failure_or_retry(u, reason, max_attempts, base, cap)


# --- status ------------------------------------------------------------------
def _eta_seconds(units: dict[str, Unit], workers: int) -> float | None:
    if not STATE.durations:
        return None
    mean = sum(STATE.durations) / len(STATE.durations)
    remaining = sum(1 for u in units.values() if u.state not in TERMINAL)
    running = sum(1 for u in units.values() if u.state is UnitState.RUNNING)
    return mean * (remaining + running) / max(1, workers)


def write_status(units: dict[str, Unit], datasets: dict[str, Dataset], workers: int) -> None:
    counts = {s.value: 0 for s in UnitState}
    for u in units.values():
        counts[u.state.value] += 1
    running = [
        {"key": u.key, "attempt": u.attempts,
         "started_at": datetime.fromtimestamp(u.started_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if u.started_at else None,
         "log": str(u.log_path) if u.log_path else None}
        for u in units.values() if u.state is UnitState.RUNNING
    ]
    ds_progress = {}
    for ds_id in datasets:
        search = [u for u in units.values() if u.dataset == ds_id and u.kind is UnitKind.SEARCH]
        done = sum(1 for u in search if u.state is UnitState.DONE)
        ds_progress[ds_id] = {
            "search_done": f"{done}/{len(search)}",
            "finalized": units[f"{ds_id}/analyze"].state is UnitState.DONE,
        }
    eta = _eta_seconds(units, workers)
    payload = {
        "updated_at": ts(),
        "pid": os.getpid(),
        "workers": workers,
        "counts": counts,
        "eta_hours": round(eta / 3600, 2) if eta is not None else None,
        "running": running,
        "datasets": ds_progress,
        "units": [
            {"key": u.key, "kind": u.kind.value, "state": u.state.value, "attempts": u.attempts,
             "rc": u.returncode, "last_error": u.last_error}
            for u in units.values()
        ],
    }
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, STATUS_PATH)
    except OSError as e:
        master_log("STATUS_WRITE_ERROR", err=str(e))


# --- signals -----------------------------------------------------------------
def install_signal_handlers() -> None:
    def handler(signum, _frame):
        STATE.shutdown = True
        master_log("SIGNAL", sig=signum)
        with REG_LOCK:
            procs = list(PROC_REG.values())
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def graceful_drain(pool: ThreadPoolExecutor, in_flight: dict[Future, Unit], grace_s: float = 120.0) -> None:
    master_log("DRAIN", n=len(in_flight))
    deadline = now() + grace_s
    while in_flight and now() < deadline:
        done, _ = wait(list(in_flight), timeout=2.0, return_when=FIRST_COMPLETED)
        for fut in done:
            in_flight.pop(fut, None)
    with REG_LOCK:
        procs = list(PROC_REG.values())
    for p in procs:  # escalate anything still alive
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    pool.shutdown(wait=False)


# --- supervisor --------------------------------------------------------------
def supervise(units: dict[str, Unit], datasets: dict[str, Dataset], *,
              workers: int, warmup: bool, seed_barrier: bool,
              max_attempts: int, base: float, cap: float) -> int:
    pool = ThreadPoolExecutor(max_workers=workers)
    in_flight: dict[Future, Unit] = {}
    try:
        while True:
            if STATE.shutdown:
                break
            # fill free slots
            while len(in_flight) < workers:
                busy = {u.dataset for u in in_flight.values()}
                u = select_next(units, datasets, busy, warmup, seed_barrier)
                if u is None:
                    break
                u.state = UnitState.RUNNING
                u.attempts += 1
                u.started_at = now()
                argv = build_argv(datasets[u.dataset], u)
                master_log("START", u, attempt=u.attempts)
                fut = pool.submit(run_unit_blocking, u, argv)
                in_flight[fut] = u

            if not in_flight:
                if all(u.state in TERMINAL for u in units.values()):
                    break
                # nothing runnable now & nothing running: sleep to soonest backoff
                pend = [u for u in units.values() if u.state is UnitState.PENDING]
                soon = min((u.not_before for u in pend if u.not_before > now()), default=None)
                if soon is None:
                    master_log("STUCK", note="no runnable/in-flight units; non-terminal remain")
                    break
                time.sleep(min(max(soon - now(), 1.0), STATUS_FLUSH_S))
                write_status(units, datasets, workers)
                continue

            done, _ = wait(list(in_flight), timeout=STATUS_FLUSH_S, return_when=FIRST_COMPLETED)
            for fut in done:
                u = in_flight.pop(fut)
                handle_completion(u, fut, datasets, units, max_attempts, base, cap)
            write_status(units, datasets, workers)

        if STATE.shutdown:
            graceful_drain(pool, in_flight)
            # interrupted RUNNING units revert to PENDING for a clean resume
            for u in units.values():
                if u.state is UnitState.RUNNING:
                    u.state = UnitState.PENDING
            write_status(units, datasets, workers)
            master_log("SHUTDOWN")
            return 130
    finally:
        pool.shutdown(wait=False)

    write_status(units, datasets, workers)
    failed = [u.key for u in units.values() if u.state in (UnitState.FAILED, UnitState.BLOCKED)]
    if failed:
        master_log("COMPLETE_WITH_FAILURES", n=len(failed), failed=",".join(failed))
        return 1
    master_log("COMPLETE_OK")
    return 0


# --- dry run -----------------------------------------------------------------
def dry_run(units: dict[str, Unit], datasets: dict[str, Dataset]) -> int:
    print(f"REPO_ROOT = {REPO_ROOT}")
    kinds = {k: 0 for k in UnitKind}
    complete = 0
    for u in units.values():
        kinds[u.kind] += 1
        if is_unit_complete(datasets[u.dataset], u):
            complete += 1
    print("Unit counts: " + ", ".join(f"{k.value}={v}" for k, v in kinds.items())
          + f"  (total={len(units)}, already-complete-on-disk={complete})")
    for ds in datasets.values():
        print(f"\n[{ds.id}] output_root={ds.output_root}  methods={ds.methods}  "
              f"seeds={ds.seeds}  checkpoints={ds.checkpoints}")
    print("\nDependency gates (warm -> random):")
    for u in units.values():
        if u.method == "motpe_warm":
            print(f"  {u.key}  requires  {u.deps[0]}")
    print("\nAnalyze finalization deps (per dataset):")
    for u in units.values():
        if u.kind is UnitKind.ANALYZE:
            print(f"  {u.key}  waits on {len(u.deps)} member units")
    print("\nExample argv per kind:")
    seen: set[UnitKind] = set()
    for u in units.values():
        if u.kind not in seen:
            seen.add(u.kind)
            print("  " + " ".join(build_argv(datasets[u.dataset], u)))
    print("\n(dry-run: nothing executed)")
    return 0


# --- cli ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment-1 matrix scheduler (2-worker, seed-major).")
    p.add_argument("--datasets", default="hotpot,musique,multihop",
                   help="comma-separated subset of hotpot,musique,multihop")
    p.add_argument("--seeds", default=None, help="comma-separated seed subset (default: each config's seeds)")
    p.add_argument("--workers", type=int, default=2, help="max concurrent units (endpoint-safe ceiling is 2)")
    p.add_argument("--include-kb-greedy", action="store_true", help="also run kb_greedy at 3 seeds per dataset")
    p.add_argument("--seed-barrier", action="store_true",
                   help="strict seed barrier (idles a worker at boundaries); default is soft (never idle)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   help="disable the single-writer cache warmup rule")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--backoff-base", type=float, default=60.0)
    p.add_argument("--backoff-cap", type=float, default=900.0)
    p.add_argument("--dry-run", action="store_true", help="print the DAG and exit without executing")
    p.add_argument("--config-map", default=None,
                   help="override config paths, e.g. 'hotpot=configs/_smoke_hotpot.yaml' (comma-separated); for smoke tests")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)  # BenchConfig.load resolves output_root against CWD
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.config_map:
        for pair in args.config_map.split(","):
            k, _, v = pair.partition("=")
            CONFIG_FOR[k.strip()] = v.strip()

    ds_ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bad = [d for d in ds_ids if d not in CONFIG_FOR]
    if bad:
        raise SystemExit(f"unknown datasets: {bad} (choose from {list(CONFIG_FOR)})")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    datasets = load_datasets(ds_ids, seeds)
    units = build_units(datasets, include_kb=args.include_kb_greedy)

    if args.dry_run:
        return dry_run(units, datasets)

    if args.workers > 2:
        master_log("WARN", note=f"workers={args.workers} exceeds the endpoint-safe ceiling of 2")

    install_signal_handlers()
    master_log("LAUNCH", datasets=",".join(ds_ids), workers=args.workers,
               kb=args.include_kb_greedy, seed_barrier=args.seed_barrier,
               warmup=args.warmup, units=len(units))
    return supervise(
        units, datasets,
        workers=args.workers, warmup=args.warmup, seed_barrier=args.seed_barrier,
        max_attempts=args.max_attempts, base=args.backoff_base, cap=args.backoff_cap,
    )


if __name__ == "__main__":
    sys.exit(main())
