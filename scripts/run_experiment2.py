#!/usr/bin/env python3
"""Experiment-2 (UniDoc Pareto) scheduler — a 2-worker, dependency-gated supervisor
for the cost-vs-accuracy Pareto experiment.

Unlike Exp-1 (a `(dataset, method, seed)` matrix over held-out QA), Exp-2 runs a
single dataset (UniDoc) with NO held-out gold: every method is scored on the
optimizer's OWN self-generated exam and also minimizes cost. The `pareto` command
is the engine; this scheduler just drives it as a resume-safe DAG of subprocesses:

    SETUP  ->  { agentic_cost, random, motpe, motpe_warm } x seeds  ->  FINALIZE

Design notes (verified against the bench + framework source):
  * SINGLE-WRITER WARMUP. The shared corpus-parse cache and `exam.json` are
    non-atomic and unlocked, so two cold processes would double-generate a
    DIFFERENT exam (a correctness bug). The SETUP unit (`pareto --setup-only`)
    builds them ONCE (single writer) and drops a `.setup_complete` marker; every
    method cell depends on SETUP, so the fan-out only starts once the shared
    caches exist. After that, N cells run concurrently and safely (the per-trial
    ingredient cache is atomic + content-addressed, and cache_max_gb is high
    enough that LRU eviction never fires).
  * SUCCESS IS DECIDED BY DISK, NOT EXIT CODE. A method cell is DONE only when
    `<root>/<method>/seed_<n>/optimizer_meta.json` reports
    `n_trials_completed >= max_trials` (`pareto.method_seed_complete`). The Pareto
    path never writes `benchmark_results.json`, so the Exp-1 sentinel does not
    apply. SETUP is DONE iff the marker exists; FINALIZE iff `hypervolume.json`
    exists.
  * HARD WARM GATE. `motpe_warm/seed_n` reads `random/seed_n/details/history.jsonl`
    as its transfer prior, so a warm cell depends on the same-seed random cell
    being DONE on disk before it launches.
  * RESUMABLE. Every method cell runs with `--resume`; the scheduler re-stats disk
    on (re)start and skips finished units, so a killed scheduler relaunched with
    the same command continues where it stopped. SETUP is idempotent (cached).

Launch detached (survives an interactive session dying), from the bench repo root:

    mkdir -p experiment-2/logs
    setsid nohup uv run python scripts/run_experiment2.py \\
        --methods agentic_cost,motpe --seeds 1 --workers 2 \\
        > experiment-2/logs/nohup.out 2>&1 &
    echo $! > experiment-2/logs/scheduler.pid

Monitor:  tail -f experiment-2/logs/scheduler.log   ;   cat experiment-2/logs/STATUS.json
Dry run:  uv run python scripts/run_experiment2.py --dry-run   (prints the DAG, runs nothing)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

# --- repo layout -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/unidoc_pareto.yaml"
LOG_DIR = REPO_ROOT / "experiment-2" / "logs"
STATUS_PATH = LOG_DIR / "STATUS.json"
MASTER_LOG = LOG_DIR / "scheduler.log"

STATUS_FLUSH_S = 15.0  # loop heartbeat: status refresh + shutdown/backoff wake

# Canonical run order within a seed (random before its warm consumer). SETUP is
# forced first (no deps), FINALIZE last (waits on every method cell).
METHOD_ORDER = {"agentic_cost": 0, "random": 1, "motpe_warm": 2, "motpe": 3}
_SEED_INF = 10 ** 6  # sort key for seed-less units (SETUP / FINALIZE)

# --- import the bench completion oracle (single source of truth) -------------
try:
    from agentic_autorag_bench.pareto import (
        SETUP_MARKER,
        ParetoConfig,
        _read_shared_cache_dir,
        method_seed_complete,
    )
except Exception as exc:  # pragma: no cover - import guard
    print(
        f"FATAL: cannot import agentic_autorag_bench.pareto ({exc}). "
        f"Run under the bench venv (uv run python ...).",
        file=sys.stderr,
    )
    raise


# --- data model --------------------------------------------------------------
class UnitKind(Enum):
    SETUP = "setup"
    METHOD = "method"
    FINALIZE = "finalize"


# SETUP scheduled first, method cells next, FINALIZE last.
_KIND_RANK = {UnitKind.SETUP: 0, UnitKind.METHOD: 1, UnitKind.FINALIZE: 2}


def _unit_sort_key(u: Unit) -> tuple[int, int, int]:
    return (_KIND_RANK[u.kind], u.seed if u.seed is not None else _SEED_INF, METHOD_ORDER.get(u.method, 50))


class UnitState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"  # a hard dependency failed permanently


TERMINAL = {UnitState.DONE, UnitState.FAILED, UnitState.BLOCKED}


@dataclass
class Experiment:
    config_path: Path
    project_config_path: Path
    output_root: Path
    shared_cache_dir: Path
    max_trials: int
    methods: list[str]
    seeds: list[int]


@dataclass
class Unit:
    kind: UnitKind
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

SETUP_KEY = "setup"
FINALIZE_KEY = "finalize"


def now() -> float:
    return time.time()


def ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
def load_experiment(config_rel: str, methods_override: list[str] | None,
                    seeds_override: list[int] | None) -> Experiment:
    cfg_path = REPO_ROOT / config_rel
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = ParetoConfig.load(cfg_path)
    methods = list(methods_override) if methods_override else list(cfg.methods)
    seeds = list(seeds_override) if seeds_override else list(cfg.seeds)
    bad = [m for m in methods if m not in METHOD_ORDER]
    if bad:
        raise SystemExit(f"unknown methods {bad} (choose from {list(METHOD_ORDER)})")
    return Experiment(
        config_path=cfg_path,
        project_config_path=cfg.project_config_path,
        output_root=cfg.output_root,
        shared_cache_dir=_read_shared_cache_dir(cfg.project_config_path),
        max_trials=cfg.max_trials,
        methods=methods,
        seeds=seeds,
    )


def build_units(exp: Experiment) -> dict[str, Unit]:
    units: dict[str, Unit] = {}
    units[SETUP_KEY] = Unit(UnitKind.SETUP, SETUP_KEY, deps=[])
    method_keys: list[str] = []
    for seed in exp.seeds:
        for method in exp.methods:
            key = f"{method}/seed_{seed}"
            deps = [SETUP_KEY]
            if method == "motpe_warm":
                # hard warm gate: needs the same-seed random cell complete on disk.
                # (The random cell must be in the run; if it isn't, run_pareto's
                # own guard also refuses — surfaced here as a blocked dependency.)
                deps.append(f"random/seed_{seed}")
            units[key] = Unit(UnitKind.METHOD, key, method=method, seed=seed, deps=deps)
            method_keys.append(key)
    units[FINALIZE_KEY] = Unit(UnitKind.FINALIZE, FINALIZE_KEY, deps=list(method_keys))
    return units


# --- completion / dependency predicates (disk-truth) -------------------------
def is_unit_complete(exp: Experiment, u: Unit) -> bool:
    if u.kind is UnitKind.SETUP:
        return (exp.shared_cache_dir / SETUP_MARKER).exists()
    if u.kind is UnitKind.METHOD:
        return method_seed_complete(exp.output_root, u.method, u.seed, exp.max_trials)
    if u.kind is UnitKind.FINALIZE:
        return (exp.output_root / "hypervolume.json").exists()
    return False


def eligibility(u: Unit, units: dict[str, Unit]) -> str:
    """'ready' | 'waiting' | 'blocked' — resolves deps by unit STATE.

    FINALIZE uses a relaxed gate: ready once every method cell is terminal
    (DONE/FAILED/BLOCKED), so one permanently-failed cell can't wedge the render
    (a warning is logged if it finalizes with holes). Method cells use a strict
    gate: every dep must be DONE.
    """
    if u.kind is UnitKind.FINALIZE:
        members = [units[k] for k in u.deps]
        if all(m.state in TERMINAL for m in members):
            return "ready"
        return "waiting"
    for dk in u.deps:  # strict: SETUP (+ motpe_warm -> random)
        d = units[dk]
        if d.state in (UnitState.FAILED, UnitState.BLOCKED):
            return "blocked"
        if d.state is not UnitState.DONE:
            return "waiting"
    return "ready"


# --- selection ---------------------------------------------------------------
def select_next(units: dict[str, Unit], exp: Experiment) -> Unit | None:
    t = now()
    candidates: list[Unit] = []
    for u in units.values():
        if u.state is not UnitState.PENDING:
            continue
        if u.not_before > t:  # in retry backoff
            continue
        elig = eligibility(u, units)
        if elig == "blocked":
            u.state = UnitState.BLOCKED
            master_log("BLOCKED", u, reason="dependency-failed")
            continue
        if elig == "waiting":
            continue
        # fast-path: already complete on disk (resume / sibling produced it)
        if is_unit_complete(exp, u):
            u.state = UnitState.DONE
            continue
        candidates.append(u)

    if not candidates:
        return None
    # SETUP first, then seed-major within canonical method order, FINALIZE last.
    return min(candidates, key=_unit_sort_key)


# --- execution ---------------------------------------------------------------
def build_argv(exp: Experiment, u: Unit) -> list[str]:
    base = [sys.executable, "-m", "agentic_autorag_bench.cli", "pareto", "-c", str(exp.config_path)]
    if u.kind is UnitKind.SETUP:
        return base + ["--setup-only"]
    if u.kind is UnitKind.METHOD:
        return base + ["--methods", u.method, "--seed", str(u.seed), "--resume"]
    if u.kind is UnitKind.FINALIZE:
        return base + ["--figure-only"]
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


def handle_completion(u: Unit, fut: Future, exp: Experiment,
                      units: dict[str, Unit], max_attempts: int, base: float, cap: float) -> None:
    u.finished_at = now()
    exc = fut.exception()
    if exc is not None:
        record_failure_or_retry(u, f"thread:{exc}", max_attempts, base, cap)
        return
    u.returncode = fut.result()

    # Disk state is authoritative for every kind (exit code is not — the pareto
    # engine can exit 0 while a cell threw internally, and FINALIZE writes
    # hypervolume.json only when there are plottable trials).
    if is_unit_complete(exp, u):
        if u.kind is UnitKind.FINALIZE:
            missing = [k for k in u.deps if units[k].state is not UnitState.DONE]
            if missing:
                master_log("INCOMPLETE_MATRIX", u, missing=",".join(missing))
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


def write_status(units: dict[str, Unit], exp: Experiment, workers: int) -> None:
    counts = {s.value: 0 for s in UnitState}
    for u in units.values():
        counts[u.state.value] += 1
    def _started(u: Unit) -> str | None:
        if not u.started_at:
            return None
        return datetime.fromtimestamp(u.started_at, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    running = [
        {"key": u.key, "attempt": u.attempts,
         "started_at": _started(u),
         "log": str(u.log_path) if u.log_path else None}
        for u in units.values() if u.state is UnitState.RUNNING
    ]
    methods_done = sum(
        1 for u in units.values() if u.kind is UnitKind.METHOD and u.state is UnitState.DONE
    )
    methods_total = sum(1 for u in units.values() if u.kind is UnitKind.METHOD)
    eta = _eta_seconds(units, workers)
    payload = {
        "updated_at": ts(),
        "pid": os.getpid(),
        "workers": workers,
        "output_root": str(exp.output_root),
        "max_trials": exp.max_trials,
        "counts": counts,
        "setup_done": units[SETUP_KEY].state is UnitState.DONE,
        "methods_done": f"{methods_done}/{methods_total}",
        "finalized": units[FINALIZE_KEY].state is UnitState.DONE,
        "eta_hours": round(eta / 3600, 2) if eta is not None else None,
        "running": running,
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
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)

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
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    pool.shutdown(wait=False)


# --- supervisor --------------------------------------------------------------
def supervise(units: dict[str, Unit], exp: Experiment, *,
              workers: int, max_attempts: int, base: float, cap: float) -> int:
    pool = ThreadPoolExecutor(max_workers=workers)
    in_flight: dict[Future, Unit] = {}
    try:
        while True:
            if STATE.shutdown:
                break
            # fill free slots
            while len(in_flight) < workers:
                u = select_next(units, exp)
                if u is None:
                    break
                u.state = UnitState.RUNNING
                u.attempts += 1
                u.started_at = now()
                argv = build_argv(exp, u)
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
                write_status(units, exp, workers)
                continue

            done, _ = wait(list(in_flight), timeout=STATUS_FLUSH_S, return_when=FIRST_COMPLETED)
            for fut in done:
                u = in_flight.pop(fut)
                handle_completion(u, fut, exp, units, max_attempts, base, cap)
            write_status(units, exp, workers)

        if STATE.shutdown:
            graceful_drain(pool, in_flight)
            # interrupted RUNNING units revert to PENDING for a clean resume
            for u in units.values():
                if u.state is UnitState.RUNNING:
                    u.state = UnitState.PENDING
            write_status(units, exp, workers)
            master_log("SHUTDOWN")
            return 130
    finally:
        pool.shutdown(wait=False)

    write_status(units, exp, workers)
    failed = [u.key for u in units.values() if u.state in (UnitState.FAILED, UnitState.BLOCKED)]
    if failed:
        master_log("COMPLETE_WITH_FAILURES", n=len(failed), failed=",".join(failed))
        return 1
    master_log("COMPLETE_OK")
    return 0


# --- dry run -----------------------------------------------------------------
def dry_run(units: dict[str, Unit], exp: Experiment) -> int:
    print(f"REPO_ROOT   = {REPO_ROOT}")
    print(f"config      = {exp.config_path}")
    print(f"output_root = {exp.output_root}")
    print(f"shared_cache= {exp.shared_cache_dir}")
    print(f"max_trials  = {exp.max_trials}")
    print(f"methods     = {exp.methods}")
    print(f"seeds       = {exp.seeds}")
    complete = sum(1 for u in units.values() if is_unit_complete(exp, u))
    kinds = {k: 0 for k in UnitKind}
    for u in units.values():
        kinds[u.kind] += 1
    print("\nUnit counts: " + ", ".join(f"{k.value}={v}" for k, v in kinds.items())
          + f"  (total={len(units)}, already-complete-on-disk={complete})")
    print("\nDAG (deps in parentheses):")
    for u in sorted(units.values(), key=_unit_sort_key):
        deps = f"  <- ({', '.join(u.deps)})" if u.deps else ""
        done = "  [done-on-disk]" if is_unit_complete(exp, u) else ""
        print(f"  {u.kind.value:9s} {u.key}{deps}{done}")
    print("\nExample argv per kind:")
    seen: set[UnitKind] = set()
    for u in sorted(units.values(), key=_unit_sort_key):
        if u.kind not in seen:
            seen.add(u.kind)
            print("  " + " ".join(build_argv(exp, u)))
    print("\n(dry-run: nothing executed)")
    return 0


# --- cli ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment-2 (UniDoc Pareto) scheduler (2-worker, resume-safe).")
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"pareto entry config (default: {DEFAULT_CONFIG})")
    p.add_argument("--methods", default=None,
                   help="comma-separated method subset (default: the config's methods list)")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed subset (default: the config's seeds list)")
    p.add_argument("--workers", type=int, default=2, help="max concurrent units (endpoint-safe ceiling is 2)")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--backoff-base", type=float, default=60.0)
    p.add_argument("--backoff-cap", type=float, default=900.0)
    p.add_argument("--dry-run", action="store_true", help="print the DAG and exit without executing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)  # ParetoConfig.load resolves output_root against CWD
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    exp = load_experiment(args.config, methods, seeds)
    units = build_units(exp)

    if args.dry_run:
        return dry_run(units, exp)

    if args.workers > 2:
        master_log("WARN", note=f"workers={args.workers} exceeds the endpoint-safe ceiling of 2")

    install_signal_handlers()
    master_log("LAUNCH", config=str(exp.config_path), workers=args.workers,
               methods=",".join(exp.methods), seeds=",".join(map(str, exp.seeds)), units=len(units))
    return supervise(
        units, exp,
        workers=args.workers, max_attempts=args.max_attempts,
        base=args.backoff_base, cap=args.backoff_cap,
    )


if __name__ == "__main__":
    sys.exit(main())
