"""Optuna MO-TPE baseline (syftr-style).

The optimizer syftr (Conway et al., AutoML-Conf 2025) runs: group-decomposed
multivariate TPE with ``constant_liar``, single-objective in ``maximize`` mode
for the accuracy experiment and two-objective (accuracy ↑, per-query LLM cost ↓)
for the Pareto experiment. We replicate the *sampler* inside the bench harness so
the comparison is fair by construction — identical search space, exam, evaluator,
budget and seeds; only the proposer differs (YAHPO/HPOBench standard).

``group=True`` is mandatory, not optional: our search space is define-by-run
conditional (``hybrid_alpha`` only under hybrid+alpha, ``reranker_top_n`` gated on
a reranker, graph fields skipped off-graph, per-trial dynamic int bounds). Optuna's
plain ``multivariate=True, group=False`` does not support dynamic search spaces;
``group=True`` (which forces multivariate) decomposes the space into the sub-spaces
each trial actually visits. This is syftr's ``HierarchicalTPESampler`` intent.

Single- vs multi-objective is config-driven (``meta.cost_aware``), mirroring how
the agentic optimizer flips ``agentic_score`` → ``agentic_cost`` on the same flag.

Ask-and-tell loop so the async ``evaluator`` integrates naturally — pruned trials
don't block subsequent suggestions. Sampler state is pickled per-trial so a
kill/resume reproduces the unattended sequence (Optuna's default storage doesn't
persist sampler state).
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agentic_autorag.config.models import ProjectConfig, TrialConfig
from agentic_autorag.output_layout import RunLayout

from agentic_autorag_bench.methods._logging import log_trial_banner
from agentic_autorag_bench.methods._sampler import config_to_optuna_params, sample_optuna
from agentic_autorag_bench.types import Budget, Evaluator, HistoryEntry, SearchResult

logger = logging.getLogger("agentic_autorag_bench.run")

_STUDY_NAME = "agentic_autorag_bench_motpe_v1"
_DB_NAME = "optuna.db"
_SAMPLER_PICKLE = "optuna_sampler.pkl"
_WALL_CLOCK_NAME = "wall_clock.json"

# Transfer-warm-start (syftr's transfer-learning, Conway et al., AutoML-Conf
# 2025). ``motpe_warm`` injects the paired ``random`` run's COMPLETED trials —
# config params AND their already-known objective value(s) — into the study as a
# FREE, UNCOUNTED prior via ``create_trial`` + ``add_trial`` (NOT ``enqueue_trial``,
# which would re-evaluate). The prior informs TPE's surrogate but is NOT part of
# the optimization budget: the method then runs the FULL ``budget.max_trials``
# optimization trials as normal TPE. Because the prior count (~budget.max_trials)
# already exceeds ``n_startup_trials``, TPE is model-guided from the first
# optimization trial. This is syftr's transfer-learning at our 40-trial scale,
# isolating the value of the warm-start PRIOR from TPE itself.
#
# The injected prior is invisible to selection, figures and downstream transfer:
# every prior trial is tagged ``transfer_prior=True`` and excluded everywhere
# history is reconstructed/selected, so ``history.jsonl`` for ``motpe_warm`` holds
# exactly ``budget.max_trials`` genuine TPE optimization trials.
TRANSFER_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
KMEANS_RANDOM_STATE = 0
_TRANSFER_SEED_META_NAME = "transfer_seed_meta.json"

# Optuna ``user_attr`` flag marking an injected free-prior trial. Read by the
# resume self-heal (``_reconstruct_history_from_optuna``) to exclude the prior
# from the method's recorded history, so the prior never leaks into selection,
# learning curves, @k data or downstream transfer.
_TRANSFER_PRIOR_ATTR = "transfer_prior"

# Cap on how many random trials we inject as the free prior — mirrors syftr's
# ``max_total``. When the paired random run has <= MAX_TRANSFER_PRIOR completed
# trials (the normal case at our 40-trial scale) we inject ALL of them with NO
# embedding and NO clustering, exactly as syftr does when candidates <= max_total.
# Only when the count EXCEEDS this cap do we fall back to bge-large KMeans
# best-per-cluster downsampling to MAX_TRANSFER_PRIOR. At <=100 the embedder is
# never loaded.
MAX_TRANSFER_PRIOR = 100

# syftr's published sampler configuration (syftr/optuna_helper.py). Exposed as a
# dict so the behavioral-equivalence test can assert our construction matches it
# without poking at TPESampler internals. ``n_startup_trials`` is Optuna's library
# default (10); syftr's 100 is sized for its 1000-trial budget — not comparable at
# 40 trials. Disclosed in the paper; identical across Exp 1 / Exp 2 and all seeds.
N_STARTUP_TRIALS = 10
SAMPLER_KWARGS: dict[str, object] = {
    "multivariate": True,
    "group": True,
    "constant_liar": True,
    "n_startup_trials": N_STARTUP_TRIALS,
}

# Cap on infeasible ask/tell-PRUNED rounds per budget slot before giving up.
# Optuna excludes PRUNED trials from TPE's KDE fit (no objective value to place
# them on the good/bad split), so the surrogate is unchanged. The retry
# terminates because ``ask()`` advances the sampler's internal RNG, yielding a
# different draw each call. For narrow infeasibilities this converges in a few
# attempts; 1000 is paranoid.
MAX_RESAMPLE_ATTEMPTS = 1000


def _make_sampler(seed: int | None):
    import optuna

    return optuna.samplers.TPESampler(seed=seed, **SAMPLER_KWARGS)


class MissingTransferSource(Exception):
    """The paired ``random`` history needed to warm-start ``motpe_warm`` is
    absent or has ZERO completed trials.

    Raised by ``MOTPESearch`` in transfer-warm mode so the matrix runner can
    SKIP this (dataset, seed) cell with a clear warning rather than aborting:
    ``random`` simply has not been run for this cell yet.
    """


@dataclass(frozen=True)
class _TransferSeed:
    """One config carried over from the random source, with its provenance.

    ``random_mean_llm_cost_per_query_usd`` is carried so a two-objective
    (``cost_aware``) study can inject the prior with BOTH objective values.
    """

    config: dict
    random_trial_number: int
    random_answer_accuracy: float
    random_mean_llm_cost_per_query_usd: float


def _default_config_embedder() -> Callable[[str], list[float]]:
    """Load the bge-large embedder and return a config-JSON → vector function.

    Mirrors syftr's ``get_embedding_model(...).get_query_embedding(text)``: each
    flow's config JSON is embedded with ``BAAI/bge-large-en-v1.5`` so KMeans
    clusters configs by structural similarity. Uses the same SentenceTransformer
    path the framework's index builder uses for HF embedders (no hand-rolled
    HTTP), loaded lazily so importing this module never pulls torch/network.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TRANSFER_EMBED_MODEL)

    def embed(text: str) -> list[float]:
        return [float(x) for x in model.encode(text)]

    return embed


def _load_random_history(source_history_path: Path) -> list[HistoryEntry]:
    """Read the paired ``random`` run's completed-trial history."""
    if not source_history_path.exists():
        raise MissingTransferSource(
            f"random history not found at {source_history_path}; run the random "
            "method for this dataset+seed before motpe_warm"
        )
    return _load_history(source_history_path)


def _dedupe_random_history(random_history: list[HistoryEntry]) -> dict[str, _TransferSeed]:
    """Deduplicate the random run by config, keeping the best-accuracy occurrence
    of each so a config that random happened to evaluate twice is injected once."""
    best_by_config: dict[str, _TransferSeed] = {}
    for entry in random_history:
        key = json.dumps(entry.config, sort_keys=True)
        prior = best_by_config.get(key)
        if prior is None or entry.answer_accuracy > prior.random_answer_accuracy:
            best_by_config[key] = _TransferSeed(
                config=entry.config,
                random_trial_number=entry.trial_number,
                random_answer_accuracy=entry.answer_accuracy,
                random_mean_llm_cost_per_query_usd=entry.mean_llm_cost_per_query_usd,
            )
    return best_by_config


def _select_transfer_seeds(
    random_history: list[HistoryEntry],
    *,
    n_seeds: int,
    embed: Callable[[str], list[float]],
) -> list[_TransferSeed]:
    """Downsample the random run to ``n_seeds`` diverse, strong configs.

    Only invoked as a fallback when the distinct-config count EXCEEDS ``n_seeds``
    (= ``MAX_TRANSFER_PRIOR``). Serializes each distinct random-trial config to
    JSON, embeds it with bge-large, KMeans over the embeddings
    (``n_clusters=n_seeds``), and keeps the MAX-accuracy member of each cluster —
    diverse (one per cluster) yet strong (best in cluster). At our 40-trial scale
    the distinct count is always <= ``n_seeds`` so this — and the embedder — is
    never reached; ``_collect_transfer_prior`` short-circuits to all-of-them.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    best_by_config = _dedupe_random_history(random_history)
    keys = list(best_by_config.keys())
    embeddings = np.array([embed(key) for key in keys])
    labels = KMeans(n_clusters=n_seeds, random_state=KMEANS_RANDOM_STATE).fit_predict(embeddings)

    selected: list[_TransferSeed] = []
    for cluster in range(n_seeds):
        members = [best_by_config[keys[i]] for i in range(len(keys)) if labels[i] == cluster]
        if members:
            selected.append(max(members, key=lambda s: s.random_answer_accuracy))
    return selected


def _collect_transfer_prior(
    random_history: list[HistoryEntry],
    *,
    max_prior: int,
    embed: Callable[[str], list[float]] | None,
) -> tuple[list[_TransferSeed], bool]:
    """Pick the free-prior trials to inject, gating the embedder behind the cap.

    Returns ``(seeds, downsampled)``. When the distinct-config count is
    ``<= max_prior`` (the normal case at our scale) ALL distinct configs are
    injected with NO embedding and NO clustering, and ``downsampled`` is False —
    the embedder is never loaded. Only when the count EXCEEDS ``max_prior`` do we
    load bge-large and KMeans-downsample to ``max_prior`` (``downsampled`` True).

    Raises ``MissingTransferSource`` when the random run has zero completed
    trials, so the matrix runner SKIPs the cell.
    """
    best_by_config = _dedupe_random_history(random_history)
    if not best_by_config:
        raise MissingTransferSource(
            "random run has zero completed trials; motpe_warm has no prior to inject"
        )
    if len(best_by_config) <= max_prior:
        return list(best_by_config.values()), False

    # Only here do we need the embedder. Load lazily if the caller did not inject
    # a deterministic one (tests pass a network-free embed).
    embed_fn = embed or _default_config_embedder()
    seeds = _select_transfer_seeds(random_history, n_seeds=max_prior, embed=embed_fn)
    return seeds, True


def _best_prior_values(seeds: list[_TransferSeed], *, cost_aware: bool) -> dict[str, float]:
    """The prior's BEST objective value(s), for disclosure: if the optimization
    trials fail to beat this, the warm start dominated. Accuracy is the max over
    the prior; in cost-aware mode we also record the cost OF that best-accuracy
    config (the single point selection would pick)."""
    best = max(seeds, key=lambda s: s.random_answer_accuracy)
    values: dict[str, float] = {"answer_accuracy": best.random_answer_accuracy}
    if cost_aware:
        values["mean_llm_cost_per_query_usd"] = best.random_mean_llm_cost_per_query_usd
    return values


def _write_transfer_seed_meta(
    path: Path,
    seeds: list[_TransferSeed],
    *,
    downsampled: bool,
    cost_aware: bool,
) -> None:
    """Record that the prior is a FREE/uncounted warm start, how many trials were
    injected, whether KMeans downsampling fired, and the prior's BEST objective
    value(s) — so the paper can disclose if the optimization trials failed to beat
    the warm start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "design": "free_prior",
                "prior_is_free_uncounted": True,
                "n_injected": len(seeds),
                "downsampled": downsampled,
                "max_transfer_prior": MAX_TRANSFER_PRIOR,
                "transfer_embed_model": TRANSFER_EMBED_MODEL if downsampled else None,
                "kmeans_random_state": KMEANS_RANDOM_STATE if downsampled else None,
                "prior_best": _best_prior_values(seeds, cost_aware=cost_aware),
                "seeds": [
                    {
                        "random_trial_number": s.random_trial_number,
                        "random_answer_accuracy": s.random_answer_accuracy,
                        "random_mean_llm_cost_per_query_usd": s.random_mean_llm_cost_per_query_usd,
                        "config": s.config,
                    }
                    for s in seeds
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@dataclass
class MOTPESearch:
    """Group-decomposed multivariate MO-TPE over the project's SearchSpace.

    Single-objective (``direction="maximize"``) when ``project.meta.cost_aware``
    is false; two-objective (``directions=["maximize", "minimize"]`` over
    ``answer_accuracy`` and ``mean_llm_cost_per_query_usd``) when it is true.

    Each ``budget.max_trials`` slot is guaranteed to evaluate a
    validation-passing config: an invalid suggestion is reported back to Optuna
    via ``tell(state=PRUNED)`` (so TPE's surrogate learns to avoid that region)
    and the slot is re-asked, up to ``MAX_RESAMPLE_ATTEMPTS`` times. This keeps
    the budget comparable to the agentic baseline, whose Proposer is
    constraint-aware. ``extras`` surfaces ``n_validation_rejects`` and
    ``n_pruned`` so the paper can report how often TPE proposed infeasible points.
    """

    project: ProjectConfig
    storage_dir: Path
    resume: bool = False
    name: str = "motpe"
    deterministic: bool = False
    # Transfer-warm-start: when true, the paired ``random`` cell's completed
    # trials (from ``transfer_source_dir``) are injected as a FREE, uncounted
    # prior before the FULL ``budget.max_trials`` optimization trials run.
    # ``config_embedder`` is injectable so the (rare, >MAX_TRANSFER_PRIOR)
    # downsampling path can use a deterministic, network-free embedding in tests.
    warm_transfer: bool = False
    transfer_source_dir: Path | None = None
    config_embedder: Callable[[str], list[float]] | None = field(default=None, repr=False)

    async def search(
        self,
        evaluator: Evaluator,
        budget: Budget,
        *,
        seed: int | None = None,
    ) -> SearchResult:
        import optuna
        from optuna.trial import TrialState

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if budget.max_trials is None:
            raise ValueError("MO-TPE search requires budget.max_trials")

        cost_aware = bool(self.project.meta.cost_aware)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.storage_dir / _DB_NAME
        sampler_path = self.storage_dir / _SAMPLER_PICKLE
        history_path = RunLayout(base=self.storage_dir).history
        wall_clock_path = self.storage_dir / _WALL_CLOCK_NAME

        # Wiping stale state on a fresh start is the bench-level ``--clean``
        # flag's job (``_clear_output_root_for`` in run.py). We deliberately do
        # NOT wipe here, even when ``resume=False``: a user who passes
        # ``--no-clean`` without ``--resume`` keeps whatever was on disk.

        sampler: optuna.samplers.BaseSampler
        if self.resume and sampler_path.exists():
            try:
                sampler = pickle.loads(sampler_path.read_bytes())
            except Exception:
                logger.warning("Failed to load pickled sampler; starting fresh", exc_info=True)
                sampler = _make_sampler(seed)
        else:
            sampler = _make_sampler(seed)

        if cost_aware:
            study = optuna.create_study(
                directions=["maximize", "minimize"],
                sampler=sampler,
                storage=f"sqlite:///{db_path}",
                study_name=_STUDY_NAME,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                storage=f"sqlite:///{db_path}",
                study_name=_STUDY_NAME,
                load_if_exists=True,
            )

        history: list[HistoryEntry] = []
        prior_wall_s = 0.0
        if self.resume:
            # A SIGKILL/Ctrl+C between ``study.ask()`` and ``study.tell()`` leaves a
            # RUNNING trial in sqlite that TPE would never fit on. Mark them
            # FAIL so resumed asks generate fresh trial ids without leftover
            # noise in ``study.get_trials()``. ``study.tell`` takes a trial NUMBER
            # (or a live Trial), NOT the FrozenTrial that ``get_trials`` returns —
            # pass ``t.number`` or it raises "Trial must be a trial object or
            # trial number" and aborts the whole resume.
            for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                study.tell(t.number, state=TrialState.FAIL)

            if history_path.exists():
                history = _load_history(history_path)
            else:
                history = _reconstruct_history_from_optuna(study, self.project, self.storage_dir)
                for entry in history:
                    _append_history(history_path, entry)
                if history:
                    logger.warning(
                        "Reconstructed %d trial(s) into a fresh history.jsonl from "
                        "optuna.db + trial_cost_ledger.jsonl. Per-trial ``metrics`` "
                        "other than accuracy default to 0 for these trials.",
                        len(history),
                    )

            if wall_clock_path.exists():
                try:
                    raw = json.loads(wall_clock_path.read_text(encoding="utf-8"))
                    prior_wall_s = float(raw.get("wall_clock_s", 0.0))
                except Exception:
                    logger.warning("Could not parse %s; wall-clock starts at 0", wall_clock_path, exc_info=True)
            logger.info(
                "Resuming MO-TPE search from trial %d/%d (prior wall=%.1fs)",
                len(history) + 1,
                budget.max_trials,
                prior_wall_s,
            )

        # Transfer-warm-start: inject the paired ``random`` cell's completed
        # trials as a FREE, uncounted prior (config params + already-known
        # objective value(s), tagged ``transfer_prior``). Only on a fresh start —
        # on resume the prior is already in optuna.db (and is excluded from the
        # reconstructed history by its ``transfer_prior`` tag).
        if self.warm_transfer and not self.resume:
            self._inject_transfer_prior(study, cost_aware=cost_aware)

        trial_usd_total = sum(h.eval_usd for h in history)
        optimizer_usd = 0.0
        n_validation_rejects = 0
        n_pruned = 0

        t_start = time.monotonic()
        for trial_num in range(len(history) + 1, budget.max_trials + 1):
            trial = None
            config = None
            for _ in range(MAX_RESAMPLE_ATTEMPTS):
                candidate_trial = study.ask()
                try:
                    candidate = sample_optuna(candidate_trial, self.project.search_space)
                except optuna.TrialPruned as exc:
                    logger.debug("trial %d resample pruned: %s", trial_num, exc)
                    study.tell(candidate_trial, state=optuna.trial.TrialState.PRUNED)
                    n_pruned += 1
                    continue

                violations = self.project.validate_trial(candidate)
                if not violations:
                    trial = candidate_trial
                    config = candidate
                    break
                logger.debug("trial %d resample rejected: %s", trial_num, "; ".join(violations))
                study.tell(candidate_trial, state=optuna.trial.TrialState.PRUNED)
                n_validation_rejects += 1

            if trial is None or config is None:
                raise RuntimeError(
                    f"MO-TPE search could not find a valid config after "
                    f"{MAX_RESAMPLE_ATTEMPTS} Optuna ask/tell rounds on trial "
                    f"{trial_num}; the search space's feasible volume is near zero."
                )

            log_trial_banner(logger, trial_num, budget.max_trials, config)

            try:
                result = await evaluator(config)
            except Exception:
                logger.exception("trial %d evaluation failed; marking failed", trial_num)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue

            if cost_aware:
                study.tell(trial, [result.answer_accuracy, result.mean_llm_cost_per_query_usd])
            else:
                study.tell(trial, result.answer_accuracy)
            entry = HistoryEntry(
                trial_number=trial_num,
                config=config.to_prompt_dump(include_graph=self.project.uses_graph()),
                answer_accuracy=result.answer_accuracy,
                metrics=result.metrics,
                eval_usd=result.eval_usd,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                embedding_tokens=result.embedding_tokens,
                mean_llm_cost_per_query_usd=result.mean_llm_cost_per_query_usd,
            )
            history.append(entry)
            trial_usd_total += result.eval_usd

            try:
                sampler_path.write_bytes(pickle.dumps(study.sampler))
            except Exception:
                logger.warning("Failed to persist Optuna sampler", exc_info=True)
            _append_history(history_path, entry)
            cumulative = prior_wall_s + (time.monotonic() - t_start)
            wall_clock_path.write_text(json.dumps({"wall_clock_s": cumulative}), encoding="utf-8")

            best = max(h.answer_accuracy for h in history)
            logger.info(
                "%s trial %d done | accuracy=%.3f | best so far=%.3f",
                self.name,
                trial_num,
                result.answer_accuracy,
                best,
            )
            logger.info("")

        if not history:
            raise RuntimeError("MO-TPE search produced no successful trials")

        # best_config is the max-accuracy point in both modes — matching what the
        # agentic_cost optimizer picks for its held-out eval. The Pareto figure
        # reads the whole frontier from history separately (per-query cost is on
        # every HistoryEntry).
        best_entry = max(history, key=lambda h: h.answer_accuracy)
        total_wall = prior_wall_s + (time.monotonic() - t_start)
        return SearchResult(
            method=self.name,
            seed=seed,
            deterministic=self.deterministic,
            best_config=best_entry.config,
            history=history,
            optimizer_usd=optimizer_usd,
            trial_usd_total=trial_usd_total,
            wall_clock_s=total_wall,
            prompt_tokens=sum(h.prompt_tokens for h in history),
            completion_tokens=sum(h.completion_tokens for h in history),
            embedding_tokens=sum(h.embedding_tokens for h in history),
            extras={
                "n_validation_rejects": n_validation_rejects,
                "n_pruned": n_pruned,
                "cost_aware": cost_aware,
                "warm_transfer": self.warm_transfer,
                "study_name": _STUDY_NAME,
                "storage": str(db_path),
            },
        )

    def _inject_transfer_prior(self, study, *, cost_aware: bool) -> None:
        """Inject the paired ``random`` cell's completed trials as a FREE prior.

        Reads the paired ``random`` run's history from ``transfer_source_dir``,
        collects the prior (ALL distinct configs when count <= MAX_TRANSFER_PRIOR
        with NO embedder; KMeans-downsampled to MAX_TRANSFER_PRIOR otherwise),
        then for EACH prior config builds a COMPLETE Optuna trial carrying its
        config params AND its already-known objective value(s) via
        ``create_trial`` + ``study.add_trial`` (NOT ``enqueue_trial`` — that would
        re-evaluate). Each injected trial is tagged ``transfer_prior`` so it is
        excluded everywhere history is reconstructed/selected. Writes
        ``transfer_seed_meta.json`` for provenance. Raises ``MissingTransferSource``
        if the random run is absent or has zero completed trials.

        Single-objective (Exp1): ``value = answer_accuracy``. Multi-objective
        (Exp2, ``cost_aware``): ``values = (accuracy, mean_llm_cost_per_query_usd)``
        matching the study directions ``[maximize, minimize]``.
        """
        import optuna

        if self.transfer_source_dir is None:
            raise MissingTransferSource("motpe_warm requires transfer_source_dir to be set")

        source_history_path = RunLayout(base=self.transfer_source_dir).history
        random_history = _load_random_history(source_history_path)
        seeds, downsampled = _collect_transfer_prior(
            random_history,
            max_prior=MAX_TRANSFER_PRIOR,
            embed=self.config_embedder,
        )

        ss = self.project.search_space
        for s in seeds:
            cfg = TrialConfig(**s.config)
            params, distributions = self._params_and_distributions(cfg, ss)
            if cost_aware:
                created = optuna.trial.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    values=[s.random_answer_accuracy, s.random_mean_llm_cost_per_query_usd],
                    params=params,
                    distributions=distributions,
                    user_attrs={_TRANSFER_PRIOR_ATTR: True},
                )
            else:
                created = optuna.trial.create_trial(
                    state=optuna.trial.TrialState.COMPLETE,
                    value=s.random_answer_accuracy,
                    params=params,
                    distributions=distributions,
                    user_attrs={_TRANSFER_PRIOR_ATTR: True},
                )
            study.add_trial(created)

        _write_transfer_seed_meta(
            RunLayout(base=self.storage_dir).details / _TRANSFER_SEED_META_NAME,
            seeds,
            downsampled=downsampled,
            cost_aware=cost_aware,
        )
        logger.info(
            "motpe_warm: injected %d FREE-prior trial(s) from %s (downsampled=%s; "
            "best prior accuracy=%.3f). TPE then runs the full optimization budget on top.",
            len(seeds),
            source_history_path,
            downsampled,
            max(s.random_answer_accuracy for s in seeds),
        )

    @staticmethod
    def _params_and_distributions(cfg: TrialConfig, search_space):
        """Recover the Optuna ``params`` AND ``distributions`` a define-by-run
        ``sample_optuna`` would have recorded for ``cfg``.

        ``create_trial`` needs both, and for a conditional define-by-run space the
        active distributions depend on the config's branch. We replay ``cfg``
        through ``sample_optuna`` on an ephemeral enqueued trial: that populates
        ``trial.params`` and ``trial.distributions`` with exactly the active dims
        (and only those), which we read back. No evaluation happens — the
        ephemeral study is discarded.
        """
        import optuna

        params_in = config_to_optuna_params(cfg, search_space)
        ephemeral = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        ephemeral.enqueue_trial(params_in, skip_if_exists=False)
        trial = ephemeral.ask()
        sample_optuna(trial, search_space)
        return dict(trial.params), dict(trial.distributions)


def _load_history(path: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(HistoryEntry(**data))
    return entries


def _append_history(path: Path, entry: HistoryEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict()) + "\n")


def _reconstruct_history_from_optuna(study, project, storage_dir: Path) -> list[HistoryEntry]:
    """Rebuild a ``HistoryEntry`` list from ``optuna.db`` + ``trial_cost_ledger.jsonl``.

    Used as a self-heal path on ``--resume`` when ``history.jsonl`` is missing.
    Each completed Optuna trial's stored params + objective give the
    ``TrialConfig`` (via ``sample_optuna`` replay) and ``answer_accuracy``
    (objective 0 in both single- and multi-objective mode);
    ``trial_cost_ledger.jsonl`` gives per-trial cost + token totals. ``metrics``
    other than accuracy are not recoverable and default to zero.

    Bench's per-iteration ``trial_num`` (1-indexed, increments only on SUCCESSFUL
    evaluation) corresponds to the sorted order of COMPLETE Optuna trials, NOT to
    ``FrozenTrial.number`` (which also increments on pruned/failed asks).

    ``motpe_warm`` injects the paired ``random`` cell's trials as a FREE prior
    tagged ``transfer_prior``. Those are COMPLETE Optuna trials too, but they are
    NOT optimization trials and must NOT appear in the method's history (else they
    leak into selection, learning curves and @k data, and shift the bench
    ``trial_num`` indexing). They are excluded here by their ``transfer_prior``
    user-attr; cold ``motpe`` trials never carry it, so this path is unchanged
    for cold runs.
    """
    import optuna

    from agentic_autorag_bench.methods._sampler import sample_optuna

    completed = sorted(
        (
            t
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if not t.user_attrs.get(_TRANSFER_PRIOR_ATTR, False)
        ),
        key=lambda t: t.number,
    )
    if not completed:
        return []

    cost_path = RunLayout(base=storage_dir).trial_cost_ledger
    cost_by_trial: dict[int, dict[str, float | int]] = {}
    if cost_path.exists():
        for line in cost_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            trial_num = int(rec.get("trial_number", 0))
            buckets = rec.get("buckets", {})
            agg = {
                "eval_usd": sum(float(b.get("usd", 0.0)) for b in buckets.values()),
                "prompt_tokens": sum(int(b.get("prompt_tokens", 0)) for b in buckets.values()),
                "completion_tokens": sum(int(b.get("completion_tokens", 0)) for b in buckets.values()),
                "embedding_tokens": sum(int(b.get("embedding_input_tokens", 0)) for b in buckets.values()),
            }
            cost_by_trial[trial_num] = agg

    # Multi-objective (cost_aware) studies store the per-query cost as objective-1
    # (``study.tell(trial, [accuracy, mean_llm_cost_per_query_usd])``). Recover it
    # so resumed Pareto trials keep the SAME cost objective the live path records —
    # it is unrecoverable from trial_cost_ledger.jsonl (which holds judge-inclusive
    # bucket totals, not the per-query deploy-time mean). Single-objective studies
    # have no cost to recover; 0.0 is correct there.
    n_objectives = len(study.directions)
    include_graph = project.uses_graph()
    entries: list[HistoryEntry] = []
    for bench_idx, ft in enumerate(completed, start=1):
        try:
            cfg = sample_optuna(ft, project.search_space)
        except Exception:
            logger.warning(
                "Could not reconstruct config for optuna trial %d during resume; skipping",
                ft.number,
                exc_info=True,
            )
            continue
        # ``values[0]`` is accuracy in both single- and multi-objective mode
        # (``FrozenTrial.value`` raises for multi-objective trials).
        score = float(ft.values[0]) if ft.values else 0.0
        cost = float(ft.values[1]) if (n_objectives > 1 and ft.values and len(ft.values) > 1) else 0.0
        agg = cost_by_trial.get(bench_idx, {})
        entries.append(
            HistoryEntry(
                trial_number=bench_idx,
                config=cfg.to_prompt_dump(include_graph=include_graph),
                answer_accuracy=score,
                metrics={
                    "answer_accuracy": score,
                    "mean_em": 0.0,
                    "mean_f1": 0.0,
                    "mean_retrieval_quality": 0.0,
                },
                eval_usd=float(agg.get("eval_usd", 0.0)),
                prompt_tokens=int(agg.get("prompt_tokens", 0)),
                completion_tokens=int(agg.get("completion_tokens", 0)),
                embedding_tokens=int(agg.get("embedding_tokens", 0)),
                mean_llm_cost_per_query_usd=cost,
            )
        )
    return entries
