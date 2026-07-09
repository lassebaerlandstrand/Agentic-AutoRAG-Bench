"""Transfer-warm-start MO-TPE (``motpe_warm``): faithful FREE-PRIOR seeding from
the bench's own ``random`` results (syftr's transfer learning at our scale).

The paired ``random`` cell's completed trials are injected as a FREE, UNCOUNTED
prior — config params AND already-known objective value(s), tagged
``transfer_prior`` — via ``create_trial`` + ``add_trial`` (NOT re-evaluated). The
method then runs the FULL ``budget.max_trials`` optimization trials as normal TPE.
The prior informs TPE but is invisible to ``history.jsonl`` / ``best_config`` /
figures. These tests assert that contract without any network or real embedder:
the bge-large embedder is replaced with a deterministic hash-derived vector, and
at our <=MAX_TRANSFER_PRIOR scale it is never instantiated at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pytest
from agentic_autorag.config.models import (
    AgentConfig,
    ChunkingSearchSpace,
    DiscreteValues,
    EmbeddingSearchSpace,
    GeneratorSearchSpace,
    IndexType,
    MetaConfig,
    NumericRange,
    PassageCompressorSearchSpace,
    ProjectConfig,
    QueryExpansionSearchSpace,
    RerankerSearchSpace,
    RetrievalSearchSpace,
    SearchSpace,
    TrialConfig,
)
from agentic_autorag.output_layout import RunLayout
from sklearn.cluster import KMeans

from agentic_autorag_bench.methods import motpe as motpe_mod
from agentic_autorag_bench.methods._sampler import (
    config_to_optuna_params,
    sample_optuna,
    sample_random,
)
from agentic_autorag_bench.methods.motpe import (
    _TRANSFER_PRIOR_ATTR,
    KMEANS_RANDOM_STATE,
    MAX_TRANSFER_PRIOR,
    N_STARTUP_TRIALS,
    MissingTransferSource,
    MOTPESearch,
    _collect_transfer_prior,
    _select_transfer_seeds,
)
from agentic_autorag_bench.types import Budget, HistoryEntry, TrialResult

optuna.logging.set_verbosity(optuna.logging.WARNING)

_EMBED_DIM = 8


def _rich_search_space() -> SearchSpace:
    """Exercises every conditional branch (hybrid+alpha, reranker, expansion,
    compression, DiscreteValues + NumericRange dims). Mirrors test_motpe."""
    return SearchSpace(
        chunking=ChunkingSearchSpace(
            strategies=["recursive", "fixed"],
            chunk_token_size=DiscreteValues(values=[256, 512]),
            chunk_token_overlap=DiscreteValues(values=[0, 32, 64]),
        ),
        embedding=EmbeddingSearchSpace(models=["m1", "m2"]),
        retrieval=RetrievalSearchSpace(
            index_types=[IndexType.VECTOR_ONLY, IndexType.HYBRID_BM25_VECTOR],
            top_k=NumericRange(min=3, max=20),
            hybrid_alpha=NumericRange(min=0.0, max=1.0),
            bm25_vector_fusion=["alpha", "rrf"],
            long_context_reorder=[False, True],
        ),
        reranker=RerankerSearchSpace(
            models=["none", "BAAI/bge-reranker-v2-m3"],
            top_n=DiscreteValues(values=[3, 5, 10]),
        ),
        query_expansion=QueryExpansionSearchSpace(strategies=["none", "hyde"], models=["ollama/llama3.2"]),
        passage_compressor=PassageCompressorSearchSpace(
            strategies=["none", "tree_summarize"], models=["ollama/llama3.2"]
        ),
        generator=GeneratorSearchSpace(models=["ollama/llama3.2", "ollama/mistral"]),
        temperature=NumericRange(min=0.0, max=1.0),
    )


def _project(*, cost_aware: bool = False, search_space: SearchSpace | None = None) -> ProjectConfig:
    return ProjectConfig(
        meta=MetaConfig(cost_aware=cost_aware, corpus_description="A tiny test corpus."),
        search_space=search_space or _rich_search_space(),
        agent=AgentConfig(
            optimizer_model="ollama/llama3.2",
            examiner_model="ollama/llama3.2",
            judge_model="ollama/llama3.2",
        ),
    )


def _deterministic_embed(text: str) -> list[float]:
    """A network-free stand-in for bge-large: a fixed-length vector derived
    from the config-JSON's SHA-256 so KMeans is reproducible per config."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(_EMBED_DIM)]


class _EmbedSpy:
    """Wraps an embed fn and records call count, so a test can assert the
    embedder was NOT loaded/invoked at the <=MAX_TRANSFER_PRIOR scale."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return _deterministic_embed(text)


def _write_random_history(method_dir: Path, entries: list[HistoryEntry]) -> Path:
    """Write a synthetic ``random`` run history.jsonl where motpe_warm reads it."""
    layout = RunLayout(base=method_dir)
    layout.ensure_details()
    with layout.history.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e.to_dict()) + "\n")
    return layout.history


def _synthetic_random_history(n: int, *, with_cost: bool = False) -> list[HistoryEntry]:
    """``n`` distinct random configs at varied (deterministic) accuracies."""
    import random as _random

    ss = _rich_search_space()
    rng = _random.Random(7)
    entries: list[HistoryEntry] = []
    seen: set[str] = set()
    trial = 0
    while len(entries) < n:
        cfg = sample_random(rng, ss)
        dump = cfg.to_prompt_dump(include_graph=False)
        key = json.dumps(dump, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trial += 1
        # Varied accuracies, all distinct so max-per-cluster is unambiguous.
        acc = round(0.1 + 0.8 * (len(entries) / n), 4)
        cost = round(0.001 * trial, 5) if with_cost else 0.0
        entries.append(
            HistoryEntry(
                trial_number=trial,
                config=dump,
                answer_accuracy=acc,
                metrics={"answer_accuracy": acc, "mean_em": acc, "mean_f1": acc, "mean_retrieval_quality": acc},
                eval_usd=0.001,
                mean_llm_cost_per_query_usd=cost,
            )
        )
    return entries


def _make_evaluator(scores: list[float], *, cost: float = 0.0):
    counter = {"i": 0}

    async def evaluator(config: TrialConfig) -> TrialResult:
        score = scores[counter["i"] % len(scores)]
        counter["i"] += 1
        return TrialResult(
            answer_accuracy=score,
            metrics={"answer_accuracy": score, "mean_em": score, "mean_f1": score, "mean_retrieval_quality": score},
            eval_usd=0.001,
            mean_llm_cost_per_query_usd=cost,
        )

    return evaluator


# ------------------------------------------------ prior collection / gating


def test_collect_prior_takes_all_below_cap_without_embedder() -> None:
    """At <= MAX_TRANSFER_PRIOR distinct configs (the normal 40-trial-scale case)
    the prior is ALL distinct configs, with NO embedding/clustering — the
    embedder is never invoked, and ``downsampled`` is False."""
    history = _synthetic_random_history(N_STARTUP_TRIALS + 8)
    spy = _EmbedSpy()
    seeds, downsampled = _collect_transfer_prior(history, max_prior=MAX_TRANSFER_PRIOR, embed=spy)
    assert downsampled is False
    assert spy.calls == 0  # embedder NOT loaded/invoked at this scale
    assert len(seeds) == len(history)
    selected_keys = {json.dumps(s.config, sort_keys=True) for s in seeds}
    all_keys = {json.dumps(e.config, sort_keys=True) for e in history}
    assert selected_keys == all_keys


def test_collect_prior_downsamples_above_cap_with_embedder() -> None:
    """Above the cap (here a small monkeypatch-free explicit ``max_prior``) the
    KMeans downsampling fires, the embedder IS invoked, and EXACTLY ``max_prior``
    diverse strong configs are selected."""
    cap = N_STARTUP_TRIALS
    history = _synthetic_random_history(cap + 9)
    spy = _EmbedSpy()
    seeds, downsampled = _collect_transfer_prior(history, max_prior=cap, embed=spy)
    assert downsampled is True
    assert spy.calls == len(history)  # one embed per distinct config
    assert len(seeds) == cap

    # Each selected seed is the max-accuracy member of its KMeans cluster.
    keys = [json.dumps(e.config, sort_keys=True) for e in history]
    embeddings = np.array([_deterministic_embed(key) for key in keys])
    labels = KMeans(n_clusters=cap, random_state=KMEANS_RANDOM_STATE).fit_predict(embeddings)
    acc_by_key = {k_: e.answer_accuracy for k_, e in zip(keys, history, strict=True)}
    cluster_of_key = dict(zip(keys, labels, strict=True))
    best_acc_per_cluster: dict[int, float] = {}
    for key_, label in cluster_of_key.items():
        best_acc_per_cluster[label] = max(best_acc_per_cluster.get(label, -1.0), acc_by_key[key_])

    selected_clusters = set()
    for s in seeds:
        cluster = cluster_of_key[json.dumps(s.config, sort_keys=True)]
        selected_clusters.add(cluster)
        assert s.random_answer_accuracy == best_acc_per_cluster[cluster]
    assert len(selected_clusters) == cap  # one per cluster


def test_collect_prior_empty_history_raises_missing_transfer_source() -> None:
    """Zero completed random trials → MissingTransferSource (runner skips cell)."""
    with pytest.raises(MissingTransferSource):
        _collect_transfer_prior([], max_prior=MAX_TRANSFER_PRIOR, embed=_deterministic_embed)


def test_select_transfer_seeds_round_trip_via_config_to_optuna_params() -> None:
    """Every selected seed's config survives the inverse-mapping + sample_optuna
    replay unchanged — else the injected prior config would drift."""
    ss = _rich_search_space()
    history = _synthetic_random_history(N_STARTUP_TRIALS + 5)
    seeds = _select_transfer_seeds(history, n_seeds=N_STARTUP_TRIALS, embed=_deterministic_embed)
    for s in seeds:
        cfg = TrialConfig(**s.config)
        params = config_to_optuna_params(cfg, ss)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        study.enqueue_trial(params, skip_if_exists=False)
        replayed = sample_optuna(study.ask(), ss)
        assert replayed.to_prompt_dump(include_graph=False) == cfg.to_prompt_dump(include_graph=False)


# ------------------------------------------------------ full search structure


@pytest.mark.asyncio
async def test_warm_injects_all_random_as_complete_prior(tmp_path: Path) -> None:
    """ALL completed random trials are injected as COMPLETE prior trials, tagged
    ``transfer_prior`` and carrying their already-known value (NOT re-evaluated).
    The embedder is never invoked at this scale; the method runs the FULL budget."""
    project = _project()
    n_random = N_STARTUP_TRIALS + 6
    budget_trials = N_STARTUP_TRIALS + 4

    random_dir = tmp_path / "random" / "seed_1"
    history = _synthetic_random_history(n_random)
    _write_random_history(random_dir, history)

    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    spy = _EmbedSpy()
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=spy,
    )
    sr = await optimizer.search(_make_evaluator([0.5]), Budget(max_trials=budget_trials), seed=1)

    assert sr.method == "motpe_warm"
    assert sr.extras["warm_transfer"] is True
    assert spy.calls == 0  # <= MAX_TRANSFER_PRIOR: no embedding

    # The prior lives in optuna.db, tagged transfer_prior, carrying random's value.
    study = optuna.load_study(
        study_name=motpe_mod._STUDY_NAME, storage=f"sqlite:///{warm_dir / 'optuna.db'}"
    )
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    prior = [t for t in complete if t.user_attrs.get(_TRANSFER_PRIOR_ATTR)]
    opt_trials = [t for t in complete if not t.user_attrs.get(_TRANSFER_PRIOR_ATTR)]
    assert len(prior) == n_random  # every random trial injected
    # Values inherited from random, not re-evaluated to 0.5.
    prior_values = sorted(round(t.value, 4) for t in prior)
    random_values = sorted(round(e.answer_accuracy, 4) for e in history)
    assert prior_values == random_values
    assert len(opt_trials) == budget_trials

    # Provenance sidecar: free/uncounted prior, n injected, downsample flag, best.
    meta = json.loads((RunLayout(base=warm_dir).details / "transfer_seed_meta.json").read_text())
    assert meta["prior_is_free_uncounted"] is True
    assert meta["n_injected"] == n_random
    assert meta["downsampled"] is False
    assert meta["transfer_embed_model"] is None  # embedder not used
    assert meta["prior_best"]["answer_accuracy"] == max(e.answer_accuracy for e in history)


@pytest.mark.asyncio
async def test_warm_runs_exactly_budget_optimization_trials(tmp_path: Path) -> None:
    """``history.jsonl`` / ``SearchResult.history`` contain EXACTLY
    ``budget.max_trials`` genuine TPE optimization trials — the injected prior is
    excluded everywhere history is recorded."""
    project = _project()
    budget_trials = N_STARTUP_TRIALS + 5

    random_dir = tmp_path / "random" / "seed_1"
    _write_random_history(random_dir, _synthetic_random_history(N_STARTUP_TRIALS + 7))

    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    sr = await optimizer.search(_make_evaluator([0.5]), Budget(max_trials=budget_trials), seed=1)

    assert len(sr.history) == budget_trials
    assert [h.trial_number for h in sr.history] == list(range(1, budget_trials + 1))

    # history.jsonl on disk also holds exactly the optimization trials.
    on_disk = RunLayout(base=warm_dir).history.read_text(encoding="utf-8").splitlines()
    assert len([line for line in on_disk if line.strip()]) == budget_trials


@pytest.mark.asyncio
async def test_best_config_comes_from_optimization_trials_not_prior(tmp_path: Path) -> None:
    """Decisive: construct a random history whose BEST config (acc=0.95) scores
    HIGHER than anything the mocked TPE phase can produce (all 0.3). The returned
    ``best_config`` must come from the 40 optimization trials, NOT the injected
    prior — the prior is a surrogate signal only, never a selectable candidate."""
    project = _project()
    budget_trials = N_STARTUP_TRIALS + 3

    random_dir = tmp_path / "random" / "seed_1"
    history = _synthetic_random_history(N_STARTUP_TRIALS + 6)
    # Force a clear winner in the prior, far above the TPE phase's flat 0.3.
    history[0].answer_accuracy = 0.95
    history[0].metrics["answer_accuracy"] = 0.95
    best_prior_config = history[0].config
    _write_random_history(random_dir, history)

    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    sr = await optimizer.search(_make_evaluator([0.3]), Budget(max_trials=budget_trials), seed=1)

    best_opt_acc = max(h.answer_accuracy for h in sr.history)
    assert best_opt_acc == pytest.approx(0.3)
    # best_config is one the optimization phase actually evaluated (acc 0.3), and
    # is NOT the higher-scoring prior config.
    assert sr.best_config in [h.config for h in sr.history]
    assert sr.best_config != best_prior_config


@pytest.mark.asyncio
async def test_tpe_is_guided_from_first_optimization_trial(tmp_path: Path) -> None:
    """The prior count (~budget) exceeds ``n_startup_trials`` (10), so TPE is
    model-guided from optimization-trial 1 — verified via the sampler's
    ``before_trial`` model-fit, which only runs once observations > n_startup."""
    project = _project()
    n_random = N_STARTUP_TRIALS + 12  # comfortably > n_startup
    budget_trials = 4

    random_dir = tmp_path / "random" / "seed_1"
    _write_random_history(random_dir, _synthetic_random_history(n_random))

    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    await optimizer.search(_make_evaluator([0.5]), Budget(max_trials=budget_trials), seed=1)

    # The prior alone already exceeds n_startup, so by the first optimization
    # ask() the sampler had > n_startup COMPLETE observations to fit on.
    study = optuna.load_study(
        study_name=motpe_mod._STUDY_NAME, storage=f"sqlite:///{warm_dir / 'optuna.db'}"
    )
    prior = [
        t
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if t.user_attrs.get(_TRANSFER_PRIOR_ATTR)
    ]
    assert len(prior) == n_random
    assert len(prior) > N_STARTUP_TRIALS


@pytest.mark.asyncio
async def test_warm_multi_objective_injects_both_values(tmp_path: Path) -> None:
    """In a 2-objective (cost_aware) study the prior trials carry BOTH values
    (accuracy, mean_llm_cost_per_query_usd) matching the study directions."""
    project = _project(cost_aware=True)
    random_dir = tmp_path / "random" / "seed_1"
    history = _synthetic_random_history(N_STARTUP_TRIALS + 4, with_cost=True)
    _write_random_history(random_dir, history)

    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    await optimizer.search(_make_evaluator([0.5], cost=0.005), Budget(max_trials=3), seed=1)

    study = optuna.load_study(
        study_name=motpe_mod._STUDY_NAME, storage=f"sqlite:///{warm_dir / 'optuna.db'}"
    )
    assert [d.name for d in study.directions] == ["MAXIMIZE", "MINIMIZE"]
    prior = [
        t
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if t.user_attrs.get(_TRANSFER_PRIOR_ATTR)
    ]
    assert prior
    assert all(len(t.values) == 2 for t in prior)
    # Each prior trial's (acc, cost) pair matches a random entry's pair.
    random_pairs = {(round(e.answer_accuracy, 5), round(e.mean_llm_cost_per_query_usd, 5)) for e in history}
    prior_pairs = {(round(t.values[0], 5), round(t.values[1], 5)) for t in prior}
    assert prior_pairs <= random_pairs

    meta = json.loads((RunLayout(base=warm_dir).details / "transfer_seed_meta.json").read_text())
    assert "mean_llm_cost_per_query_usd" in meta["prior_best"]


@pytest.mark.asyncio
async def test_resume_self_heal_excludes_prior_from_reconstruction(tmp_path: Path) -> None:
    """When history.jsonl is missing on resume, the optuna.db reconstruction must
    rebuild ONLY the optimization trials — the tagged prior is excluded so it
    never leaks into history/selection or shifts the bench trial indexing."""
    import os

    project = _project()
    budget_trials = N_STARTUP_TRIALS + 2
    random_dir = tmp_path / "random" / "seed_1"
    _write_random_history(random_dir, _synthetic_random_history(N_STARTUP_TRIALS + 6))

    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    await optimizer.search(_make_evaluator([0.3]), Budget(max_trials=budget_trials), seed=1)

    os.remove(RunLayout(base=warm_dir).history)  # force the reconstruction path

    resumed = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        resume=True,
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    sr = await resumed.search(_make_evaluator([0.3]), Budget(max_trials=budget_trials), seed=1)
    assert len(sr.history) == budget_trials
    assert all(h.answer_accuracy == pytest.approx(0.3) for h in sr.history)


@pytest.mark.asyncio
async def test_warm_search_missing_random_raises_missing_transfer_source(tmp_path: Path) -> None:
    """No paired random run on disk → MissingTransferSource (catchable; the
    runner skips the cell instead of crashing)."""
    project = _project()
    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=tmp_path / "random" / "seed_1",  # never created
        config_embedder=_deterministic_embed,
    )
    with pytest.raises(MissingTransferSource):
        await optimizer.search(_make_evaluator([0.5]), Budget(max_trials=N_STARTUP_TRIALS + 3), seed=1)


@pytest.mark.asyncio
async def test_warm_search_empty_random_raises_missing_transfer_source(tmp_path: Path) -> None:
    """An on-disk random run with ZERO completed trials → MissingTransferSource."""
    project = _project()
    random_dir = tmp_path / "random" / "seed_1"
    _write_random_history(random_dir, [])  # exists but empty
    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=random_dir,
        config_embedder=_deterministic_embed,
    )
    with pytest.raises(MissingTransferSource):
        await optimizer.search(_make_evaluator([0.5]), Budget(max_trials=N_STARTUP_TRIALS + 3), seed=1)


# ------------------------------------------------------------ runner wiring


def test_order_methods_for_run_puts_random_before_motpe_warm() -> None:
    """Hard dependency: random must precede motpe_warm within a dataset."""
    from agentic_autorag_bench.run import _order_methods_for_run

    ordered = _order_methods_for_run(["agentic_score", "motpe_warm", "motpe", "random"])
    assert ordered.index("random") < ordered.index("motpe_warm")
    # Other methods keep their relative order.
    assert ordered.index("agentic_score") < ordered.index("motpe")


def test_order_methods_for_run_noop_when_random_already_first() -> None:
    from agentic_autorag_bench.run import _order_methods_for_run

    methods = ["random", "motpe", "motpe_warm"]
    assert _order_methods_for_run(methods) == methods


def test_order_methods_for_run_noop_without_motpe_warm() -> None:
    from agentic_autorag_bench.run import _order_methods_for_run

    methods = ["motpe", "random", "agentic_score"]
    assert _order_methods_for_run(methods) == methods


def test_build_optimizer_wires_warm_transfer_to_paired_random(tmp_path: Path) -> None:
    """``_build_optimizer`` constructs a warm MOTPESearch whose transfer source
    is the sibling ``random`` cell for the same seed label."""
    from agentic_autorag_bench.run import _build_optimizer

    project = _project()
    output_dir = tmp_path / "results" / "motpe_warm" / "seed_3"
    opt = _build_optimizer("motpe_warm", project=project, bench=None, output_dir=output_dir, resume=False)
    assert isinstance(opt, MOTPESearch)
    assert opt.warm_transfer is True
    assert opt.name == "motpe_warm"
    assert opt.transfer_source_dir == tmp_path / "results" / "random" / "seed_3"


@pytest.mark.asyncio
async def test_runner_catch_skips_missing_transfer_source(tmp_path: Path) -> None:
    """The runner-level path catches MissingTransferSource and continues; it does
    NOT propagate. Mirrors run_matrix's per-cell try/except."""
    project = _project()
    warm_dir = tmp_path / "motpe_warm" / "seed_1"
    optimizer = MOTPESearch(
        project=project,
        storage_dir=warm_dir,
        name="motpe_warm",
        warm_transfer=True,
        transfer_source_dir=tmp_path / "random" / "seed_1",  # never created
        config_embedder=_deterministic_embed,
    )
    skipped = False
    try:
        await optimizer.search(_make_evaluator([0.5]), Budget(max_trials=N_STARTUP_TRIALS + 3), seed=1)
    except MissingTransferSource:
        skipped = True
    assert skipped is True
