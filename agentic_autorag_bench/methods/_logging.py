"""Per-trial console formatting shared by baseline search methods."""

from __future__ import annotations

import logging

from agentic_autorag.config.models import TrialConfig

_BAR = "=" * 60


def log_trial_banner(
    logger: logging.Logger,
    trial_num: int,
    max_trials: int,
    config: TrialConfig,
) -> None:
    logger.info(_BAR)
    logger.info("TRIAL %d/%d", trial_num, max_trials)
    logger.info(_BAR)
    reasoning_tag = " +reasoning" if config.reasoning else ""
    logger.info(
        "Config | chunk=%s strategy=%s embed=%s index=%s top_k=%s reranker=%s llm=%s%s temp=%s",
        config.chunk_token_size,
        config.chunking_strategy,
        config.embedding_model,
        config.index_type.value,
        config.top_k,
        config.reranker,
        config.llm_model,
        reasoning_tag,
        config.temperature,
    )
