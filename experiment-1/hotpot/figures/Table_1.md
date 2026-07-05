# HotpotQA-distractor held-out scores

EM / F1 / Judge columns: mean ± sample SD (ddof=1) across the N held-out evaluations pooled per method — one per seed, plus any extra `replay-holdout` re-evaluations of the winning config. With 3 seeds and no replays, N=3 and the SD is the across-seed standard deviation. Token / cost / wall columns are mean across seeds (search-side, unaffected by hold-out replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to find the winning config).

| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic (Ours) | 0.547 ± 0.007 | 0.711 ± 0.004 | 0.828 ± 0.012 | 3 | 0.692 | 0.839 | 0.397 | 0.984 | 6.28M | 997.2k | 18.08M | $0.7244 | $3.2328 | $3.9572 | 3602s¹ |
| Agentic@10 | 0.540 ± 0.020 | 0.699 ± 0.018 | 0.817 ± 0.017 | 3 | 0.682 | 0.847 | 0.393 | 0.985 | 2.05M | 286.7k | 9.02M | $0.1973 | $1.1159 | $1.3132 | 1201s¹ |
| Agentic@20 | 0.551 ± 0.008 | 0.715 ± 0.011 | 0.831 ± 0.018 | 3 | 0.689 | 0.835 | 0.396 | 0.985 | 4.15M | 641.5k | 11.75M | $0.4305 | $2.2992 | $2.7296 | 2402s¹ |
| Agentic (no KB, no diag) | 0.541 ± 0.023 | 0.694 ± 0.026 | 0.803 ± 0.029 | 3 | 0.674 | 0.852 | 0.393 | 0.986 | 4.87M | 1.02M | 22.60M | $0.1355 | $3.6331 | $3.7686 | 2932s¹ |
| MO-TPE | 0.509 ± 0.070 | 0.679 ± 0.060 | 0.784 ± 0.053 | 3 | 0.678 | 0.862 | 0.400 | 0.987 | 4.16M | 604.2k | 60.45M | $0.0000 | $2.3195 | $2.3195 | 2450s¹ |
| MO-TPE (transfer warm-start) | 0.414 ± 0.127 | 0.540 ± 0.163 | 0.749 ± 0.096 | 3 | 0.647 | 0.800 | 0.375 | 0.983 | 4.22M | 710.4k | 60.44M | $0.0000 | $1.9181 | $1.9181 | 4474s¹ |
| Random | 0.486 ± 0.045 | 0.644 ± 0.051 | 0.746 ± 0.065 | 3 | 0.661 | 0.822 | 0.385 | 0.986 | 4.25M | 388.8k | 78.71M | $0.0000 | $1.6684 | $1.6684 | 3669s¹ |
| kb-greedy | 0.470 ± 0.017 | 0.601 ± 0.023 | 0.693 ± 0.031 | 3 | 0.625 | 0.769 | 0.362 | 0.986 | 0 | 0 | 0 | $0.0000 | $0.0000 | $0.0000 | 0s¹ |

¹ Wall-clock is reported for context only — rate limits and shared caches make it an unfair primary metric. Token counts are the recommended cost proxy.
