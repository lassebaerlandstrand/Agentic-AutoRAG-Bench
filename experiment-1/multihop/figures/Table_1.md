# MultiHop-RAG held-out scores

EM / F1 / Judge columns: mean ± sample SD (ddof=1) across the N held-out evaluations pooled per method — one per seed, plus any extra `replay-holdout` re-evaluations of the winning config. With 10 seeds and no replays, N=10 and the SD is the across-seed standard deviation. Token / cost / wall columns are mean across seeds (search-side, unaffected by hold-out replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to find the winning config).

| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic (Ours) | 0.729 ± 0.054 | 0.748 ± 0.055 | 0.790 ± 0.029 | 10 | 0.247 | 0.526 | 0.210 | 0.844 | 15.73M | 1.27M | 11.03M | $0.7856 | $9.4562 | $10.2418 | 4461s¹ |
| Agentic@10 | 0.679 ± 0.056 | 0.697 ± 0.055 | 0.773 ± 0.021 | 10 | 0.263 | 0.562 | 0.221 | 0.859 | 5.22M | 357.3k | 5.24M | $0.2052 | $2.7143 | $2.9195 | 1507s¹ |
| Agentic@20 | 0.704 ± 0.060 | 0.723 ± 0.061 | 0.783 ± 0.026 | 10 | 0.248 | 0.545 | 0.216 | 0.849 | 10.65M | 827.2k | 8.07M | $0.4849 | $6.0499 | $6.5348 | 3014s¹ |
| Agentic (no KB, no diag) | 0.729 ± 0.037 | 0.748 ± 0.035 | 0.780 ± 0.021 | 10 | 0.258 | 0.502 | 0.202 | 0.850 | 10.38M | 1.21M | 12.47M | $0.1346 | $6.6953 | $6.8299 | 3970s¹ |
| MO-TPE | 0.664 ± 0.045 | 0.677 ± 0.042 | 0.724 ± 0.028 | 10 | 0.232 | 0.451 | 0.187 | 0.821 | 8.79M | 796.8k | 39.06M | $0.0000 | $4.2118 | $4.2118 | 2591s¹ |
| MO-TPE (transfer warm-start) | 0.666 ± 0.051 | 0.682 ± 0.047 | 0.731 ± 0.045 | 10 | 0.241 | 0.413 | 0.173 | 0.843 | 7.85M | 827.7k | 34.01M | $0.0000 | $4.4466 | $4.4466 | 2604s¹ |
| Random | 0.633 ± 0.065 | 0.651 ± 0.063 | 0.706 ± 0.048 | 10 | 0.192 | 0.387 | 0.156 | 0.816 | 7.37M | 626.5k | 45.93M | $0.0000 | $3.2372 | $3.2372 | 2610s¹ |
| kb-greedy | 0.507 ± 0.011 | 0.518 ± 0.011 | 0.544 ± 0.014 | 10 | 0.204 | 0.449 | 0.182 | 0.842 | 0 | 0 | 0 | $0.0000 | $0.0000 | $0.0000 | 0s¹ |

¹ Wall-clock is reported for context only — rate limits and shared caches make it an unfair primary metric. Token counts are the recommended cost proxy.
