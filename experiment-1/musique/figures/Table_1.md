# MuSiQue-Ans held-out scores

EM / F1 / Judge columns: mean ± sample SD (ddof=1) across the N held-out evaluations pooled per method — one per seed, plus any extra `replay-holdout` re-evaluations of the winning config. With 3 seeds and no replays, N=3 and the SD is the across-seed standard deviation. Token / cost / wall columns are mean across seeds (search-side, unaffected by hold-out replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to find the winning config).

| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic (Ours) | 0.201 ± 0.034 | 0.266 ± 0.035 | 0.348 ± 0.042 | 3 | 0.097 | 0.197 | 0.087 | 0.832 | 7.18M | 1.64M | 16.78M | $0.7539 | $6.0067 | $6.7605 | 3536s¹ |
| Agentic@10 | 0.212 ± 0.012 | 0.272 ± 0.011 | 0.344 ± 0.012 | 3 | 0.101 | 0.212 | 0.092 | 0.833 | 2.37M | 302.1k | 9.11M | $0.1900 | $1.6396 | $1.8296 | 1179s¹ |
| Agentic@20 | 0.192 ± 0.011 | 0.262 ± 0.020 | 0.334 ± 0.035 | 3 | 0.096 | 0.201 | 0.088 | 0.831 | 4.77M | 970.6k | 13.03M | $0.4396 | $3.6258 | $4.0654 | 2358s¹ |
| Agentic (no KB, no diag) | 0.200 ± 0.034 | 0.256 ± 0.034 | 0.311 ± 0.035 | 3 | 0.098 | 0.188 | 0.083 | 0.793 | 5.24M | 2.53M | 19.54M | $0.1304 | $11.2282 | $11.3586 | 3336s¹ |
| MO-TPE | 0.183 ± 0.041 | 0.244 ± 0.060 | 0.295 ± 0.057 | 3 | 0.098 | 0.176 | 0.079 | 0.777 | 4.90M | 925.4k | 47.22M | $0.0000 | $3.4216 | $3.4216 | 2615s¹ |
| MO-TPE (transfer warm-start) | 0.203 ± 0.020 | 0.262 ± 0.019 | 0.319 ± 0.015 | 3 | 0.111 | 0.204 | 0.090 | 0.816 | 4.96M | 945.1k | 40.70M | $0.0000 | $3.1113 | $3.1113 | 2389s¹ |
| Random | 0.164 ± 0.029 | 0.212 ± 0.041 | 0.273 ± 0.054 | 3 | 0.094 | 0.158 | 0.072 | 0.741 | 4.77M | 543.5k | 56.64M | $0.0000 | $2.1579 | $2.1579 | 3403s¹ |
| kb-greedy | 0.149 ± 0.008 | 0.191 ± 0.004 | 0.259 ± 0.020 | 3 | 0.084 | 0.157 | 0.074 | 0.779 | 0 | 0 | 0 | $0.0000 | $0.0000 | $0.0000 | 0s¹ |

¹ Wall-clock is reported for context only — rate limits and shared caches make it an unfair primary metric. Token counts are the recommended cost proxy.
