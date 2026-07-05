# MultiHop-RAG held-out scores

EM / F1 / Judge columns: mean ± sample SD (ddof=1) across the N held-out evaluations pooled per method — one per seed, plus any extra `replay-holdout` re-evaluations of the winning config. With 3 seeds and no replays, N=3 and the SD is the across-seed standard deviation. Token / cost / wall columns are mean across seeds (search-side, unaffected by hold-out replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to find the winning config).

| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic (Ours) | 0.684 ± 0.054 | 0.703 ± 0.051 | 0.780 ± 0.009 | 3 | 0.260 | 0.487 | 0.210 | 0.837 | 14.09M | 1.53M | 16.63M | $0.7898 | $7.6948 | $8.4846 | 4383s¹ |
| Agentic@10 | 0.674 ± 0.046 | 0.694 ± 0.049 | 0.769 ± 0.012 | 3 | 0.245 | 0.523 | 0.213 | 0.847 | 5.25M | 536.4k | 6.15M | $0.2135 | $3.0289 | $3.2424 | 1461s¹ |
| Agentic@20 | 0.678 ± 0.063 | 0.697 ± 0.060 | 0.770 ± 0.018 | 3 | 0.256 | 0.511 | 0.212 | 0.845 | 9.87M | 1.01M | 12.43M | $0.4913 | $5.6092 | $6.1005 | 2922s¹ |
| Agentic (no KB, no diag) | 0.716 ± 0.068 | 0.734 ± 0.065 | 0.780 ± 0.012 | 3 | 0.253 | 0.520 | 0.206 | 0.847 | 10.74M | 1.03M | 11.12M | $0.1347 | $4.0167 | $4.1514 | 4185s¹ |
| MO-TPE | 0.690 ± 0.037 | 0.704 ± 0.036 | 0.746 ± 0.025 | 3 | 0.241 | 0.466 | 0.192 | 0.822 | 8.82M | 899.8k | 40.91M | $0.0000 | $3.9320 | $3.9320 | 2464s¹ |
| MO-TPE (transfer warm-start) | 0.644 ± 0.030 | 0.666 ± 0.028 | 0.757 ± 0.034 | 3 | 0.232 | 0.378 | 0.160 | 0.825 | 8.31M | 1.01M | 33.81M | $0.0000 | $5.2522 | $5.2522 | 2665s¹ |
| Random | 0.640 ± 0.067 | 0.656 ± 0.062 | 0.710 ± 0.049 | 3 | 0.227 | 0.439 | 0.175 | 0.841 | 8.15M | 611.6k | 45.57M | $0.0000 | $2.9366 | $2.9366 | 2684s¹ |
| kb-greedy | 0.507 ± 0.013 | 0.517 ± 0.013 | 0.539 ± 0.008 | 3 | 0.203 | 0.453 | 0.182 | 0.844 | 0 | 0 | 0 | $0.0000 | $0.0000 | $0.0000 | 0s¹ |

¹ Wall-clock is reported for context only — rate limits and shared caches make it an unfair primary metric. Token counts are the recommended cost proxy.
