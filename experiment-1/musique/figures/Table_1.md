# MuSiQue-Ans held-out scores

EM / F1 / Judge columns: mean ± sample SD (ddof=1) across the N held-out evaluations pooled per method — one per seed, plus any extra `replay-holdout` re-evaluations of the winning config. With 10 seeds and no replays, N=10 and the SD is the across-seed standard deviation. Token / cost / wall columns are mean across seeds (search-side, unaffected by hold-out replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to find the winning config).

| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic (Ours) | 0.196 ± 0.031 | 0.262 ± 0.028 | 0.330 ± 0.025 | 10 | 0.095 | 0.185 | 0.085 | 0.821 | 7.23M | 1.70M | 16.02M | $0.7612 | $6.5456 | $7.3068 | 3983s¹ |
| Agentic@10 | 0.177 ± 0.023 | 0.231 ± 0.028 | 0.303 ± 0.030 | 10 | 0.093 | 0.180 | 0.082 | 0.821 | 2.39M | 360.8k | 9.20M | $0.2018 | $1.5744 | $1.7762 | 1350s¹ |
| Agentic@20 | 0.198 ± 0.027 | 0.261 ± 0.028 | 0.326 ± 0.030 | 10 | 0.097 | 0.186 | 0.085 | 0.819 | 4.84M | 1.04M | 13.52M | $0.4541 | $4.0144 | $4.4685 | 2700s¹ |
| Agentic (no KB, no diag) | 0.180 ± 0.022 | 0.230 ± 0.028 | 0.298 ± 0.024 | 10 | 0.099 | 0.179 | 0.079 | 0.794 | 4.78M | 1.95M | 20.14M | $0.1256 | $8.5531 | $8.6787 | 3564s¹ |
| MO-TPE | 0.193 ± 0.024 | 0.254 ± 0.032 | 0.301 ± 0.032 | 10 | 0.100 | 0.186 | 0.082 | 0.791 | 4.73M | 745.7k | 48.19M | $0.0000 | $3.0499 | $3.0499 | 2462s¹ |
| MO-TPE (transfer warm-start) | 0.183 ± 0.030 | 0.243 ± 0.032 | 0.301 ± 0.039 | 10 | 0.104 | 0.185 | 0.083 | 0.790 | 4.78M | 886.7k | 40.63M | $0.0000 | $3.5525 | $3.5525 | 2561s¹ |
| Random | 0.165 ± 0.029 | 0.215 ± 0.035 | 0.270 ± 0.036 | 10 | 0.095 | 0.167 | 0.076 | 0.785 | 4.56M | 542.6k | 56.72M | $0.0000 | $2.4250 | $2.4250 | 2852s¹ |
| kb-greedy | 0.150 ± 0.008 | 0.187 ± 0.008 | 0.254 ± 0.014 | 10 | 0.084 | 0.157 | 0.074 | 0.779 | 0 | 0 | 0 | $0.0000 | $0.0000 | $0.0000 | 0s¹ |

¹ Wall-clock is reported for context only — rate limits and shared caches make it an unfair primary metric. Token counts are the recommended cost proxy.
