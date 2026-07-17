# HotpotQA-distractor held-out scores

EM / F1 / Judge columns: mean ± sample SD (ddof=1) across the N held-out evaluations pooled per method — one per seed, plus any extra `replay-holdout` re-evaluations of the winning config. With 10 seeds and no replays, N=10 and the SD is the across-seed standard deviation. Token / cost / wall columns are mean across seeds (search-side, unaffected by hold-out replays). `Search $` = Optimizer $ + Trial $ (the one-time bill to find the winning config).

| Method | EM | Token-F1 | LLM Judge | N | Joint-R@2 | Joint-R@5 | MRR-complete | MRR-first | LLM in | LLM out | Embed in | Optimizer $ | Trial $ | Search $ | Wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic (Ours) | 0.566 ± 0.015 | 0.723 ± 0.014 | 0.825 ± 0.015 | 10 | 0.695 | 0.844 | 0.401 | 0.984 | 6.12M | 833.0k | 13.85M | $0.7060 | $4.0378 | $4.7438 | 3459s¹ |
| Agentic@10 | 0.558 ± 0.015 | 0.715 ± 0.017 | 0.817 ± 0.014 | 10 | 0.696 | 0.844 | 0.399 | 0.984 | 2.02M | 234.5k | 8.38M | $0.1889 | $1.2922 | $1.4811 | 1181s¹ |
| Agentic@20 | 0.568 ± 0.012 | 0.727 ± 0.009 | 0.835 ± 0.013 | 10 | 0.697 | 0.845 | 0.401 | 0.985 | 4.12M | 546.6k | 11.95M | $0.4351 | $2.6459 | $3.0810 | 2362s¹ |
| Agentic (no KB, no diag) | 0.550 ± 0.020 | 0.711 ± 0.024 | 0.819 ± 0.025 | 10 | 0.680 | 0.855 | 0.395 | 0.985 | 4.41M | 1.10M | 23.74M | $0.1335 | $5.2219 | $5.3554 | 3251s¹ |
| MO-TPE | 0.515 ± 0.057 | 0.673 ± 0.056 | 0.784 ± 0.056 | 10 | 0.632 | 0.814 | 0.376 | 0.980 | 4.03M | 560.6k | 63.41M | $0.0000 | $2.1844 | $2.1844 | 2514s¹ |
| MO-TPE (transfer warm-start) | 0.521 ± 0.044 | 0.677 ± 0.055 | 0.786 ± 0.057 | 10 | 0.617 | 0.792 | 0.367 | 0.975 | 4.22M | 670.4k | 60.44M | $0.0000 | $2.1576 | $2.1576 | 2985s¹ |
| Random | 0.506 ± 0.038 | 0.664 ± 0.041 | 0.774 ± 0.046 | 10 | 0.614 | 0.791 | 0.367 | 0.976 | 3.98M | 386.1k | 78.86M | $0.0000 | $1.8595 | $1.8595 | 3100s¹ |
| kb-greedy | 0.480 ± 0.016 | 0.611 ± 0.016 | 0.702 ± 0.019 | 10 | 0.625 | 0.769 | 0.362 | 0.986 | 0 | 0 | 0 | $0.0000 | $0.0000 | $0.0000 | 0s¹ |

¹ Wall-clock is reported for context only — rate limits and shared caches make it an unfair primary metric. Token counts are the recommended cost proxy.
