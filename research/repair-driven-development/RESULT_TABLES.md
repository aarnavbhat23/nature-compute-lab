# Development result tables

Held-out 48-interval normalized mean squared error; lower is better. Each cell is the mean ± sample standard deviation across three generated systems within that family, not a confidence interval. These familiar families were used during development; do not pool them into a universal ranking.

### updates

| Model / condition | Saturating | Polynomial | Forced sinusoidal |
|---|---:|---:|---:|
| original / 240 updates | 0.00763273 ± 0.00153 | 0.0378854 ± 0.0124 | 0.148282 ± 0.0483 |
| polarized / 240 updates | 0.00986219 ± 0.00204 | 0.041642 ± 0.0113 | 0.139469 ± 0.039 |
| occupancy / 240 updates | 0.00444363 ± 0.00258 | 0.00637101 ± 0.00114 | 0.0793322 ± 0.0345 |
| neural_730 / 240 updates | 0.00177317 ± 0.00022 | 0.00948299 ± 0.00217 | 0.147996 ± 0.0486 |
| neural_808 / 240 updates | 0.00342085 ± 0.000853 | 0.00696448 ± 0.00302 | 0.105062 ± 0.0346 |
| original / 960 updates | 0.000554492 ± 9.73e-05 | 0.012851 ± 0.0055 | 0.0738647 ± 0.0288 |
| polarized / 960 updates | 0.000558595 ± 4.56e-05 | 0.0134548 ± 0.00575 | 0.0703574 ± 0.03 |
| occupancy / 960 updates | 0.00123748 ± 7.73e-05 | 0.00455426 ± 0.000875 | 0.116905 ± 0.0429 |
| neural_730 / 960 updates | 0.00100261 ± 0.000315 | 0.00454845 ± 0.00191 | 0.0941962 ± 0.0298 |
| neural_808 / 960 updates | 0.00167001 ± 0.000638 | 0.00368494 ± 0.000676 | 0.120966 ± 0.0554 |

### walltime

| Model / condition | Saturating | Polynomial | Forced sinusoidal |
|---|---:|---:|---:|
| original | 0.00133411 ± 0.000399 | 0.0188025 ± 0.00349 | 0.0650996 ± 0.00438 |
| occupancy | 0.00086276 ± 0.000193 | 0.00493129 ± 0.000744 | 0.0910938 ± 0.0466 |
| neural_730 | 0.000645704 ± 0.000129 | 0.00474535 ± 0.00182 | 0.0928327 ± 0.0453 |
| neural_808 | 0.000808071 ± 0.000147 | 0.0100472 ± 0.00395 | 0.12953 ± 0.0427 |

### horizon

| Model / condition | Saturating | Polynomial | Forced sinusoidal |
|---|---:|---:|---:|
| original / validation 12 | 0.000689555 ± 0.000168 | 0.0136343 ± 0.00154 | 0.113092 ± 0.0465 |
| original / validation 48 | 0.000455923 ± 7.35e-05 | 0.0101548 ± 0.000998 | 0.0847499 ± 0.0218 |
| occupancy / validation 12 | 0.00261607 ± 0.000941 | 0.00429792 ± 0.00112 | 0.0988717 ± 0.0263 |
| occupancy / validation 48 | 0.00144914 ± 0.000204 | 0.00409012 ± 0.000919 | 0.0657105 ± 0.0179 |
| neural_730 / validation 12 | 0.00149397 ± 0.000981 | 0.00478589 ± 0.00415 | 0.122144 ± 0.0464 |
| neural_730 / validation 48 | 0.000789246 ± 0.000224 | 0.00279078 ± 0.000608 | 0.0685734 ± 0.00904 |
| neural_808 / validation 12 | 0.00114185 ± 0.000249 | 0.00499322 ± 0.00293 | 0.133015 ± 0.0761 |
| neural_808 / validation 48 | 0.000985938 ± 0.000116 | 0.00317972 ± 0.001 | 0.0697419 ± 0.00916 |

Long validation selection improved 31/36 individual model/system outcomes, and all 12 model/family means.
