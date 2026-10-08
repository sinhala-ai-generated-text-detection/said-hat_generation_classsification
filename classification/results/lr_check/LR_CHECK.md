# Learning-rate check on a held-out Wikipedia fold

XLM-R base, trained on news + social media (75/25 pair-grouped re-split, seed 42), one run per rate — read as point estimates, not means.

| Model      | held_out   |    lr |   dev_macro_f1 |   test_macro_f1 |
|:-----------|:-----------|------:|---------------:|----------------:|
| XLM-R base | wikipedia  | 2e-05 |         0.91   |          0.8475 |
| XLM-R base | wikipedia  | 3e-05 |         0.8986 |          0.8507 |
| XLM-R base | wikipedia  | 5e-05 |         0.7418 |          0.8291 |


Dev macro-F1 spread across the three rates: 0.1682. Best rate on this fold's own dev: 2e-05 (dev 0.9100, unseen-wikipedia test 0.8475).
