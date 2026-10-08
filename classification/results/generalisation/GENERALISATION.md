# Generalisation — leave-one-generator-out and training-size control

Normalized Sinhala-HAT, single seed (42), each model's in-distribution LR.

## Leave-one-generator-out

`indist_same_slice` is the model's in-distribution 3-seed score on the held-out generator's slice of the test split. `real` marks a drop larger than twice the larger of that slice's seed spread and the model's overall seed std.

| Model         | held_out_generator   |    lr |   holdout_f1 |   indist_same_slice |   drop |   threshold | real   |
|:--------------|:---------------------|------:|-------------:|--------------------:|-------:|------------:|:-------|
| LaBSE         | deepseek-v3          | 3e-05 |       0.8058 |              0.8486 | 0.0429 |      0.0244 | yes    |
| LaBSE         | gemini-2.5-pro       | 3e-05 |       0.8545 |              0.8901 | 0.0356 |      0.0206 | yes    |
| LaBSE         | gpt-4o               | 3e-05 |       0.786  |              0.8733 | 0.0872 |      0.0317 | yes    |
| SinBERT-large | deepseek-v3          | 3e-05 |       0.8059 |              0.8567 | 0.0508 |      0.0197 | yes    |
| SinBERT-large | gemini-2.5-pro       | 3e-05 |       0.8912 |              0.9234 | 0.0321 |      0.0355 | noise  |
| SinBERT-large | gpt-4o               | 3e-05 |       0.8373 |              0.8902 | 0.0529 |      0.0152 | yes    |
| SinhalaBERTo  | deepseek-v3          | 2e-05 |       0.7983 |              0.8429 | 0.0446 |      0.0152 | yes    |
| SinhalaBERTo  | gemini-2.5-pro       | 2e-05 |       0.8733 |              0.9021 | 0.0288 |      0.021  | yes    |
| SinhalaBERTo  | gpt-4o               | 2e-05 |       0.849  |              0.8767 | 0.0277 |      0.0121 | yes    |
| XLM-R base    | deepseek-v3          | 2e-05 |       0.8051 |              0.8657 | 0.0606 |      0.0202 | yes    |
| XLM-R base    | gemini-2.5-pro       | 2e-05 |       0.909  |              0.937  | 0.028  |      0.017  | yes    |
| XLM-R base    | gpt-4o               | 2e-05 |       0.8413 |              0.9156 | 0.0743 |      0.0281 | yes    |


## Training-size control

Same dev/test as the in-distribution runs; training cut to 4,500 rows (the size a held-out fold trains on). `cost_of_less_data_vs_mean` is the 3-seed in-distribution mean minus the size-control run; `mean_LODO_drop_vs_full` is the same mean minus the mean leave-one-domain-out score, for comparison.

| Model         |   n_train_full |   n_train_control |   indist_full_mean_3seeds |   indist_full_seed42 |   size_control_seed42 |   cost_of_less_data_vs_mean |   mean_LODO_holdout_f1 |   mean_LODO_drop_vs_full |
|:--------------|---------------:|------------------:|--------------------------:|---------------------:|----------------------:|----------------------------:|-----------------------:|-------------------------:|
| LaBSE         |           6294 |              4500 |                    0.8713 |               0.8741 |                0.8659 |                      0.0054 |                 0.7013 |                   0.17   |
| SinBERT-large |           6294 |              4500 |                    0.8908 |               0.8911 |                0.8807 |                      0.0101 |                 0.6665 |                   0.2243 |
| SinhalaBERTo  |           6294 |              4500 |                    0.8747 |               0.874  |                0.8511 |                      0.0236 |                 0.6295 |                   0.2452 |
| XLM-R base    |           6294 |              4500 |                    0.9074 |               0.914  |                0.8918 |                      0.0156 |                 0.6459 |                   0.2615 |
