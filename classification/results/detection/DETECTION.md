# Binary detection — in-distribution and leave-one-domain-out

Normalized Sinhala-HAT, test split (675 pairs). In-distribution: mean and seed std over 3 seeds at the dev-selected LR; 95% CI from a pair-clustered bootstrap (5,000 resamples, test-set sampling only).

## In-distribution: test macro-F1

| Model         |    lr |   n_seeds |   test_macro_f1_mean |   seed_std |   ci95_low |   ci95_high |   dev_macro_f1_mean |
|:--------------|------:|----------:|---------------------:|-----------:|-----------:|------------:|--------------------:|
| XLM-R base    | 2e-05 |         3 |               0.9074 |     0.0085 |     0.8945 |      0.9197 |              0.9104 |
| SinhalaBERTo  | 2e-05 |         3 |               0.8747 |     0.0007 |     0.8596 |      0.8893 |              0.8819 |
| LaBSE         | 3e-05 |         3 |               0.8713 |     0.0103 |     0.8558 |      0.8861 |              0.8807 |
| SinBERT-large | 3e-05 |         3 |               0.8908 |     0.0033 |     0.8765 |      0.9042 |              0.8997 |


## In-distribution: by domain

| Model         |   news |   social_media |   wikipedia |
|:--------------|-------:|---------------:|------------:|
| LaBSE         | 0.9094 |         0.7966 |      0.9093 |
| SinBERT-large | 0.9128 |         0.8815 |      0.8747 |
| SinhalaBERTo  | 0.8803 |         0.8749 |      0.8675 |
| XLM-R base    | 0.9362 |         0.8675 |      0.9175 |


## Leave-one-domain-out (single seed 42)

`indist_same_slice` is the in-distribution 3-seed score on the held-out domain's test slice. `real` marks a drop larger than twice the larger of that slice's seed spread and the model's overall seed std.

| Model         | held_out     |    lr |   holdout_f1 |   indist_same_slice |   drop |   threshold | real   |
|:--------------|:-------------|------:|-------------:|--------------------:|-------:|------------:|:-------|
| XLM-R base    | news         | 2e-05 |       0.7547 |              0.9362 | 0.1815 |      0.0171 | yes    |
| XLM-R base    | social_media | 2e-05 |       0.3587 |              0.8675 | 0.5088 |      0.0408 | yes    |
| XLM-R base    | wikipedia    | 2e-05 |       0.8244 |              0.9175 | 0.0931 |      0.0198 | yes    |
| SinhalaBERTo  | news         | 2e-05 |       0.7469 |              0.8803 | 0.1334 |      0.0084 | yes    |
| SinhalaBERTo  | social_media | 2e-05 |       0.4005 |              0.8749 | 0.4744 |      0.0174 | yes    |
| SinhalaBERTo  | wikipedia    | 2e-05 |       0.7412 |              0.8675 | 0.1263 |      0.0177 | yes    |
| LaBSE         | news         | 3e-05 |       0.8176 |              0.9094 | 0.0918 |      0.0205 | yes    |
| LaBSE         | social_media | 3e-05 |       0.4527 |              0.7966 | 0.344  |      0.0355 | yes    |
| LaBSE         | wikipedia    | 3e-05 |       0.8335 |              0.9093 | 0.0758 |      0.0246 | yes    |
| SinBERT-large | news         | 3e-05 |       0.7096 |              0.9128 | 0.2032 |      0.0195 | yes    |
| SinBERT-large | social_media | 3e-05 |       0.5128 |              0.8815 | 0.3686 |      0.0067 | yes    |
| SinBERT-large | wikipedia    | 3e-05 |       0.7771 |              0.8747 | 0.0976 |      0.015  | yes    |
