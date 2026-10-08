# Learning-rate sweep

Dev macro-F1, one run per rate (seed 42, up to 12 epochs, early stopping patience 2, batch 16). The best rate per model is what the main runs use.

| Model         |    lr |   dev_macro_f1 |   epochs_used | best   |
|:--------------|------:|---------------:|--------------:|:-------|
| XLM-R base    | 2e-05 |         0.9185 |             8 | True   |
| XLM-R base    | 3e-05 |         0.9156 |            10 | False  |
| XLM-R base    | 5e-05 |         0.7342 |             3 | False  |
| SinhalaBERTo  | 2e-05 |         0.8829 |             5 | True   |
| SinhalaBERTo  | 3e-05 |         0.8793 |             6 | False  |
| SinhalaBERTo  | 5e-05 |         0.8807 |             5 | False  |
| LaBSE         | 2e-05 |         0.8541 |             4 | False  |
| LaBSE         | 3e-05 |         0.8806 |             7 | True   |
| LaBSE         | 5e-05 |         0.8748 |             5 | False  |
| SinBERT-large | 2e-05 |         0.9014 |             6 | False  |
| SinBERT-large | 3e-05 |         0.9072 |             5 | True   |
| SinBERT-large | 5e-05 |         0.8933 |             4 | False  |
