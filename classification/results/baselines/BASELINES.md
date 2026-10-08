# Stylometric and TF-IDF baselines

`raw` is Sinhala-HAT as published; `normalized` is the encoders' training corpus. Same train/dev/test split as the encoders.

## Stylometric logistic regression

| text       | feature_set   |   n_features |   dev_roc_auc |   test_roc_auc |   test_macro_f1 |   test_accuracy |
|:-----------|:--------------|-------------:|--------------:|---------------:|----------------:|----------------:|
| raw        | core_14       |           14 |        0.763  |         0.7226 |          0.6567 |          0.657  |
| raw        | full_15       |           15 |        0.7766 |         0.7318 |          0.6723 |          0.6726 |
| raw        | reduced_11    |           11 |        0.6986 |         0.6749 |          0.632  |          0.6326 |
| normalized | core_14       |           14 |        0.7569 |         0.7119 |          0.6536 |          0.6541 |
| normalized | full_15       |           15 |        0.7759 |         0.7261 |          0.6604 |          0.6607 |
| normalized | reduced_11    |           11 |        0.701  |         0.6775 |          0.6334 |          0.6341 |


### Stylometric, full_15, test macro-F1 by domain

| text       |   news |   social_media |   wikipedia |
|:-----------|-------:|---------------:|------------:|
| normalized | 0.6828 |         0.568  |      0.7252 |
| raw        | 0.6891 |         0.5719 |      0.7525 |


## TF-IDF logistic regression

| text       | features   |   C |   dev_macro_f1 |   test_macro_f1 |   test_accuracy |   test_roc_auc |
|:-----------|:-----------|----:|---------------:|----------------:|----------------:|---------------:|
| raw        | word_1_2   | 100 |         0.8777 |          0.8731 |          0.8733 |         0.946  |
| raw        | char_2_5   |  10 |         0.8874 |          0.8709 |          0.8711 |         0.9496 |
| normalized | word_1_2   | 100 |         0.8784 |          0.8723 |          0.8726 |         0.9464 |
| normalized | char_2_5   |  10 |         0.8844 |          0.8754 |          0.8756 |         0.9481 |


### TF-IDF, test macro-F1 by domain

|                            |   news |   social_media |   wikipedia |
|:---------------------------|-------:|---------------:|------------:|
| ('char_2_5', 'normalized') | 0.8889 |         0.8452 |      0.8921 |
| ('char_2_5', 'raw')        | 0.8889 |         0.8452 |      0.8774 |
| ('word_1_2', 'normalized') | 0.9012 |         0.8054 |      0.9117 |
| ('word_1_2', 'raw')        | 0.9074 |         0.8004 |      0.9117 |
