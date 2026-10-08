# Stylometric effect sizes (Cohen's d)

14 core stylometric features x 3 domains x 3 generators, on Sinhala-HAT as published. Each cell: that generator's AI documents in that domain (Mode A + Mode B pooled) vs. ALL human documents of that domain. **d = (AI mean - human mean) / pooled SD — positive d means the feature is higher in AI text.** The `_human_minus_ai` files use the opposite sign.

## Pooled over all domains and generators

| feature             |   cohens_d_ai_minus_human |   cohens_d_human_minus_ai |   n_ai |   n_human |
|:--------------------|--------------------------:|--------------------------:|-------:|----------:|
| ttr                 |                    0.3067 |                   -0.3067 |   4497 |      4497 |
| mattr               |                    0.3259 |                   -0.3259 |   4497 |      4497 |
| stem_ttr            |                    0.2938 |                   -0.2938 |   4497 |      4497 |
| herdan_c            |                    0.3107 |                   -0.3107 |   4497 |      4497 |
| ttr_inflation       |                    0.2301 |                   -0.2301 |   4497 |      4497 |
| bigram_diversity    |                    0.3519 |                   -0.3519 |   4497 |      4497 |
| trigram_diversity   |                    0.2877 |                   -0.2877 |   4497 |      4497 |
| avg_word_length     |                    0.0053 |                   -0.0053 |   4497 |      4497 |
| avg_sentence_length |                   -0.0164 |                    0.0164 |   4497 |      4497 |
| sentence_length_sd  |                   -0.2981 |                    0.2981 |   4497 |      4497 |
| punct_density       |                   -0.022  |                    0.022  |   4497 |      4497 |
| fertility           |                   -0.38   |                    0.38   |   4497 |      4497 |
| doc_len_words       |                   -0.2343 |                    0.2343 |   4497 |      4497 |
| doc_len_chars       |                   -0.2449 |                    0.2449 |   4497 |      4497 |


## Full grid (AI - human)

| feature             |   ('news', 'deepseek-v3') |   ('news', 'gemini-2.5-pro') |   ('news', 'gpt-4o') |   ('social_media', 'deepseek-v3') |   ('social_media', 'gemini-2.5-pro') |   ('social_media', 'gpt-4o') |   ('wikipedia', 'deepseek-v3') |   ('wikipedia', 'gemini-2.5-pro') |   ('wikipedia', 'gpt-4o') |
|:--------------------|--------------------------:|-----------------------------:|---------------------:|----------------------------------:|-------------------------------------:|-----------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| ttr                 |                     0.253 |                        0.575 |                0.793 |                             0.208 |                                0.388 |                        0.357 |                         -0.037 |                             0.239 |                     0.858 |
| mattr               |                     0.073 |                        0.645 |                0.592 |                             0.214 |                                0.394 |                        0.355 |                         -0.111 |                             0.491 |                     0.57  |
| stem_ttr            |                     0.251 |                        0.544 |                0.785 |                             0.169 |                                0.384 |                        0.364 |                         -0.039 |                             0.214 |                     0.864 |
| herdan_c            |                     0.216 |                        0.599 |                0.728 |                             0.234 |                                0.404 |                        0.361 |                         -0.187 |                             0.231 |                     0.706 |
| ttr_inflation       |                     0.263 |                        0.379 |                0.632 |                             0.091 |                                0.171 |                        0.172 |                          0.012 |                             0.034 |                     0.707 |
| bigram_diversity    |                     0.299 |                        0.55  |                0.607 |                             0.212 |                                0.299 |                        0.295 |                         -0.124 |                             0.299 |                     0.61  |
| trigram_diversity   |                     0.28  |                        0.394 |                0.42  |                             0.136 |                                0.214 |                        0.2   |                         -0.065 |                             0.261 |                     0.418 |
| avg_word_length     |                    -0.29  |                       -0.126 |                0.021 |                             0.013 |                                0.239 |                        0.174 |                         -0.156 |                            -0.216 |                     0.088 |
| avg_sentence_length |                    -0.22  |                        0.062 |               -0.288 |                             0.011 |                               -0.075 |                       -0.186 |                          0.213 |                             0.471 |                     0.223 |
| sentence_length_sd  |                    -0.484 |                       -0.554 |               -0.58  |                           nan     |                              nan     |                      nan     |                         -0.363 |                            -0.307 |                    -0.372 |
| punct_density       |                     0.02  |                        0.058 |                0.322 |                           nan     |                              nan     |                      nan     |                         -0.147 |                            -0.099 |                    -0.143 |
| fertility           |                    -0.581 |                       -0.598 |               -0.134 |                            -0.343 |                               -0.287 |                       -0.224 |                         -0.51  |                            -0.613 |                    -0.4   |
| doc_len_words       |                    -0.272 |                       -0.195 |               -0.374 |                             0.011 |                               -0.075 |                       -0.186 |                         -0.273 |                            -0.158 |                    -0.396 |
| doc_len_chars       |                    -0.28  |                       -0.198 |               -0.375 |                             0.01  |                               -0.052 |                       -0.166 |                         -0.294 |                            -0.18  |                    -0.412 |


## 15 largest effects by |d|

| domain    | generator      | feature          |   cohens_d |   n_ai |   n_human |   ai_mean |   human_mean |
|:----------|:---------------|:-----------------|-----------:|-------:|----------:|----------:|-------------:|
| wikipedia | gpt-4o         | stem_ttr         |     0.8638 |    500 |      1500 |    0.768  |       0.679  |
| wikipedia | gpt-4o         | ttr              |     0.8578 |    500 |      1500 |    0.7885 |       0.7011 |
| news      | gpt-4o         | ttr              |     0.7932 |    500 |      1497 |    0.8499 |       0.7564 |
| news      | gpt-4o         | stem_ttr         |     0.7849 |    500 |      1497 |    0.8317 |       0.7373 |
| news      | gpt-4o         | herdan_c         |     0.7285 |    500 |      1497 |    0.9639 |       0.9414 |
| wikipedia | gpt-4o         | ttr_inflation    |     0.7067 |    500 |      1500 |    0.8662 |       0.7982 |
| wikipedia | gpt-4o         | herdan_c         |     0.7064 |    500 |      1500 |    0.9535 |       0.9362 |
| news      | gemini-2.5-pro | mattr            |     0.6449 |    500 |      1497 |    0.9174 |       0.8835 |
| news      | gpt-4o         | ttr_inflation    |     0.6316 |    500 |      1497 |    0.9284 |       0.8545 |
| wikipedia | gemini-2.5-pro | fertility        |    -0.6131 |    500 |      1500 |    2.0166 |       2.2254 |
| wikipedia | gpt-4o         | bigram_diversity |     0.6104 |    500 |      1500 |    0.9705 |       0.9307 |
| news      | gpt-4o         | bigram_diversity |     0.6073 |    500 |      1497 |    0.9742 |       0.9173 |
| news      | gemini-2.5-pro | herdan_c         |     0.5986 |    500 |      1497 |    0.96   |       0.9414 |
| news      | gemini-2.5-pro | fertility        |    -0.5984 |    500 |      1497 |    1.7433 |       1.8635 |
| news      | gpt-4o         | mattr            |     0.5924 |    500 |      1497 |    0.9154 |       0.8835 |
