# Semantic preservation via embedding similarity

LaBSE cosine similarity and multilingual BERTScore F1 (xlm-roberta-base, no baseline rescaling), between each AI document and its paired human source. Both texts truncated to the first 500 words before scoring. Inference only, no fine-tuning.

## Mode B (paraphrase vs. its source) — the main result

### Overall

|         |    n |   labse_cosine_mean |   labse_cosine_std |   bertscore_f1_mean |   bertscore_f1_std |
|:--------|-----:|--------------------:|-------------------:|--------------------:|-------------------:|
| overall | 3297 |              0.9472 |             0.0443 |              0.9136 |              0.032 |


### By domain

| domain       |    n |   labse_cosine_mean |   labse_cosine_std |   bertscore_f1_mean |   bertscore_f1_std |
|:-------------|-----:|--------------------:|-------------------:|--------------------:|-------------------:|
| news         |  897 |              0.9738 |             0.0235 |              0.9352 |             0.0293 |
| social_media | 1500 |              0.9221 |             0.046  |              0.9032 |             0.0251 |
| wikipedia    |  900 |              0.9626 |             0.0332 |              0.9096 |             0.0344 |


### By generator

| generator      |    n |   labse_cosine_mean |   labse_cosine_std |   bertscore_f1_mean |   bertscore_f1_std |
|:---------------|-----:|--------------------:|-------------------:|--------------------:|-------------------:|
| deepseek-v3    | 1097 |              0.9534 |             0.0461 |              0.9259 |             0.0349 |
| gemini-2.5-pro | 1100 |              0.9489 |             0.0424 |              0.9119 |             0.0254 |
| gpt-4o         | 1100 |              0.9393 |             0.0431 |              0.9031 |             0.0307 |


### By domain x generator

|                                    |   n |   labse_cosine_mean |   labse_cosine_std |   bertscore_f1_mean |   bertscore_f1_std |
|:-----------------------------------|----:|--------------------:|-------------------:|--------------------:|-------------------:|
| ('news', 'deepseek-v3')            | 297 |              0.9809 |             0.0214 |              0.949  |             0.029  |
| ('news', 'gemini-2.5-pro')         | 300 |              0.9773 |             0.0164 |              0.9342 |             0.0208 |
| ('news', 'gpt-4o')                 | 300 |              0.9634 |             0.0277 |              0.9225 |             0.0308 |
| ('social_media', 'deepseek-v3')    | 500 |              0.9231 |             0.0476 |              0.9068 |             0.0258 |
| ('social_media', 'gemini-2.5-pro') | 500 |              0.9218 |             0.0452 |              0.9022 |             0.0225 |
| ('social_media', 'gpt-4o')         | 500 |              0.9213 |             0.0451 |              0.9004 |             0.0266 |
| ('wikipedia', 'deepseek-v3')       | 300 |              0.9766 |             0.027  |              0.9346 |             0.036  |
| ('wikipedia', 'gemini-2.5-pro')    | 300 |              0.9658 |             0.0244 |              0.9057 |             0.0203 |
| ('wikipedia', 'gpt-4o')            | 300 |              0.9453 |             0.0386 |              0.8883 |             0.0272 |


## Mode A (title-conditioned) vs. the SAME paired human document — lower-similarity reference

Mode A never saw this document, only its title. Low similarity here is expected and is what makes the Mode B numbers above meaningful, not a paraphrase-quality result on its own.

### Overall

|         |    n |   labse_cosine_mean |   labse_cosine_std |   bertscore_f1_mean |   bertscore_f1_std |
|:--------|-----:|--------------------:|-------------------:|--------------------:|-------------------:|
| overall | 1200 |              0.7938 |              0.074 |               0.841 |             0.0184 |


### By domain

| domain    |   n |   labse_cosine_mean |   labse_cosine_std |   bertscore_f1_mean |   bertscore_f1_std |
|:----------|----:|--------------------:|-------------------:|--------------------:|-------------------:|
| news      | 600 |              0.7751 |             0.069  |              0.846  |             0.0178 |
| wikipedia | 600 |              0.8125 |             0.0741 |              0.8361 |             0.0177 |


## Note

Named-entity retention was not computed here (no Sinhala NER tool identified).
