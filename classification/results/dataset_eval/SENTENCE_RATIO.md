# Sentence-count ratio

Mode B (paraphrase) pairs: n=3297. Ratio = AI sentence count / human-source sentence count, sentences counted with the deterministic Sinhala segmenter (abbreviation/decimal/initial/quote-aware). Word-count ratio is included alongside as a cross-check against the already-reported 0.89-0.99 figure.

## Overall (Mode B)

|         |    n |   sentence_ratio_mean |   sentence_ratio_median |   sentence_ratio_std |   word_ratio_mean |   word_ratio_median |
|:--------|-----:|----------------------:|------------------------:|---------------------:|------------------:|--------------------:|
| overall | 3297 |                  1.16 |                       1 |               1.2074 |            0.9442 |              0.9615 |


## By domain (Mode B)

| domain       |    n |   sentence_ratio_mean |   sentence_ratio_median |   sentence_ratio_std |   word_ratio_mean |   word_ratio_median |
|:-------------|-----:|----------------------:|------------------------:|---------------------:|------------------:|--------------------:|
| news         |  897 |                1.5614 |                  1      |               1.6234 |            0.8887 |              0.9438 |
| social_media | 1500 |                1      |                  1      |               0      |            0.993  |              0.9728 |
| wikipedia    |  900 |                1.0265 |                  0.9077 |               1.5802 |            0.9183 |              0.9632 |


## By generator (Mode B)

| generator      |    n |   sentence_ratio_mean |   sentence_ratio_median |   sentence_ratio_std |   word_ratio_mean |   word_ratio_median |
|:---------------|-----:|----------------------:|------------------------:|---------------------:|------------------:|--------------------:|
| deepseek-v3    | 1097 |                1.2247 |                       1 |               1.6276 |            0.999  |              1      |
| gemini-2.5-pro | 1100 |                1.161  |                       1 |               0.9229 |            1.0125 |              1      |
| gpt-4o         | 1100 |                1.0944 |                       1 |               0.9328 |            0.8213 |              0.8635 |


## By domain x generator (Mode B)

|                                    |   n |   sentence_ratio_mean |   sentence_ratio_median |   sentence_ratio_std |   word_ratio_mean |   word_ratio_median |
|:-----------------------------------|----:|----------------------:|------------------------:|---------------------:|------------------:|--------------------:|
| ('news', 'deepseek-v3')            | 297 |                1.6129 |                  1      |               1.9678 |            0.9171 |              0.9723 |
| ('news', 'gemini-2.5-pro')         | 300 |                1.5004 |                  1      |               1.3576 |            0.9584 |              0.9906 |
| ('news', 'gpt-4o')                 | 300 |                1.5716 |                  1      |               1.4881 |            0.7908 |              0.8573 |
| ('social_media', 'deepseek-v3')    | 500 |                1      |                  1      |               0      |            1.0644 |              1.0345 |
| ('social_media', 'gemini-2.5-pro') | 500 |                1      |                  1      |               0      |            0.9965 |              1      |
| ('social_media', 'gpt-4o')         | 500 |                1      |                  1      |               0      |            0.9181 |              0.9221 |
| ('wikipedia', 'deepseek-v3')       | 300 |                1.2149 |                  1      |               2.3755 |            0.9714 |              0.9877 |
| ('wikipedia', 'gemini-2.5-pro')    | 300 |                1.0901 |                  0.9333 |               1.06   |            1.0931 |              1.0415 |
| ('wikipedia', 'gpt-4o')            | 300 |                0.7745 |                  0.7303 |               0.7985 |            0.6904 |              0.7029 |


## For context: Mode A (title-conditioned), vs. the same paired human document

Mode A does not paraphrase this document — it was written from only the title. Included as a reference point only.

|                  |    n |   sentence_ratio_mean |   sentence_ratio_median |   sentence_ratio_std |   word_ratio_mean |   word_ratio_median |
|:-----------------|-----:|----------------------:|------------------------:|---------------------:|------------------:|--------------------:|
| overall (mode A) | 1200 |                1.1902 |                  0.7692 |               1.4575 |            0.7122 |              0.7228 |

| domain    |   n |   sentence_ratio_mean |   sentence_ratio_median |   sentence_ratio_std |   word_ratio_mean |   word_ratio_median |
|:----------|----:|----------------------:|------------------------:|---------------------:|------------------:|--------------------:|
| news      | 600 |                1.6434 |                  1      |               1.6428 |            0.7958 |              0.7984 |
| wikipedia | 600 |                0.737  |                  0.5714 |               1.0685 |            0.6286 |              0.6594 |
