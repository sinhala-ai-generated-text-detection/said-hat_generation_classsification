# SAID — binary human vs. AI detection

Code and results for binary detection on **Sinhala-HAT**: 8,994 documents (4,497
human–AI pairs) in news, Wikipedia and social media, from three generators (GPT-4o,
Gemini 2.5 Pro, DeepSeek-V3).

## Data

The dataset is on the Hugging Face Hub:

```python
from datasets import load_dataset

ds = load_dataset("said-sinhala-ai-dataset/said-hat")                  # all three domains
ds = load_dataset("said-sinhala-ai-dataset/said-hat", "news")          # or one domain
```

`src/data_utils.load_dataset_df()` loads all splits as one DataFrame and renames the
Hub's `validation` split to `dev`. It also puts the rows in a fixed order, which the saved
predictions depend on, and checks the pairing and split integrity. Splits are 70/15/15 by
pair (6,294 / 1,350 / 1,350), so a human text and its AI partner always land in the same
split.

Social-media posts have no punctuation in either class. Keep this in mind for any
punctuation or sentence-based feature.

## Layout

```
classification/
├── src/          data loading, normalization, fine-tuning engine, stylometric features, bootstrap stats
├── scripts/      01–09 below, plus run_all.sh
└── results/      one folder per script; each has a markdown report and CSV tables
```

## Pipeline

| Script | What it does | Output |
|---|---|---|
| `01_build_corpus.py` | Normalize the Hub dataset for training: Latin tokens → `<LAT>`, emoji and wiki markup removed, whitespace collapsed | `results/corpus/` (git-ignored) |
| `02_train_detection.py` | Four encoders, in-distribution (LR search, 3 seeds) and leave-one-domain-out | `results/detection/` |
| `03_generator_holdout_size_control.py` | Leave-one-generator-out; training-size control (4,500 rows) | `results/generalisation/` |
| `04_lr_check_wikipedia.py` | XLM-R LR sweep on a held-out Wikipedia fold | `results/lr_check/` |
| `05_lr_sweep_confusion.py` | LR sweep table (`.tex` too) and confusion-matrix figure | `results/figures/` |
| `06_baselines.py` | Stylometric and TF-IDF logistic regression, on raw and normalized text | `results/baselines/` |
| `07_stylometric_effect_size.py` | Cohen's *d* for 14 stylometric measures by domain and generator | `results/stylometric_effect_size/` |
| `08_sentence_ratio.py` | Sentence- and word-count ratio of each paraphrase to its source | `results/dataset_eval/` |
| `09_semantic_similarity.py` | LaBSE cosine and BERTScore between each AI text and its source (GPU) | `results/dataset_eval/` |
| `10_summary_figures.py` | Summary figures: model comparison and held-out domain/generator transfer | `results/figures/` |

Run from `classification/`:

```
pip install -r requirements.txt
bash scripts/run_all.sh            # or any single script: python scripts/02_train_detection.py
```

Training is resumable: a run whose `run_id` is already in that folder's `runs.csv` is
skipped. Scripts 02–04 take `--analyse-only`, which rebuilds the tables from the saved runs
and predictions without a GPU.

**Training setup.** XLM-R base, SinhalaBERTo, LaBSE and SinBERT-large. Batch size 16, LR
from {2, 3, 5}×10⁻⁵ chosen on dev macro-F1, up to 12 epochs with early stopping (patience
2), fp16. Long texts are split into sliding windows (50% overlap) and the window logits are
averaged. Window lengths per model and domain are in `results/detection/max_length_by_model.json`.
Held-out folds re-split the seen data 75/25 by pair and use one seed (42). fp16 training is
not bit-reproducible, so expect reruns to differ by about 0.006 macro-F1 in-distribution and
0.02 cross-domain.

## Results

In-distribution test macro-F1, mean ± s.d. over 3 seeds, with a 95% pair-clustered
bootstrap CI (`results/detection/indist_summary.csv`):

| Model | LR | Macro-F1 | 95% CI |
|---|---|---|---|
| XLM-R base | 2e-5 | 0.907 ± 0.009 | 0.895–0.920 |
| SinBERT-large | 3e-5 | 0.891 ± 0.003 | 0.877–0.904 |
| SinhalaBERTo | 2e-5 | 0.875 ± 0.001 | 0.860–0.889 |
| LaBSE | 3e-5 | 0.871 ± 0.010 | 0.856–0.886 |

Baselines (`results/baselines/`, normalized text): stylometric LR with 15 features 0.660;
character 2–5-gram TF-IDF LR 0.875.

Generalisation: removing a whole domain costs 0.08–0.51 macro-F1, and social media is the
hardest domain to reach (0.36–0.51 when held out). Removing one generator costs
0.028–0.087. Cutting training to 4,500 rows costs at most 0.024, so the held-out drops are
not a data-size effect. See `results/detection/lodo_summary.csv`,
`results/generalisation/logo_summary.csv` and `size_control.csv`.

## Notes

- Per-document `test_predictions.csv` files are not included; training writes them alongside
  `runs.csv`.
