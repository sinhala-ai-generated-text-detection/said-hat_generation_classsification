#!/usr/bin/env bash
# Full pipeline. Training scripts are resumable: a run already in that experiment's
# runs.csv is skipped, so killing and restarting picks up where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

step() { echo "===== $(date) : $1 ====="; }

step "build corpus";                    python3 scripts/01_build_corpus.py
step "detection: in-dist + LODO";       python3 scripts/02_train_detection.py
step "LOGO + size control";             python3 scripts/03_generator_holdout_size_control.py
step "LR check on held-out Wikipedia";  python3 scripts/04_lr_check_wikipedia.py
step "LR sweep table + confusion";      python3 scripts/05_lr_sweep_confusion.py
step "stylometric + TF-IDF baselines";  python3 scripts/06_baselines.py
step "stylometric effect sizes";        python3 scripts/07_stylometric_effect_size.py
step "sentence-count ratio";            python3 scripts/08_sentence_ratio.py
step "semantic similarity";             python3 scripts/09_semantic_similarity.py
step "summary figures";                 python3 scripts/10_summary_figures.py
step "DONE"
