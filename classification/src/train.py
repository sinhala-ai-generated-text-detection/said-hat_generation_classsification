"""
Fine-tuning engine for the encoder classifiers.

Sliding-window chunking with logit-averaging pooling back to document level,
LR search then multi-seed main runs, and a resumable run log (a run whose
run_id is already in runs.csv is skipped). The per-model, per-domain window
budget is computed once and saved, so every experiment uses the same windows.

Protocol: batch 16 (fallback 8 on OOM), LR searched
over {2e-5, 3e-5, 5e-5} on dev macro-F1 (search phase, seed 42), up to 12
epochs with early-stopping patience 2, fp16, sliding windows with a 50%
overlap (stride_factor=2), 3 main seeds (42, 123, 2026).
"""
from __future__ import annotations

import gc
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           DataCollatorWithPadding, EarlyStoppingCallback,
                           Trainer, TrainerCallback, TrainingArguments,
                           set_seed)

from data_utils import MODELS

LR_CANDIDATES = [2e-5, 3e-5, 5e-5]
SEEDS_MAIN = [42, 123, 2026]
SEED_SEARCH = 42
BATCH_SIZE = 16
FALLBACK_BATCH = 8
MAX_EPOCHS = 12
PATIENCE = 2
STRIDE_FACTOR = 2
LENGTH_CHOICES = [128, 256, 384, 512]
MAX_LENGTH_REFERENCE = {"news": 512, "social_media": 256, "wikipedia": 512}


def model_cap(tok) -> int:
    m = getattr(tok, "model_max_length", 512)
    return 512 if (m is None or m > 100000) else int(m)


def compute_max_length_by_model(models: dict, reference_df: pd.DataFrame) -> dict:
    """Per-model, per-domain window: the smallest of LENGTH_CHOICES covering the 95th
    percentile token length of up to 300 documents per class in that domain."""
    out = {}
    for nice, hf in models.items():
        tok = AutoTokenizer.from_pretrained(hf, use_fast=True)
        cap = model_cap(tok)
        budget = {}
        for dom in sorted(reference_df.domain.unique()):
            probe_txt = (reference_df[reference_df.domain == dom]
                         .groupby("label", group_keys=False).head(300).text)
            lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"])
                              for t in probe_txt])
            p95 = int(np.percentile(lens, 95))
            budget[dom] = min(next((c for c in LENGTH_CHOICES if c >= p95), 512), cap)
        out[nice] = budget
        del tok
    return out


def make_ds(d: pd.DataFrame, tok, budget: dict, stride_factor: int = STRIDE_FACTOR):
    if not tok.is_fast:
        raise RuntimeError(f"{tok.__class__.__name__} is slow; sliding window needs a fast tokenizer.")
    cap = model_cap(tok)
    input_ids, attn, labels, doc_ids = [], [], [], []
    for doc_pos, (_, row) in enumerate(d.iterrows()):
        max_len = min(budget.get(row["domain"], 256), cap)
        step = max(1, max_len // stride_factor)
        overlap = max_len - step
        enc = tok(row["text"], max_length=max_len, truncation=True,
                   return_overflowing_tokens=True, stride=overlap)
        for ids, am in zip(enc["input_ids"], enc["attention_mask"]):
            input_ids.append(ids); attn.append(am)
            labels.append(row["label"]); doc_ids.append(doc_pos)
    ds = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attn, "labels": labels})
    return ds, np.array(doc_ids)


def pool_to_docs(logits, doc_ids, n_docs):
    summed = np.zeros((n_docs, logits.shape[1])); counts = np.zeros(n_docs)
    np.add.at(summed, doc_ids, logits); np.add.at(counts, doc_ids, 1)
    return np.argmax(summed / counts[:, None], axis=1)


class Engine:
    """Bundles the mutable eval-time state and disk paths for one results
    directory, so Engines writing to different directories never share run logs."""

    def __init__(self, results_dir: str, max_length_by_model: dict, models: dict = None):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.epoch_log = os.path.join(results_dir, "epoch_log.csv")
        self.runs_log = os.path.join(results_dir, "runs.csv")
        self.best_json = os.path.join(results_dir, "best_so_far.json")
        self.max_length_by_model = max_length_by_model
        self.models = models or MODELS   # per-instance, so extra models don't touch the global default
        self._eval = {"doc_ids": None, "labels": None}

    def pred_path(self, run_id: str) -> str:
        return f"{self.results_dir}/preds_{run_id.replace('|', '_').replace('/', '-')}.npy"

    def metrics_fn(self, p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if self._eval["doc_ids"] is None:
            pred, true = np.argmax(logits, axis=1), p.label_ids
        else:
            true = self._eval["labels"]
            pred = pool_to_docs(logits, self._eval["doc_ids"], len(true))
        return {"macro_f1": f1_score(true, pred, average="macro"),
                "accuracy": accuracy_score(true, pred)}

    def already_done(self, run_id: str):
        if not os.path.exists(self.runs_log):
            return None
        hit = pd.read_csv(self.runs_log).query("run_id == @run_id")
        return hit.iloc[-1].to_dict() if len(hit) else None

    def load_runs(self, phase: str | None = None) -> pd.DataFrame:
        if not os.path.exists(self.runs_log):
            return pd.DataFrame()
        r = pd.read_csv(self.runs_log).drop_duplicates("run_id", keep="last")
        return r[r.phase == phase] if phase else r

    def _epoch_logger(self, run_id, model_name, lr, seed):
        engine = self

        class EpochLogger(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kw):
                if not metrics:
                    return
                row = {"run_id": run_id, "model": model_name, "lr": lr, "seed": seed,
                       "epoch": round(state.epoch or 0, 2), "step": state.global_step,
                       "eval_loss": metrics.get("eval_loss"),
                       "macro_f1": metrics.get("eval_macro_f1"),
                       "accuracy": metrics.get("eval_accuracy"),
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                pd.DataFrame([row]).to_csv(engine.epoch_log, mode="a", index=False,
                                            header=not os.path.exists(engine.epoch_log))
                best = json.load(open(engine.best_json)) if os.path.exists(engine.best_json) else {}
                if (row["macro_f1"] or -1) > best.get(run_id, {}).get("macro_f1", -1):
                    best[run_id] = row
                    json.dump(best, open(engine.best_json, "w"), indent=2)
                print(f"    [saved] epoch {row['epoch']}  dev macro-F1 {row['macro_f1']:.4f}", flush=True)

        return EpochLogger()

    def build_args(self, outdir, seed, lr, epochs, batch):
        common = dict(output_dir=outdir, learning_rate=lr,
                      per_device_train_batch_size=batch, per_device_eval_batch_size=64,
                      num_train_epochs=epochs, weight_decay=0.01,
                      max_grad_norm=1.0, save_total_limit=1, load_best_model_at_end=True,
                      metric_for_best_model="macro_f1", greater_is_better=True,
                      seed=seed, fp16=True,
                      logging_steps=100, report_to="none", disable_tqdm=True,
                      eval_strategy="epoch", save_strategy="epoch",
                      warmup_steps=0.1, lr_scheduler_type="linear")
        rejected = set(common) - set(inspect.signature(TrainingArguments.__init__).parameters)
        if rejected:
            raise ValueError(f"TrainingArguments rejects these: {sorted(rejected)}")
        return TrainingArguments(**common)

    def run(self, nice, tr, dv, te, seed, lr, tag, epochs=MAX_EPOCHS, note="", save_model_dir=None):
        run_id = f"{tag}|{nice}|lr{lr}|seed{seed}" + (f"|{note}" if note else "")
        prior = self.already_done(run_id)
        if prior:
            print(f"    skip — already done (test macro-F1 {prior['macro_f1']:.4f})", flush=True)
            pf = self.pred_path(run_id)
            return prior, (np.load(pf) if os.path.exists(pf) else None)

        t0 = time.time()
        set_seed(seed)
        model_name = self.models[nice]
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        budget = self.max_length_by_model.get(nice, MAX_LENGTH_REFERENCE)
        train_ds, _ = make_ds(tr, tok, budget)
        dev_ds, dev_docs = make_ds(dv, tok, budget)
        test_ds, test_docs = make_ds(te, tok, budget)

        self._eval["doc_ids"], self._eval["labels"] = dev_docs, np.asarray(dv["label"])

        def _make_trainer(bs, mdl):
            return Trainer(
                model=mdl, args=self.build_args(f"/tmp/said_hat_out_{tag}_{seed}", seed, lr, epochs, bs),
                train_dataset=train_ds, eval_dataset=dev_ds,
                data_collator=DataCollatorWithPadding(tok), compute_metrics=self.metrics_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE),
                           self._epoch_logger(run_id, model_name, lr, seed)])

        used_batch = BATCH_SIZE
        trainer = _make_trainer(used_batch, model)
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError:
            print(f"    !! OOM at batch {BATCH_SIZE} — retrying at {FALLBACK_BATCH}", flush=True)
            del trainer; gc.collect(); torch.cuda.empty_cache()
            set_seed(seed)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            used_batch = FALLBACK_BATCH
            trainer = _make_trainer(used_batch, model)
            trainer.train()

        dev_f1 = trainer.evaluate()["eval_macro_f1"]
        stopped_at = round(trainer.state.epoch or epochs, 2)

        if save_model_dir:
            os.makedirs(save_model_dir, exist_ok=True)
            trainer.save_model(save_model_dir)
            tok.save_pretrained(save_model_dir)
            json.dump(budget, open(os.path.join(save_model_dir, "window_budget.json"), "w"), indent=2)
            print(f"    [checkpoint saved] {save_model_dir}", flush=True)

        y_true = np.asarray(te["label"])
        self._eval["doc_ids"], self._eval["labels"] = test_docs, y_true
        logits = trainer.predict(test_ds).predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred = pool_to_docs(logits, test_docs, len(y_true))

        p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="macro")
        res = {"run_id": run_id, "phase": tag, "Model": nice, "hf": model_name, "note": note,
               "seed": seed, "lr": lr, "batch": used_batch, "budget": json.dumps(budget),
               "epochs_allowed": epochs, "epochs_used": stopped_at,
               "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
               "dev_macro_f1": dev_f1, "macro_f1": f,
               "accuracy": accuracy_score(y_true, pred), "precision": p, "recall": r,
               "minutes": round((time.time() - t0) / 60, 1)}
        pd.DataFrame([res]).to_csv(self.runs_log, mode="a", index=False,
                                    header=not os.path.exists(self.runs_log))
        np.save(self.pred_path(run_id), pred)

        # per-document test predictions, for breakdowns by generator, mode or length
        doc_df = te.reset_index(drop=True).copy()
        doc_df["prediction"] = pred
        doc_df["run_id"] = run_id
        preds_csv = os.path.join(self.results_dir, "test_predictions.csv")
        cols = [c for c in ["text_id", "domain", "label", "prediction", "word_count",
                             "generator", "generation_mode", "run_id"] if c in doc_df.columns]
        doc_df[cols].to_csv(preds_csv, mode="a", index=False, header=not os.path.exists(preds_csv))

        self._eval["doc_ids"], self._eval["labels"] = None, None
        del model, trainer; gc.collect(); torch.cuda.empty_cache()
        return res, pred
