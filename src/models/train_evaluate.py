"""
07_train_evaluate.py
--------------------
Train classifiers on each embedding condition and evaluate on held-out test pairs.

Models:
  - LogisticRegression (L2, balanced class weights, multiclass=multinomial)
  - XGBClassifier (multi:softprob, balanced via compute_sample_weight)

Baselines:
  - Majority class: predicts neutral (class 2) for all test samples
  - GO baseline: results loaded from go_baseline_summary.json (computed in 06)

Evaluation metrics:
  - Macro-AUROC (OvR)
  - Macro-AUPRC (OvR, then averaged)

Also prints:
  - Train/test pair counts and class distributions per condition
  - Final results table

Saves:
  - models/{condition}_{model_type}.pkl
  - results/predictions_{condition}_{model_type}.csv
  - results/evaluation_summary.csv  (final results table)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SPLITS_DIR = Path(__file__).parent / "splits"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_NAMES = {0: "SL", 1: "buffering", 2: "neutral"}
N_CLASSES = 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(condition: str, split: str) -> tuple:
    """Load (X, y) from splits/{condition}_{split}.npz."""
    path = SPLITS_DIR / f"{condition}_{split}.npz"
    if not path.exists():
        return None, None, None, None
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    ga = data["gene_A"]
    gb = data["gene_B"]
    return X, y, ga, gb


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute macro-AUROC and macro-AUPRC for multiclass classification.
    y_prob: (n_samples, n_classes)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Macro-AUROC (OvR)
    try:
        macro_auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError as e:
        # Can happen if a class is absent in y_true
        macro_auroc = float("nan")
        print(f"    AUROC error: {e}")

    # Per-class AUPRC, then macro-average
    auprc_list = []
    for cls in range(N_CLASSES):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue  # skip degenerate classes
        auprc_list.append(average_precision_score(y_bin, y_prob[:, cls]))
    macro_auprc = float(np.mean(auprc_list)) if auprc_list else float("nan")

    return {"macro_auroc": macro_auroc, "macro_auprc": macro_auprc}


def majority_class_metrics(y_true: np.ndarray) -> dict:
    """
    Majority class baseline: predict neutral (class 2) with probability 1.0
    for all samples. Returns macro-AUROC and macro-AUPRC.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    majority_cls = int(pd.Series(y_true).mode()[0])  # most frequent class
    n = len(y_true)

    # Build constant probability matrix
    y_prob = np.zeros((n, N_CLASSES), dtype=np.float32)
    y_prob[:, majority_cls] = 1.0

    metrics = compute_metrics(y_true, y_prob)
    metrics["majority_class"] = LABEL_NAMES.get(majority_cls, str(majority_cls))
    return metrics


def print_class_dist(y: np.ndarray, label: str):
    total = len(y)
    dist = {LABEL_NAMES[c]: int(np.sum(y == c)) for c in range(N_CLASSES)}
    parts = [f"{k}: {v} ({v/total*100:.1f}%)" for k, v in dist.items()]
    print(f"  {label} class distribution (n={total}): " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_logreg(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    return lr


def train_xgboost(X_train, y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [WARN] XGBoost not installed. Skipping XGBoost model.")
        return None

    from sklearn.utils.class_weight import compute_sample_weight

    sample_weight = compute_sample_weight("balanced", y_train)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        tree_method="hist",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    return xgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Discover available conditions
    available_conditions = []
    for f in sorted(SPLITS_DIR.glob("*_train.npz")):
        cond = f.name.replace("_train.npz", "")
        if (SPLITS_DIR / f"{cond}_test.npz").exists():
            available_conditions.append(cond)

    if not available_conditions:
        print("[ERROR] No split files found. Run 05_build_dataset.py first.")
        sys.exit(1)

    print(f"Conditions found: {available_conditions}\n")

    all_results = []

    for condition in available_conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        X_train, y_train, ga_train, gb_train = load_split(condition, "train")
        X_test, y_test, ga_test, gb_test = load_split(condition, "test")

        if X_train is None:
            print(f"  [SKIP] Missing data for {condition}")
            continue

        print(f"  Feature dim: {X_train.shape[1]}")
        print_class_dist(y_train, "TRAIN")
        print_class_dist(y_test, "TEST")
        print(f"  Train pairs: {len(y_train)} | Test pairs: {len(y_test)}")

        # --- Majority class baseline ---
        maj_metrics = majority_class_metrics(y_test)
        print(
            f"\n  [Baseline - Majority Class ({maj_metrics['majority_class']})] "
            f"AUROC={maj_metrics['macro_auroc']:.4f}, AUPRC={maj_metrics['macro_auprc']:.4f}"
        )
        all_results.append({
            "condition": condition,
            "model": "majority_class",
            "macro_auroc": maj_metrics["macro_auroc"],
            "macro_auprc": maj_metrics["macro_auprc"],
            "n_train": len(y_train),
            "n_test": len(y_test),
        })

        # --- Logistic Regression ---
        print(f"\n  [Logistic Regression]")
        lr_model = train_logreg(X_train, y_train)
        y_prob_lr = lr_model.predict_proba(X_test)
        lr_metrics = compute_metrics(y_test, y_prob_lr)
        print(f"    AUROC={lr_metrics['macro_auroc']:.4f}, AUPRC={lr_metrics['macro_auprc']:.4f}")

        with open(MODELS_DIR / f"{condition}_lr.pkl", "wb") as f:
            pickle.dump(lr_model, f)

        pd.DataFrame({
            "gene_A": ga_test,
            "gene_B": gb_test,
            "y_true": y_test,
            "prob_SL": y_prob_lr[:, 0],
            "prob_buffering": y_prob_lr[:, 1],
            "prob_neutral": y_prob_lr[:, 2],
            "y_pred": lr_model.predict(X_test),
        }).to_csv(RESULTS_DIR / f"predictions_{condition}_lr.csv", index=False)

        all_results.append({
            "condition": condition,
            "model": "logistic_regression",
            "macro_auroc": lr_metrics["macro_auroc"],
            "macro_auprc": lr_metrics["macro_auprc"],
            "n_train": len(y_train),
            "n_test": len(y_test),
        })

        # --- XGBoost ---
        print(f"\n  [XGBoost]")
        xgb_model = train_xgboost(X_train, y_train)
        if xgb_model is not None:
            y_prob_xgb = xgb_model.predict_proba(X_test)
            xgb_metrics = compute_metrics(y_test, y_prob_xgb)
            print(f"    AUROC={xgb_metrics['macro_auroc']:.4f}, AUPRC={xgb_metrics['macro_auprc']:.4f}")

            with open(MODELS_DIR / f"{condition}_xgb.pkl", "wb") as f:
                pickle.dump(xgb_model, f)

            pd.DataFrame({
                "gene_A": ga_test,
                "gene_B": gb_test,
                "y_true": y_test,
                "prob_SL": y_prob_xgb[:, 0],
                "prob_buffering": y_prob_xgb[:, 1],
                "prob_neutral": y_prob_xgb[:, 2],
                "y_pred": xgb_model.predict(X_test),
            }).to_csv(RESULTS_DIR / f"predictions_{condition}_xgb.csv", index=False)

            all_results.append({
                "condition": condition,
                "model": "xgboost",
                "macro_auroc": xgb_metrics["macro_auroc"],
                "macro_auprc": xgb_metrics["macro_auprc"],
                "n_train": len(y_train),
                "n_test": len(y_test),
            })

    # --- Add GO baseline results ---
    go_summary_path = RESULTS_DIR / "go_baseline_summary.json"
    if go_summary_path.exists():
        with open(go_summary_path) as f:
            go_summary = json.load(f)
        # Load GO split sizes
        go_train_path = SPLITS_DIR / "go_train.npz"
        go_test_path = SPLITS_DIR / "go_test.npz"
        n_go_train = len(np.load(go_train_path)["y"]) if go_train_path.exists() else None
        n_go_test = len(np.load(go_test_path)["y"]) if go_test_path.exists() else None
        all_results.append({
            "condition": "go_features",
            "model": "logistic_regression",
            "macro_auroc": go_summary.get("go_lr_auroc", float("nan")),
            "macro_auprc": go_summary.get("go_lr_auprc", float("nan")),
            "n_train": n_go_train,
            "n_test": n_go_test,
        })
        print(f"\n[GO Baseline] AUROC={go_summary.get('go_lr_auroc', 'N/A'):.4f}, "
              f"AUPRC={go_summary.get('go_lr_auprc', 'N/A'):.4f}")

    # --- Final results table ---
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["condition", "model"])

    print("\n\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(f"\n{'Condition':<30} {'Model':<25} {'AUROC':>8} {'AUPRC':>8} {'N_train':>9} {'N_test':>8}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        auroc = f"{row['macro_auroc']:.4f}" if not pd.isna(row["macro_auroc"]) else "  N/A  "
        auprc = f"{row['macro_auprc']:.4f}" if not pd.isna(row["macro_auprc"]) else "  N/A  "
        n_tr = str(row["n_train"]) if row["n_train"] else " N/A"
        n_te = str(row["n_test"]) if row["n_test"] else " N/A"
        print(f"{row['condition']:<30} {row['model']:<25} {auroc:>8} {auprc:>8} {n_tr:>9} {n_te:>8}")
    print("="*80)

    results_df.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    print(f"\nSaved full results to {RESULTS_DIR / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    main()
