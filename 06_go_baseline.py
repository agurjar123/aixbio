"""
06_go_baseline.py
-----------------
Build GO-term binary feature vectors for each gene and train baseline models.

Data source: Human GOA file from geneontology.org (downloaded locally, no API calls).
  URL: http://geneontology.org/gene-associations/goa_human.gaf.gz (GAF 2.2 format)

Feature per gene: binary vector of GO term membership (one column per unique GO ID
  across all genes in the dataset).

Pair feature: [go_A, go_B, go_A * go_B]  — same construction as embedding conditions.

Models trained here:
  1. Logistic regression (same hyperparameters as main models)
  2. Linear regression on GO features to predict continuous GI score

Saves:
  splits/go_train.npz, splits/go_test.npz
  models/go_lr.pkl
  results/go_linear_regression_continuous.csv
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent / "data"
SPLITS_DIR = Path(__file__).parent / "splits"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
SPLITS_DIR.mkdir(exist_ok=True)

PAIRS_CSV = DATA_DIR / "labeled_pairs.csv"
GENE_SPLIT_FILE = SPLITS_DIR / "gene_split.json"

GOA_URL = "http://geneontology.org/gene-associations/goa_human.gaf.gz"
GOA_FILE = DATA_DIR / "goa_human.gaf.gz"

HEADERS = {"User-Agent": "Mozilla/5.0"}
LABEL_NAMES = {0: "SL", 1: "buffering", 2: "neutral"}


# ---------------------------------------------------------------------------
# Download GO annotation file
# ---------------------------------------------------------------------------

def download_goa():
    if GOA_FILE.exists():
        print(f"  GOA file cached: {GOA_FILE.name} ({GOA_FILE.stat().st_size // (1024**2)} MB)")
        return
    print(f"  Downloading Human GOA from geneontology.org...")
    r = requests.get(GOA_URL, stream=True, timeout=300, headers=HEADERS)
    r.raise_for_status()
    with open(GOA_FILE, "wb") as fh:
        for chunk in r.iter_content(65536):
            fh.write(chunk)
    print(f"  Saved {GOA_FILE.name} ({GOA_FILE.stat().st_size // (1024**2)} MB)")


# ---------------------------------------------------------------------------
# Parse GAF 2.2 file
# ---------------------------------------------------------------------------

def parse_goa(target_genes: set) -> dict:
    """
    Parse GAF 2.2 file and return {gene_symbol: set_of_go_ids}.
    Filters to genes in target_genes and excludes IEA (inferred from electronic annotation)
    to focus on experimentally or manually curated terms.
    """
    gene_go = {}
    n_lines = 0
    with gzip.open(GOA_FILE, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):  # header/comment lines
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            n_lines += 1

            # GAF 2.2 columns:
            # 0: DB, 1: DB_Object_ID, 2: DB_Object_Symbol, 3: Qualifier,
            # 4: GO_ID, 5: DB:Reference, 6: Evidence_Code, 7: With/From,
            # 8: Aspect, 9: DB_Object_Name, 10: DB_Object_Synonym, 11: DB_Object_Type, ...
            symbol = parts[2].strip()
            go_id = parts[4].strip()
            evidence = parts[6].strip()

            if symbol not in target_genes:
                continue
            # Optionally exclude IEA (electronic annotation only)
            # Keep IEA to maximize annotation coverage for this PoC
            if not go_id.startswith("GO:"):
                continue

            if symbol not in gene_go:
                gene_go[symbol] = set()
            gene_go[symbol].add(go_id)

    print(f"  Parsed {n_lines:,} GAF lines.")
    print(f"  Genes with GO annotations: {len(gene_go)} / {len(target_genes)}")
    missing = target_genes - set(gene_go.keys())
    if missing:
        print(f"  Genes with NO GO annotations ({len(missing)}): {sorted(missing)[:20]}")
    return gene_go


# ---------------------------------------------------------------------------
# Build binary GO vectors
# ---------------------------------------------------------------------------

def build_go_matrix(gene_go: dict, all_genes: list) -> tuple:
    """
    Build binary GO membership matrix.
    Returns: (gene_list, go_terms_sorted, matrix) where matrix is ndarray (n_genes, n_terms).
    Genes with no annotations get all-zero rows.
    """
    all_go_terms = sorted(set(go_id for terms in gene_go.values() for go_id in terms))
    go_idx = {go: i for i, go in enumerate(all_go_terms)}
    n_genes = len(all_genes)
    n_terms = len(all_go_terms)

    print(f"  Total unique GO terms: {n_terms}")

    matrix = np.zeros((n_genes, n_terms), dtype=np.float32)
    for i, gene in enumerate(all_genes):
        for go_id in gene_go.get(gene, set()):
            if go_id in go_idx:
                matrix[i, go_idx[go_id]] = 1.0

    return all_genes, all_go_terms, matrix


def build_pair_features(vec_A: np.ndarray, vec_B: np.ndarray) -> np.ndarray:
    return np.concatenate([vec_A, vec_B, vec_A * vec_B])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PAIRS_CSV.exists():
        print(f"[ERROR] {PAIRS_CSV} not found. Run 01_download_data.py first.")
        sys.exit(1)
    if not GENE_SPLIT_FILE.exists():
        print(f"[ERROR] {GENE_SPLIT_FILE} not found. Run 05_build_dataset.py first.")
        sys.exit(1)

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.utils.class_weight import compute_sample_weight

    pairs = pd.read_csv(PAIRS_CSV)
    all_genes = sorted(set(pairs["gene_A"]) | set(pairs["gene_B"]))
    print(f"Target genes: {len(all_genes)}")

    with open(GENE_SPLIT_FILE) as f:
        split_data = json.load(f)
    train_genes = set(split_data["train_genes"])
    test_genes = set(split_data["test_genes"])

    # --- Download and parse GO annotations ---
    print("\n[Step 1] GO annotations...")
    download_goa()
    gene_go = parse_goa(set(all_genes))

    # --- Build per-gene GO vectors ---
    print("\n[Step 2] Building binary GO membership matrix...")
    gene_list, go_terms, go_matrix = build_go_matrix(gene_go, all_genes)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # --- Assign pairs to splits ---
    def pair_split(row):
        if row["gene_A"] in test_genes or row["gene_B"] in test_genes:
            return "test"
        return "train"

    pairs["split"] = pairs.apply(pair_split, axis=1)
    train_pairs = pairs[pairs["split"] == "train"]
    test_pairs = pairs[pairs["split"] == "test"]
    print(f"\nTrain pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # --- Build feature matrices ---
    print("\n[Step 3] Building GO pair feature matrices...")

    def build_X_y(split_pairs):
        X_list, y_list, gi_list = [], [], []
        for _, row in split_pairs.iterrows():
            ga, gb = row["gene_A"], row["gene_B"]
            if ga not in gene_to_idx or gb not in gene_to_idx:
                continue
            vec_A = go_matrix[gene_to_idx[ga]]
            vec_B = go_matrix[gene_to_idx[gb]]
            X_list.append(build_pair_features(vec_A, vec_B))
            y_list.append(row["label_int"])
            gi_list.append(row["GI_score"])
        if not X_list:
            return None, None, None
        return np.stack(X_list), np.array(y_list, dtype=np.int32), np.array(gi_list, dtype=np.float32)

    X_train, y_train, gi_train = build_X_y(train_pairs)
    X_test, y_test, gi_test = build_X_y(test_pairs)

    if X_train is None:
        print("[ERROR] No valid pairs for GO baseline.")
        sys.exit(1)

    print(f"GO feature dim: {X_train.shape[1]}")
    print(f"Train: {X_train.shape[0]} pairs, Test: {X_test.shape[0]} pairs")

    # Save GO feature splits
    np.savez_compressed(SPLITS_DIR / "go_train.npz", X=X_train, y=y_train, gi=gi_train)
    np.savez_compressed(SPLITS_DIR / "go_test.npz", X=X_test, y=y_test, gi=gi_test)
    print("Saved GO splits.")

    # --- Logistic regression (classification) ---
    print("\n[Step 4] Training GO logistic regression...")
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, multi_class="multinomial",
        solver="lbfgs", random_state=42, n_jobs=-1
    )
    lr.fit(X_train, y_train)

    y_prob_test = lr.predict_proba(X_test)  # (n_test, 3)
    macro_auroc = roc_auc_score(y_test, y_prob_test, multi_class="ovr", average="macro")
    # Per-class AUPRC then macro-average
    auprc_per_class = []
    for cls in range(3):
        y_bin = (y_test == cls).astype(int)
        if y_bin.sum() == 0:
            continue
        auprc_per_class.append(average_precision_score(y_bin, y_prob_test[:, cls]))
    macro_auprc = np.mean(auprc_per_class)

    print(f"  GO LR — Macro-AUROC: {macro_auroc:.4f}, Macro-AUPRC: {macro_auprc:.4f}")

    with open(MODELS_DIR / "go_lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    print("  Saved GO logistic regression model.")

    # --- Linear regression on GI score (continuous) ---
    print("\n[Step 5] Linear regression on GO features → continuous GI score...")
    linreg = LinearRegression()
    linreg.fit(X_train, gi_train)
    gi_pred_train = linreg.predict(X_train)
    gi_pred_test = linreg.predict(X_test)

    train_r2 = 1 - np.sum((gi_train - gi_pred_train)**2) / np.sum((gi_train - gi_train.mean())**2)
    test_r2 = 1 - np.sum((gi_test - gi_pred_test)**2) / np.sum((gi_test - gi_test.mean())**2)
    print(f"  GO LinReg — Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

    # Save continuous predictions
    pd.DataFrame({
        "gi_true": gi_test,
        "gi_pred": gi_pred_test,
        "label_int": y_test,
    }).to_csv(RESULTS_DIR / "go_linear_regression_continuous.csv", index=False)

    print("\n=== GO Baseline Results ===")
    print(f"  Classification (LR) — Macro-AUROC: {macro_auroc:.4f}, Macro-AUPRC: {macro_auprc:.4f}")
    print(f"  Regression (LinReg) — Test R²: {test_r2:.4f}")

    # Save summary for use by 07_train_evaluate.py
    summary = {
        "go_lr_auroc": macro_auroc,
        "go_lr_auprc": macro_auprc,
        "go_linreg_test_r2": test_r2,
    }
    with open(RESULTS_DIR / "go_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
