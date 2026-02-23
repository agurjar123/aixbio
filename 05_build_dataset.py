"""
05_build_dataset.py
-------------------
Build gene-pair feature matrices and gene-level train/test splits.

Split logic (strict gene-level holdout):
  - 20% of unique genes randomly assigned to held-out set (seed=42)
  - Any pair where EITHER gene is in the held-out set → test set
  - Only pairs where BOTH genes are in training genes → train set
  - This guarantees zero-shot generalization (no held-out gene seen during training)

Pair feature vector for each condition:
  [emb_A | emb_B | emb_A * emb_B]  (concatenation + elementwise product)

Conditions: evo2_only, enformer_only, evo2_enformer_combined

Saves to splits/:
  - gene_split.json          (train_genes, test_genes)
  - {condition}_train.npz    (X, y, gene_A, gene_B)
  - {condition}_test.npz
  - split_summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EMB_DIR = Path(__file__).parent / "embeddings"
SPLITS_DIR = Path(__file__).parent / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

PAIRS_CSV = DATA_DIR / "labeled_pairs.csv"
GENE_SPLIT_FILE = SPLITS_DIR / "gene_split.json"
SUMMARY_FILE = SPLITS_DIR / "split_summary.txt"

RANDOM_SEED = 42
TEST_GENE_FRACTION = 0.20

LABEL_NAMES = {0: "SL", 1: "buffering", 2: "neutral"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embedding(prefix: str, gene: str) -> np.ndarray | None:
    """Load cached embedding, return None if not found."""
    path = EMB_DIR / f"{prefix}_{gene}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def build_pair_features(emb_A: np.ndarray, emb_B: np.ndarray) -> np.ndarray:
    """Concatenate [emb_A, emb_B, emb_A * emb_B]."""
    return np.concatenate([emb_A, emb_B, emb_A * emb_B])


def gene_level_split(unique_genes: list, rng: np.random.Generator) -> tuple[set, set]:
    """Randomly assign 20% of genes to test set."""
    genes_arr = np.array(sorted(unique_genes))
    rng.shuffle(genes_arr)
    n_test = max(1, int(len(genes_arr) * TEST_GENE_FRACTION))
    test_genes = set(genes_arr[:n_test].tolist())
    train_genes = set(genes_arr[n_test:].tolist())
    return train_genes, test_genes


def print_class_distribution(y: np.ndarray, split_name: str, lines: list):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    msg_lines = [f"\n  {split_name} class distribution (n={total}):"]
    for cls, cnt in zip(unique, counts):
        label = LABEL_NAMES.get(int(cls), str(cls))
        msg_lines.append(f"    {label} (class {cls}): {cnt} ({cnt/total*100:.1f}%)")
    for m in msg_lines:
        print(m)
        lines.append(m)


def build_condition(
    pairs: pd.DataFrame,
    train_genes: set,
    test_genes: set,
    emb_prefixes: list,
    condition_name: str,
    summary_lines: list,
) -> bool:
    """
    Build train/test feature matrices for a given embedding condition.
    Returns True if successful.
    """
    train_out = SPLITS_DIR / f"{condition_name}_train.npz"
    test_out = SPLITS_DIR / f"{condition_name}_test.npz"

    if train_out.exists() and test_out.exists():
        print(f"  [{condition_name}] already built, skipping.")
        return True

    print(f"\n[Building condition: {condition_name}]")

    # Load all embeddings needed
    all_genes = set(pairs["gene_A"]) | set(pairs["gene_B"])
    embeddings = {}
    missing_genes = []
    for gene in sorted(all_genes):
        parts = []
        for prefix in emb_prefixes:
            emb = load_embedding(prefix, gene)
            if emb is None:
                parts = None
                break
            parts.append(emb)
        if parts is None:
            missing_genes.append(gene)
        else:
            embeddings[gene] = np.concatenate(parts)

    if missing_genes:
        print(f"  Missing embeddings for {len(missing_genes)} genes: {missing_genes[:10]}{'...' if len(missing_genes)>10 else ''}")

    # Filter pairs to those where both genes have embeddings
    valid_mask = pairs.apply(
        lambda r: r["gene_A"] in embeddings and r["gene_B"] in embeddings, axis=1
    )
    valid_pairs = pairs[valid_mask].copy()
    print(f"  Valid pairs (both genes have embeddings): {len(valid_pairs)} / {len(pairs)}")

    if len(valid_pairs) == 0:
        print(f"  [ERROR] No valid pairs for condition {condition_name}. Skipping.")
        return False

    # Split pairs
    def pair_split(row):
        a_test = row["gene_A"] in test_genes
        b_test = row["gene_B"] in test_genes
        if a_test or b_test:
            return "test"
        return "train"

    valid_pairs["split"] = valid_pairs.apply(pair_split, axis=1)
    train_pairs = valid_pairs[valid_pairs["split"] == "train"]
    test_pairs = valid_pairs[valid_pairs["split"] == "test"]

    print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    # Verify no gene overlap
    train_gene_set = set(train_pairs["gene_A"]) | set(train_pairs["gene_B"])
    test_gene_set = set(test_pairs["gene_A"]) | set(test_pairs["gene_B"])
    overlap = train_gene_set & test_gene_set
    assert not overlap, f"GENE OVERLAP DETECTED in {condition_name}: {overlap}"
    print(f"  Gene overlap check: PASSED (0 genes appear in both splits)")

    # Build feature matrices
    def make_XY(split_df):
        X_list, y_list, ga_list, gb_list = [], [], [], []
        for _, row in split_df.iterrows():
            emb_A = embeddings[row["gene_A"]]
            emb_B = embeddings[row["gene_B"]]
            X_list.append(build_pair_features(emb_A, emb_B))
            y_list.append(row["label_int"])
            ga_list.append(row["gene_A"])
            gb_list.append(row["gene_B"])
        if not X_list:
            return None, None, None, None
        return (
            np.stack(X_list),
            np.array(y_list, dtype=np.int32),
            np.array(ga_list),
            np.array(gb_list),
        )

    X_train, y_train, ga_train, gb_train = make_XY(train_pairs)
    X_test, y_test, ga_test, gb_test = make_XY(test_pairs)

    if X_train is None or X_test is None:
        print(f"  [ERROR] Empty split for {condition_name}.")
        return False

    print(f"  Feature vector dimension: {X_train.shape[1]}")

    # Print class distributions
    print_class_distribution(y_train, "TRAIN", summary_lines)
    print_class_distribution(y_test, "TEST", summary_lines)

    # Save
    np.savez_compressed(
        train_out, X=X_train, y=y_train, gene_A=ga_train, gene_B=gb_train
    )
    np.savez_compressed(
        test_out, X=X_test, y=y_test, gene_A=ga_test, gene_B=gb_test
    )
    print(f"  Saved: {train_out.name}, {test_out.name}")

    summary = (
        f"\n=== {condition_name} ===\n"
        f"  Feature dim: {X_train.shape[1]}\n"
        f"  Train: {len(y_train)} pairs | Test: {len(y_test)} pairs\n"
        f"  Train genes: {len(set(ga_train.tolist()) | set(gb_train.tolist()))} | "
        f"Test genes: {len(set(ga_test.tolist()) | set(gb_test.tolist()))}"
    )
    summary_lines.append(summary)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PAIRS_CSV.exists():
        print(f"[ERROR] {PAIRS_CSV} not found. Run 01_download_data.py first.")
        sys.exit(1)

    pairs = pd.read_csv(PAIRS_CSV)
    print(f"Loaded {len(pairs)} labeled pairs.")

    unique_genes = sorted(set(pairs["gene_A"]) | set(pairs["gene_B"]))
    print(f"Unique genes: {len(unique_genes)}")

    # --- Gene-level split ---
    if GENE_SPLIT_FILE.exists():
        print(f"\nLoading existing gene split from {GENE_SPLIT_FILE}")
        with open(GENE_SPLIT_FILE) as f:
            split_data = json.load(f)
        train_genes = set(split_data["train_genes"])
        test_genes = set(split_data["test_genes"])
    else:
        print(f"\nGenerating gene-level split (seed={RANDOM_SEED})...")
        rng = np.random.default_rng(RANDOM_SEED)
        train_genes, test_genes = gene_level_split(unique_genes, rng)
        split_data = {
            "train_genes": sorted(train_genes),
            "test_genes": sorted(test_genes),
            "seed": RANDOM_SEED,
            "test_fraction": TEST_GENE_FRACTION,
        }
        with open(GENE_SPLIT_FILE, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved gene split: {len(train_genes)} train, {len(test_genes)} test genes")

    overlap = train_genes & test_genes
    assert not overlap, f"Gene sets overlap! {overlap}"
    print(f"Train genes: {len(train_genes)}, Test genes: {len(test_genes)}")
    print(f"Gene overlap: NONE (verified)")

    summary_lines = [
        f"Gene-level split: {len(train_genes)} train genes, {len(test_genes)} test genes",
        f"Seed: {RANDOM_SEED}, Test fraction: {TEST_GENE_FRACTION}",
    ]

    # --- Check which embeddings are available ---
    evo2_available = len(list(EMB_DIR.glob("evo2_*.npy"))) > 0
    enformer_available = len(list(EMB_DIR.glob("enformer_*.npy"))) > 0

    print(f"\nEvo2 embeddings available: {evo2_available}")
    print(f"Enformer embeddings available: {enformer_available}")

    if not evo2_available and not enformer_available:
        print("[ERROR] No embeddings found. Run 03_enformer_embeddings.py and/or 04_evo2_embeddings.py first.")
        sys.exit(1)

    # --- Build datasets for each condition ---
    conditions = []
    if enformer_available:
        conditions.append(("enformer_only", ["enformer"]))
    if evo2_available:
        conditions.append(("evo2_only", ["evo2"]))
    if evo2_available and enformer_available:
        conditions.append(("evo2_enformer_combined", ["evo2", "enformer"]))

    for cond_name, prefixes in conditions:
        build_condition(pairs, train_genes, test_genes, prefixes, cond_name, summary_lines)

    # --- Save summary ---
    summary_text = "\n".join(summary_lines)
    SUMMARY_FILE.write_text(summary_text)
    print(f"\nSummary saved to {SUMMARY_FILE}")
    print(summary_text)


if __name__ == "__main__":
    main()
