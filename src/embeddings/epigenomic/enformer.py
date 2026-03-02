"""
03_enformer_embeddings.py
--------------------------
Extract Enformer embeddings for each gene's 196,608 bp promoter sequence.

Model: EleutherAI/enformer-official-rough (via enformer-pytorch package)
Input: enformer_seq from tss_windows.csv (exactly 196,608 bp)
Output: K562 DNASE and H3K27ac track values at center ±5 bins (22-dim vector per gene)

Caches: embeddings/enformer_{gene_name}.npy  (shape: [22,])
Skips genes already cached.

Runtime note: CPU forward pass takes ~30-120s per gene. GPU strongly recommended.
  On CPU with many genes, consider running overnight or on a subset first.
"""

import sys
import io
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
EMB_DIR = Path(__file__).parent / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

TSS_CSV = DATA_DIR / "tss_windows.csv"
ENFORMER_LEN = 196_608

# Enformer output: 896 bins total, covering the center 114,688 bp
# Center bin index (0-based): 896 // 2 = 448, but indexing: use 447 (the bin straddling center)
# We extract center ±5 bins = 11 bins × 2 tracks = 22 dimensions
N_BINS_HALF = 5  # bins on each side of center
ENFORMER_OUTPUT_BINS = 896

# K562 track identifiers to look for in targets_human.txt
K562_TRACK_PATTERNS = {
    "DNASE": ["DNASE:K562", "DNase:K562", "dnase_k562", "K562_DNASE"],
    "H3K27ac": ["CHIP:H3K27ac:K562", "H3K27ac:K562", "k562_h3k27ac", "K562_H3K27ac"],
}

TARGETS_URL = (
    "https://raw.githubusercontent.com/calico/basenji/master/"
    "manuscripts/cross2020/targets_human.txt"
)
TARGETS_FILE = DATA_DIR / "enformer_targets_human.txt"

HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def check_dependencies():
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import enformer_pytorch  # noqa: F401
    except ImportError:
        missing.append("enformer-pytorch")
    if missing:
        print(
            f"[ERROR] Missing dependencies: {missing}\n"
            f"Install with: pip install {' '.join(missing)}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Track index discovery
# ---------------------------------------------------------------------------

def download_targets_file():
    if TARGETS_FILE.exists():
        return
    print("  Downloading Enformer targets_human.txt...")
    r = requests.get(TARGETS_URL, timeout=60, headers=HEADERS)
    r.raise_for_status()
    TARGETS_FILE.write_bytes(r.content)
    print("  Saved targets_human.txt")


def find_k562_track_indices():
    """
    Parse targets_human.txt and find track indices for K562 DNASE and H3K27ac.
    Returns dict: {"DNASE": [idx, ...], "H3K27ac": [idx, ...]}
    """
    download_targets_file()
    df = pd.read_csv(TARGETS_FILE, sep="\t", header=0)

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    # Common column names: 'index', 'genome', 'strand', 'experiment' or 'description'
    desc_col = None
    for candidate in ["description", "experiment", "identifier", "target"]:
        if candidate in df.columns:
            desc_col = candidate
            break
    if desc_col is None:
        desc_col = df.columns[-1]  # last column is usually the description

    result = {"DNASE": [], "H3K27ac": []}
    for i, row in df.iterrows():
        desc = str(row[desc_col]).upper()
        for track_type, patterns in K562_TRACK_PATTERNS.items():
            for pat in patterns:
                if pat.upper() in desc:
                    result[track_type].append(i)
                    break

    print(f"  K562 DNASE track indices: {result['DNASE']}")
    print(f"  K562 H3K27ac track indices: {result['H3K27ac']}")

    if not result["DNASE"] or not result["H3K27ac"]:
        print("  WARNING: Could not find K562 tracks by pattern matching.")
        print("  Attempting fuzzy match for K562...")
        for i, row in df.iterrows():
            desc = str(row[desc_col]).upper()
            if "K562" in desc:
                if "DNASE" in desc and not result["DNASE"]:
                    result["DNASE"].append(i)
                    print(f"    Fuzzy DNASE match: idx={i}, '{row[desc_col]}'")
                elif "H3K27AC" in desc and not result["H3K27ac"]:
                    result["H3K27ac"].append(i)
                    print(f"    Fuzzy H3K27ac match: idx={i}, '{row[desc_col]}'")

    if not result["DNASE"] or not result["H3K27ac"]:
        print("  WARNING: Still missing tracks. Will use first K562 tracks found.")
        for i, row in df.iterrows():
            desc = str(row[desc_col]).upper()
            if "K562" in desc:
                if not result["DNASE"]:
                    result["DNASE"].append(i)
                    print(f"    Fallback track 1: idx={i}, '{row[desc_col]}'")
                elif not result["H3K27ac"]:
                    result["H3K27ac"].append(i)
                    print(f"    Fallback track 2: idx={i}, '{row[desc_col]}'")
                if result["DNASE"] and result["H3K27ac"]:
                    break

    return result


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(seq: str):
    """
    Convert DNA sequence to one-hot tensor [L, 4].
    N and other ambiguous bases → [0.25, 0.25, 0.25, 0.25].
    Returns numpy array of shape (L, 4).
    """
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = BASE_TO_IDX.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
        else:
            arr[i, :] = 0.25  # N or ambiguous
    return arr


# ---------------------------------------------------------------------------
# Enformer forward pass
# ---------------------------------------------------------------------------

def run_enformer(model, seq: str, track_indices: dict, device):
    """
    Run Enformer forward pass on a single 196,608 bp sequence.
    Returns a 1D numpy array: K562 DNASE and H3K27ac values at center ±N_BINS_HALF bins.
    """
    import torch

    ohe = one_hot_encode(seq)  # (196608, 4)
    # enformer_pytorch expects (batch, length, 4) or can accept (length, 4)
    x = torch.tensor(ohe, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, 4)

    with torch.no_grad():
        out = model(x)
        # enformer_pytorch returns a dict with 'human' key: tensor (1, 896, 5313)
        if isinstance(out, dict):
            preds = out["human"]  # (1, 896, 5313)
        else:
            preds = out
        preds = preds.squeeze(0).cpu().numpy()  # (896, 5313)

    center = ENFORMER_OUTPUT_BINS // 2  # = 448, center bin
    bin_start = max(0, center - N_BINS_HALF)
    bin_end = min(ENFORMER_OUTPUT_BINS, center + N_BINS_HALF + 1)
    center_bins = preds[bin_start:bin_end, :]  # (11, 5313)

    features = []
    for track_type in ["DNASE", "H3K27ac"]:
        idxs = track_indices[track_type]
        if idxs:
            # Mean over matching tracks, then concatenate bin values
            track_vals = center_bins[:, idxs].mean(axis=1)  # (11,)
        else:
            track_vals = np.zeros(bin_end - bin_start, dtype=np.float32)
        features.append(track_vals)

    return np.concatenate(features).astype(np.float32)  # (22,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_dependencies()

    import torch
    from enformer_pytorch import Enformer

    if not TSS_CSV.exists():
        print(f"[ERROR] {TSS_CSV} not found. Run 02_get_tss.py first.")
        sys.exit(1)

    tss_df = pd.read_csv(TSS_CSV)
    genes = tss_df["gene"].tolist()
    print(f"\nTotal genes to embed: {len(genes)}")

    # --- Find K562 track indices ---
    print("\n[Step 1] Locating K562 DNASE and H3K27ac track indices...")
    track_indices = find_k562_track_indices()

    # --- Load Enformer model ---
    print("\n[Step 2] Loading Enformer model from HuggingFace...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cpu":
        print("  WARNING: Running Enformer on CPU will be slow (~30-120s per gene).")

    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model = model.to(device)
    model.eval()
    print("  Model loaded.")

    # --- Process genes ---
    print("\n[Step 3] Extracting embeddings...")
    cached = 0
    processed = 0
    failed = []

    for _, row in tss_df.iterrows():
        gene = row["gene"]
        cache_path = EMB_DIR / f"enformer_{gene}.npy"

        if cache_path.exists():
            cached += 1
            continue

        enformer_seq = row["enformer_seq"]
        if len(enformer_seq) != ENFORMER_LEN:
            print(f"  [WARN] {gene}: enformer_seq length {len(enformer_seq)} != {ENFORMER_LEN}, skipping")
            failed.append(gene)
            continue

        t0 = time.time()
        try:
            embedding = run_enformer(model, enformer_seq, track_indices, device)
            np.save(cache_path, embedding)
            elapsed = time.time() - t0
            processed += 1
            if processed % 10 == 1 or processed <= 5:
                print(f"  [{processed}] {gene}: shape={embedding.shape}, elapsed={elapsed:.1f}s")
        except Exception as e:
            print(f"  [ERROR] {gene}: {e}")
            failed.append(gene)

    total_cached = cached + processed
    print(f"\nDone. Cached={cached}, Newly processed={processed}, Failed={len(failed)}")
    print(f"Total embeddings available: {total_cached}/{len(genes)}")
    if failed:
        print(f"Failed genes: {failed}")

    # Verify a cached file
    sample_files = list(EMB_DIR.glob("enformer_*.npy"))
    if sample_files:
        sample = np.load(sample_files[0])
        print(f"Sample embedding shape: {sample.shape} (expected (22,))")


if __name__ == "__main__":
    main()
