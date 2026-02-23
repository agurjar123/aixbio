"""
04_evo2_embeddings.py
---------------------
Extract Evo2 embeddings for each gene's ±2 kb promoter sequence.

API: NVIDIA BioNeMo hosted Evo2
  Primary endpoint:  https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward
  Fallback endpoint: https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward

Extracts last-layer hidden state via output_layers=["decoder.final_norm"],
mean-pools across sequence positions → 1D embedding vector per gene.

API key is read from the .env file: EVO2_API_KEY

Caches: embeddings/evo2_{gene_name}.npy
Skips genes already cached.
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
EMB_DIR = Path(__file__).parent / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

ENV_FILE = Path(__file__).parent / ".env"
TSS_CSV = DATA_DIR / "tss_windows.csv"

EVO2_40B_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
EVO2_7B_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-7b/forward"

# Layer to extract: final layer normalization output (before LM head)
# For evo2-40b: 50 layers, TransformerLayers at {3,10,17,24,31,35,42,49}
OUTPUT_LAYER = "decoder.final_norm"

RATE_LIMIT_SLEEP = 1.5  # seconds between API calls
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    """Load EVO2_API_KEY from .env file or environment."""
    api_key = os.environ.get("EVO2_API_KEY", "")
    if not api_key and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("EVO2_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if not api_key:
        print(
            "[ERROR] EVO2_API_KEY not found.\n"
            f"Set it in {ENV_FILE} as: EVO2_API_KEY=nvapi-...\n"
            "or export it as an environment variable."
        )
        sys.exit(1)
    return api_key


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_evo2_api(sequence: str, api_key: str, endpoint_url: str) -> np.ndarray:
    """
    POST to the Evo2 /forward endpoint.
    Returns the mean-pooled last-layer hidden state as a 1D numpy array.

    Response format: Base64-encoded NPZ containing the requested layer tensor.
    Tensor shape from API: typically [seq_len, hidden_dim] or [1, seq_len, hidden_dim].
    We mean-pool over the sequence dimension.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "sequence": sequence,
        "output_layers": [OUTPUT_LAYER],
    }

    r = requests.post(endpoint_url, headers=headers, json=payload, timeout=300)

    if r.status_code == 422:
        # Validation error — print details for debugging
        raise RuntimeError(f"Validation error (422): {r.text[:500]}")
    if r.status_code in (404, 503):
        raise ConnectionError(f"Endpoint unavailable ({r.status_code}): {r.text[:200]}")
    r.raise_for_status()

    resp = r.json()

    # The API returns a dict with layer names as keys, values are Base64-encoded NPZ bytes
    # Key format varies: may be the layer name directly or nested under "outputs"
    layer_data = None

    # Try direct key
    if OUTPUT_LAYER in resp:
        layer_data = resp[OUTPUT_LAYER]
    # Try "outputs" wrapper
    elif "outputs" in resp and OUTPUT_LAYER in resp["outputs"]:
        layer_data = resp["outputs"][OUTPUT_LAYER]
    # Try any key containing the layer name
    else:
        for k, v in resp.items():
            if OUTPUT_LAYER in str(k) or "hidden" in str(k).lower() or "embedding" in str(k).lower():
                layer_data = v
                break
        if layer_data is None:
            # Log all keys for debugging
            available_keys = list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__
            raise RuntimeError(
                f"Could not find layer '{OUTPUT_LAYER}' in API response. "
                f"Available keys: {available_keys}"
            )

    # Decode Base64 → NPZ → numpy array
    if isinstance(layer_data, str):
        npz_bytes = base64.b64decode(layer_data)
        buf = io.BytesIO(npz_bytes)
        with np.load(buf, allow_pickle=False) as npz:
            # The NPZ may have one array or multiple; take the first
            arr_key = list(npz.files)[0]
            tensor = npz[arr_key].astype(np.float32)
    elif isinstance(layer_data, list):
        # Some responses return a nested list directly
        tensor = np.array(layer_data, dtype=np.float32)
    else:
        raise RuntimeError(f"Unexpected layer_data type: {type(layer_data)}")

    # Shape handling: could be [seq_len, hidden_dim] or [1, seq_len, hidden_dim]
    if tensor.ndim == 3:
        tensor = tensor[0]  # remove batch dim → [seq_len, hidden_dim]
    if tensor.ndim != 2:
        raise RuntimeError(f"Unexpected tensor shape: {tensor.shape}")

    # Mean pool over sequence positions → [hidden_dim]
    embedding = tensor.mean(axis=0)
    return embedding


def call_with_retry(sequence: str, api_key: str) -> np.ndarray:
    """Try 40b first, fall back to 7b, with retries."""
    for endpoint_name, url in [("evo2-40b", EVO2_40B_URL), ("evo2-7b", EVO2_7B_URL)]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                emb = call_evo2_api(sequence, api_key, url)
                return emb, endpoint_name
            except ConnectionError as e:
                print(f"    {endpoint_name} unavailable: {e}. Switching endpoint...")
                break  # try next endpoint
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"    Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    {endpoint_name} failed after {MAX_RETRIES} attempts: {e}")
    raise RuntimeError("Both evo2-40b and evo2-7b endpoints failed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = load_api_key()
    print(f"API key loaded (length {len(api_key)}, starts with {api_key[:10]}...)")

    if not TSS_CSV.exists():
        print(f"[ERROR] {TSS_CSV} not found. Run 02_get_tss.py first.")
        sys.exit(1)

    tss_df = pd.read_csv(TSS_CSV)
    genes_to_process = []
    for _, row in tss_df.iterrows():
        gene = row["gene"]
        cache_path = EMB_DIR / f"evo2_{gene}.npy"
        if not cache_path.exists():
            genes_to_process.append((gene, row["evo2_seq"]))

    print(f"\nTotal genes: {len(tss_df)}")
    print(f"Already cached: {len(tss_df) - len(genes_to_process)}")
    print(f"To process: {len(genes_to_process)}")

    if not genes_to_process:
        print("All embeddings already cached.")
        return

    # --- Dry-run on first gene to validate API and print shape ---
    print("\n[Step 1] Dry-run on first gene to verify API response format...")
    first_gene, first_seq = genes_to_process[0]
    print(f"  Gene: {first_gene}, sequence length: {len(first_seq)} bp")
    try:
        emb, endpoint_used = call_with_retry(first_seq, api_key)
        cache_path = EMB_DIR / f"evo2_{first_gene}.npy"
        np.save(cache_path, emb)
        print(f"  SUCCESS: embedding shape = {emb.shape}, endpoint = {endpoint_used}")
        print(f"  Embedding stats: mean={emb.mean():.4f}, std={emb.std():.4f}")
        genes_to_process = genes_to_process[1:]
    except Exception as e:
        print(f"  [FATAL] Dry-run failed: {e}")
        print("  Cannot proceed without a working API connection. Exiting.")
        sys.exit(1)

    time.sleep(RATE_LIMIT_SLEEP)

    # --- Process remaining genes ---
    print(f"\n[Step 2] Processing {len(genes_to_process)} remaining genes...")
    failed = []
    for i, (gene, seq) in enumerate(genes_to_process):
        cache_path = EMB_DIR / f"evo2_{gene}.npy"
        if cache_path.exists():
            continue  # race condition safety

        try:
            emb, endpoint_used = call_with_retry(seq, api_key)
            np.save(cache_path, emb)
            if (i + 1) % 10 == 1 or (i + 1) <= 5:
                print(f"  [{i+2}/{len(tss_df)}] {gene}: shape={emb.shape} via {endpoint_used}")
        except Exception as e:
            print(f"  [ERROR] {gene}: {e}")
            failed.append(gene)

        time.sleep(RATE_LIMIT_SLEEP)

    cached_count = len(list(EMB_DIR.glob("evo2_*.npy")))
    print(f"\nDone. Total embeddings cached: {cached_count}/{len(tss_df)}")
    if failed:
        print(f"Failed genes ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
