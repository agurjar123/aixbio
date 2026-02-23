"""
01_download_data.py
-------------------
Download Horlbeck et al. 2018 (Cell, DOI: 10.1016/j.cell.2018.06.010) supplementary tables.
- Table S4: single-perturbation phenotypes (gamma scores) — used as filter only
- Table S5: gene-pair GI scores — source of pair universe and labels

Label logic (applied to filtered pairs only):
  SL       = GI_score <= 10th percentile
  buffering = GI_score >= 90th percentile
  neutral   = middle 80%

Saves: data/labeled_pairs.csv
"""

import os
import sys
import re
import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_PATH = DATA_DIR / "labeled_pairs.csv"

# ---------------------------------------------------------------------------
# URL discovery helpers
# ---------------------------------------------------------------------------

PAPER_SUPPLEMENTAL_URL = (
    "https://www.cell.com/cell/supplemental/S0092-8674(18)30735-9"
)
PAPER_FULLTEXT_URL = (
    "https://www.cell.com/cell/fulltext/S0092-8674(18)30735-9"
)
# Mendeley repository for Horlbeck 2018
MENDELEY_API = "https://data.mendeley.com/api/datasets/rdzk59n6j4/versions/1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def discover_cell_press_urls():
    """Fetch the paper supplemental page and extract mmc*.xlsx links."""
    print("  Attempting to discover Cell Press supplementary URLs...")
    try:
        r = requests.get(PAPER_SUPPLEMENTAL_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        # Cell Press attachment URLs contain /attachment/.../mmc{N}.xlsx
        links = re.findall(
            r'https://www\.cell\.com/cms/[^\s"\']+mmc\d+\.xlsx', r.text
        )
        if not links:
            # Try the fulltext page
            r2 = requests.get(PAPER_FULLTEXT_URL, headers=HEADERS, timeout=30)
            links = re.findall(
                r'https://www\.cell\.com/cms/[^\s"\']+mmc\d+\.xlsx', r2.text
            )
        links = sorted(set(links))
        print(f"  Found {len(links)} supplementary xlsx link(s): {links}")
        return links
    except Exception as e:
        print(f"  Cell Press discovery failed: {e}")
        return []


def discover_mendeley_urls():
    """Query Mendeley Data API for file listings."""
    print("  Attempting Mendeley Data API...")
    try:
        r = requests.get(MENDELEY_API, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        files = data.get("files", [])
        result = {}
        for f in files:
            name = f.get("filename", "")
            # Look for files with table names in them
            dl_url = f.get("content_details", {}).get("download_url", "")
            if not dl_url:
                # Try alternate key
                dl_url = f.get("download_url", "")
            if name and dl_url:
                result[name] = dl_url
        print(f"  Mendeley returned {len(result)} file(s)")
        return result
    except Exception as e:
        print(f"  Mendeley API failed: {e}")
        return {}


def download_file(url, dest_path, description="file"):
    """Download a file from URL to dest_path with progress output."""
    if dest_path.exists():
        print(f"  Cached: {dest_path.name}")
        return True
    print(f"  Downloading {description} from {url[:80]}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
        print(f"  Saved {dest_path.name} ({dest_path.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Identify which mmc file is S4 and which is S5
# ---------------------------------------------------------------------------

def identify_table(xlsx_path):
    """
    Return ('S4', df) or ('S5', df) or (None, None) based on content.
    Table S4 has single-gene phenotype columns; Table S5 has GI score columns.
    """
    try:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, nrows=5)
            cols = [str(c).lower() for c in df.columns]
            col_str = " ".join(cols)
            # S4 typically has columns about single perturbation / gamma / phenotype
            if any(k in col_str for k in ["gamma", "phenotype", "single", "growth"]):
                full_df = xl.parse(sheet)
                return "S4", full_df
            # S5 has columns about GI score / interaction / double
            if any(k in col_str for k in ["gi", "interaction", "double", "score"]):
                full_df = xl.parse(sheet)
                return "S5", full_df
        # Try by first column heuristic — S4 has gene column, S5 has gene_A/gene_B
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, nrows=3)
            cols_str = " ".join(str(c).lower() for c in df.columns)
            if "gene_a" in cols_str or "gene a" in cols_str:
                full_df = xl.parse(sheet)
                return "S5", full_df
        return None, None
    except Exception as e:
        print(f"    Could not read {xlsx_path.name}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Parse S4 (single phenotypes)
# ---------------------------------------------------------------------------

def parse_s4(df):
    """
    Return a Series mapping gene_name -> single_phenotype (float).
    Handles various column naming conventions used in Horlbeck tables.
    """
    df.columns = [str(c).strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}

    # Find gene column
    gene_col = None
    for candidate in ["gene", "gene_name", "genename", "target_gene", "symbol"]:
        if candidate in col_lower:
            gene_col = col_lower[candidate]
            break
    if gene_col is None:
        # Try first column
        gene_col = df.columns[0]
        print(f"    S4: using first column '{gene_col}' as gene column")

    # Find phenotype column — gamma, phenotype, growth score, rho
    pheno_col = None
    for candidate in ["gamma", "phenotype", "growth", "rho", "fitness", "score", "mean"]:
        for orig_lower, orig in col_lower.items():
            if candidate in orig_lower and orig != gene_col:
                pheno_col = orig
                break
        if pheno_col:
            break
    if pheno_col is None:
        # Use the second numeric column
        for col in df.columns:
            if col != gene_col and pd.api.types.is_numeric_dtype(df[col]):
                pheno_col = col
                break
    if pheno_col is None:
        raise ValueError(f"Cannot identify phenotype column in S4. Columns: {list(df.columns)}")

    print(f"    S4 columns used: gene='{gene_col}', phenotype='{pheno_col}'")
    s4 = df[[gene_col, pheno_col]].dropna(subset=[gene_col])
    s4 = s4.rename(columns={gene_col: "gene", pheno_col: "single_phenotype"})
    s4["gene"] = s4["gene"].astype(str).str.strip()
    s4["single_phenotype"] = pd.to_numeric(s4["single_phenotype"], errors="coerce")
    s4 = s4.dropna(subset=["single_phenotype"])
    return s4.set_index("gene")["single_phenotype"]


# ---------------------------------------------------------------------------
# Parse S5 (GI scores)
# ---------------------------------------------------------------------------

def parse_s5(df):
    """
    Return a DataFrame with columns [gene_A, gene_B, GI_score].
    Handles multiple naming conventions.
    """
    df.columns = [str(c).strip() for c in df.columns]
    col_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    # Find gene A column
    gene_a_col = None
    for cand in ["gene_a", "genea", "gene1", "query_gene", "gene_query"]:
        if cand in col_lower:
            gene_a_col = col_lower[cand]
            break
    if gene_a_col is None:
        gene_a_col = df.columns[0]
        print(f"    S5: using '{gene_a_col}' as gene_A column")

    # Find gene B column
    gene_b_col = None
    for cand in ["gene_b", "geneb", "gene2", "library_gene", "target_gene"]:
        if cand in col_lower:
            gene_b_col = col_lower[cand]
            break
    if gene_b_col is None:
        gene_b_col = df.columns[1]
        print(f"    S5: using '{gene_b_col}' as gene_B column")

    # Find GI score column
    gi_col = None
    for cand in ["gi_score", "gi", "score", "interaction_score", "epsilon", "double"]:
        for orig_lower, orig in col_lower.items():
            if cand in orig_lower and orig not in (gene_a_col, gene_b_col):
                gi_col = orig
                break
        if gi_col:
            break
    if gi_col is None:
        # Use first numeric column after gene columns
        for col in df.columns:
            if col not in (gene_a_col, gene_b_col) and pd.api.types.is_numeric_dtype(df[col]):
                gi_col = col
                break
    if gi_col is None:
        raise ValueError(f"Cannot identify GI score column in S5. Columns: {list(df.columns)}")

    print(f"    S5 columns used: gene_A='{gene_a_col}', gene_B='{gene_b_col}', GI='{gi_col}'")
    s5 = df[[gene_a_col, gene_b_col, gi_col]].copy()
    s5.columns = ["gene_A", "gene_B", "GI_score"]
    s5["gene_A"] = s5["gene_A"].astype(str).str.strip()
    s5["gene_B"] = s5["gene_B"].astype(str).str.strip()
    s5["GI_score"] = pd.to_numeric(s5["GI_score"], errors="coerce")
    s5 = s5.dropna()
    return s5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if OUT_PATH.exists():
        print(f"Output already exists: {OUT_PATH}. Delete to re-run.")
        df = pd.read_csv(OUT_PATH)
        print(f"Loaded {len(df)} labeled pairs. Label distribution:")
        print(df["label"].value_counts().to_string())
        return df

    # --- Step 1: locate / download xlsx files ---
    xlsx_paths = list(DATA_DIR.glob("mmc*.xlsx"))

    if len(xlsx_paths) < 2:
        print("\n[Step 1] Discovering supplementary tables...")
        cell_urls = discover_cell_press_urls()
        for url in cell_urls:
            num = re.search(r"mmc(\d+)", url)
            fname = f"mmc{num.group(1)}.xlsx" if num else "mmc_unknown.xlsx"
            download_file(url, DATA_DIR / fname, description=fname)
        xlsx_paths = list(DATA_DIR.glob("mmc*.xlsx"))

    if len(xlsx_paths) < 2:
        print("\n  Trying Mendeley Data...")
        mendeley_files = discover_mendeley_urls()
        for name, url in mendeley_files.items():
            if name.lower().endswith(".xlsx") or name.lower().endswith(".csv"):
                download_file(url, DATA_DIR / name, description=name)
        xlsx_paths = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.csv"))

    if len(xlsx_paths) < 2:
        print("\n  Falling back to GEO sgRNA-level data (will compute gene-level GI scores)...")
        result = compute_from_geo()
        if result is not None:
            result.to_csv(OUT_PATH, index=False)
            print(f"\nSaved {len(result)} labeled pairs to {OUT_PATH}")
            return result
        print(
            "\n[ERROR] Could not automatically download supplementary tables.\n"
            "Please manually download Table S4 and Table S5 from:\n"
            "  https://www.cell.com/cell/supplemental/S0092-8674(18)30735-9\n"
            "or:\n"
            "  https://data.mendeley.com/datasets/rdzk59n6j4/1\n"
            "Save them as data/mmc4.xlsx (Table S4) and data/mmc5.xlsx (Table S5).\n"
            "Then re-run this script."
        )
        sys.exit(1)

    # --- Step 2: identify S4 and S5 among downloaded files ---
    print("\n[Step 2] Identifying Table S4 and S5 from downloaded files...")
    s4_series = None
    s5_df = None
    unidentified = []

    for path in sorted(xlsx_paths):
        print(f"  Reading {path.name}...")
        table_id, df = identify_table(path)
        if table_id == "S4" and s4_series is None:
            s4_series = parse_s4(df)
            print(f"  -> Identified as Table S4 ({len(s4_series)} genes)")
        elif table_id == "S5" and s5_df is None:
            s5_df = parse_s5(df)
            print(f"  -> Identified as Table S5 ({len(s5_df)} pairs)")
        else:
            unidentified.append((path.name, table_id))

    if s4_series is None or s5_df is None:
        # Try to read all sheets from any remaining files with broader heuristics
        for path in sorted(xlsx_paths):
            if s4_series is not None and s5_df is not None:
                break
            try:
                xl = pd.ExcelFile(path, engine="openpyxl")
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    if df.empty:
                        continue
                    if s5_df is None and df.shape[1] >= 3:
                        # Check if looks like pairs (two string cols + numeric)
                        ncols = df.select_dtypes(include="number").shape[1]
                        if ncols >= 1:
                            try:
                                s5_df = parse_s5(df)
                                if len(s5_df) > 100:
                                    print(f"  -> Forced S5 from {path.name} sheet '{sheet}'")
                                    break
                            except Exception:
                                pass
            except Exception:
                pass

    if s4_series is None:
        print(
            "[ERROR] Could not parse Table S4 (single-perturbation phenotypes).\n"
            "Please ensure the file is present in data/ and named mmc*.xlsx."
        )
        sys.exit(1)
    if s5_df is None:
        print("[ERROR] Could not parse Table S5 (GI scores).")
        sys.exit(1)

    # --- Step 3: join S4 onto S5 ---
    print("\n[Step 3] Joining single phenotypes onto gene pairs...")
    print(f"  S5 pairs before filter: {len(s5_df)}")
    s5_df = s5_df.merge(
        s4_series.rename("pheno_A"), left_on="gene_A", right_index=True, how="left"
    )
    s5_df = s5_df.merge(
        s4_series.rename("pheno_B"), left_on="gene_B", right_index=True, how="left"
    )
    missing_pheno = s5_df[["pheno_A", "pheno_B"]].isna().any(axis=1).sum()
    if missing_pheno > 0:
        print(f"  Warning: {missing_pheno} pairs have missing single phenotypes; dropping.")
    s5_df = s5_df.dropna(subset=["pheno_A", "pheno_B"])

    # --- Step 4: filter to pairs where both single phenotypes < 0 ---
    print("\n[Step 4] Filtering to both-negative single phenotype pairs...")
    filtered = s5_df[(s5_df["pheno_A"] < 0) & (s5_df["pheno_B"] < 0)].copy()
    print(f"  Pairs after filter: {len(filtered)} (from {len(s5_df)})")
    if len(filtered) < 50:
        print("  Warning: very few pairs after filtering. Check phenotype column sign conventions.")

    # --- Step 5: label by empirical percentiles of filtered GI scores ---
    print("\n[Step 5] Computing labels from empirical GI score distribution...")
    p10 = np.percentile(filtered["GI_score"], 10)
    p90 = np.percentile(filtered["GI_score"], 90)
    print(f"  GI score 10th pct: {p10:.4f}, 90th pct: {p90:.4f}")
    print(f"  GI score range: [{filtered['GI_score'].min():.4f}, {filtered['GI_score'].max():.4f}]")

    def assign_label(gi):
        if gi <= p10:
            return "SL"
        elif gi >= p90:
            return "buffering"
        else:
            return "neutral"

    filtered["label"] = filtered["GI_score"].map(assign_label)
    # Encode as integer for model use: 0=SL, 1=buffering, 2=neutral
    label_map = {"SL": 0, "buffering": 1, "neutral": 2}
    filtered["label_int"] = filtered["label"].map(label_map)
    filtered["p10_threshold"] = p10
    filtered["p90_threshold"] = p90

    print("\nLabel distribution:")
    vc = filtered["label"].value_counts()
    for label, count in vc.items():
        print(f"  {label}: {count} ({count/len(filtered)*100:.1f}%)")

    # --- Step 6: save ---
    filtered.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(filtered)} labeled pairs to {OUT_PATH}")
    unique_genes = set(filtered["gene_A"]) | set(filtered["gene_B"])
    print(f"Unique genes in dataset: {len(unique_genes)}")
    return filtered


if __name__ == "__main__":
    main()
