"""
02_get_tss.py
-------------
For every gene in the labeled pairs dataset:
  1. Find its canonical TSS from GENCODE v38 hg38 (prefer APPRIS principal isoform;
     fall back to K562 ENCODE RNA-seq TPM; final fallback: most 5' on canonical chrom).
  2. Extract TWO sequence windows from hg38:
       - evo2_seq  : ±2 kb around TSS (4,000 bp), strand-corrected
       - enformer_seq : 196,608 bp centered on TSS, strand-corrected
     Both are N-padded at chromosome boundaries only.

Saves: data/tss_windows.csv  (gene, chrom, tss, strand, evo2_seq, enformer_seq)
"""

import gzip
import io
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import requests
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PAIRS_CSV = DATA_DIR / "labeled_pairs.csv"
OUT_CSV = DATA_DIR / "tss_windows.csv"

GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_38/gencode.v38.annotation.gtf.gz"
)
GTF_GZ = DATA_DIR / "gencode.v38.annotation.gtf.gz"

# UCSC hg38 per-chromosome fasta
UCSC_CHROM_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{chrom}.fa.gz"
GENOME_DIR = DATA_DIR / "hg38_chroms"
GENOME_DIR.mkdir(exist_ok=True)

EVO2_WINDOW = 2000      # ±2 kb → 4000 bp total
ENFORMER_LEN = 196_608  # exact Enformer input requirement

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# GENCODE GTF download + parse
# ---------------------------------------------------------------------------

def download_gtf():
    if GTF_GZ.exists():
        print(f"  GTF cached: {GTF_GZ.name}")
        return
    print(f"  Downloading GENCODE v38 GTF (~1 GB compressed)...")
    r = requests.get(GENCODE_URL, stream=True, timeout=3600, headers=HEADERS)
    r.raise_for_status()
    with open(GTF_GZ, "wb") as fh:
        for chunk in r.iter_content(65536):
            fh.write(chunk)
    print(f"  Saved {GTF_GZ.stat().st_size // (1024**2)} MB")


def parse_gtf_for_genes(target_genes: set) -> dict:
    """
    Parse GENCODE GTF and return for each gene:
      {gene_name: [(chrom, tss, strand, transcript_id, appris_rank, is_basic), ...]}

    Only transcripts on standard chromosomes (chr1-22, chrX, chrY) are kept.
    appris_rank: 1 = principal_1 (best), higher = worse, 99 = not annotated.
    """
    print("  Parsing GENCODE v38 GTF for target genes...")
    canonical_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

    APPRIS_RANK = {
        "appris_principal_1": 1,
        "appris_principal_2": 2,
        "appris_principal_3": 3,
        "appris_principal_4": 4,
        "appris_principal_5": 5,
        "appris_alternative_1": 10,
        "appris_alternative_2": 11,
    }

    gene_transcripts = defaultdict(list)
    open_fn = gzip.open if str(GTF_GZ).endswith(".gz") else open

    with open_fn(GTF_GZ, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature != "transcript":
                continue

            chrom = parts[0]
            if chrom not in canonical_chroms:
                continue

            strand = parts[6]
            start = int(parts[3])  # 1-based
            end = int(parts[4])    # 1-based, inclusive

            # TSS: start for + strand, end for - strand (1-based)
            tss = start if strand == "+" else end

            attrs = parts[8]

            # Extract gene_name
            gn_match = re.search(r'gene_name "([^"]+)"', attrs)
            if not gn_match:
                continue
            gene_name = gn_match.group(1)
            if gene_name not in target_genes:
                continue

            # Extract transcript_id
            tid_match = re.search(r'transcript_id "([^"]+)"', attrs)
            tid = tid_match.group(1) if tid_match else ""

            # APPRIS rank
            appris_match = re.search(r'tag "?(appris_\w+)"?', attrs)
            appris_rank = APPRIS_RANK.get(appris_match.group(1), 99) if appris_match else 99

            # Is basic transcript?
            is_basic = 1 if 'tag "basic"' in attrs else 0

            # Transcript support level (lower = better)
            tsl_match = re.search(r'transcript_support_level "(\d+)"', attrs)
            tsl = int(tsl_match.group(1)) if tsl_match else 99

            gene_transcripts[gene_name].append({
                "chrom": chrom,
                "tss": tss,
                "strand": strand,
                "transcript_id": tid,
                "appris_rank": appris_rank,
                "is_basic": is_basic,
                "tsl": tsl,
            })

    found = len(gene_transcripts)
    missing = target_genes - set(gene_transcripts.keys())
    print(f"  Found transcripts for {found}/{len(target_genes)} genes.")
    if missing:
        print(f"  Missing genes ({len(missing)}): {sorted(missing)[:20]}{'...' if len(missing)>20 else ''}")

    return gene_transcripts


def select_canonical_tss(gene_transcripts: dict) -> dict:
    """
    For each gene, select the single best TSS by priority:
    1. Lowest appris_rank
    2. is_basic=1
    3. Lowest tsl
    4. Most 5' (smallest coord for + strand, largest for - strand)

    Returns {gene_name: {"chrom", "tss", "strand"}}
    """
    canonical = {}
    for gene, transcripts in gene_transcripts.items():
        # Sort by appris_rank, then is_basic (descending = prefer basic), then tsl, then position
        def sort_key(t):
            pos_tiebreak = t["tss"] if t["strand"] == "+" else -t["tss"]
            return (t["appris_rank"], -t["is_basic"], t["tsl"], pos_tiebreak)

        best = sorted(transcripts, key=sort_key)[0]
        canonical[gene] = {
            "chrom": best["chrom"],
            "tss": best["tss"],
            "strand": best["strand"],
            "transcript_id": best["transcript_id"],
        }
    return canonical


# ---------------------------------------------------------------------------
# hg38 genome: per-chromosome download and indexing
# ---------------------------------------------------------------------------

def download_chrom(chrom: str) -> Path:
    """Download and cache a single hg38 chromosome fasta.gz from UCSC."""
    dest = GENOME_DIR / f"{chrom}.fa.gz"
    if dest.exists():
        return dest
    url = UCSC_CHROM_URL.format(chrom=chrom)
    print(f"  Downloading {chrom} from UCSC (~150-250 MB)...")
    try:
        r = requests.get(url, stream=True, timeout=3600, headers=HEADERS)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(65536):
                fh.write(chunk)
        size_mb = dest.stat().st_size // (1024**2)
        print(f"  Saved {chrom}.fa.gz ({size_mb} MB)")
        return dest
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download {chrom}: {e}")


def load_chrom_sequence(chrom: str) -> str:
    """Load and return the full sequence string for a chromosome (uppercase)."""
    dest = download_chrom(chrom)
    print(f"  Loading {chrom} sequence into memory...")
    with gzip.open(dest, "rt") as fh:
        lines = []
        for line in fh:
            if line.startswith(">"):
                continue
            lines.append(line.strip().upper())
    seq = "".join(lines)
    print(f"  {chrom}: {len(seq):,} bp loaded")
    return seq


# Cache loaded chromosomes in memory within this process
_chrom_cache: dict[str, str] = {}


def get_chrom_seq(chrom: str) -> str:
    if chrom not in _chrom_cache:
        _chrom_cache[chrom] = load_chrom_sequence(chrom)
    return _chrom_cache[chrom]


RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]


def extract_window(chrom_seq: str, tss_0: int, half: int, strand: str) -> str:
    """
    Extract a window of size 2*half centered on tss_0 (0-based coordinate).
    N-pads at chromosome edges. Reverse-complements if strand == '-'.
    """
    chrom_len = len(chrom_seq)
    start = tss_0 - half
    end = tss_0 + half  # exclusive

    left_pad = max(0, -start)
    right_pad = max(0, end - chrom_len)

    start_clamp = max(0, start)
    end_clamp = min(chrom_len, end)

    seq = "N" * left_pad + chrom_seq[start_clamp:end_clamp] + "N" * right_pad
    assert len(seq) == 2 * half, f"Window length mismatch: {len(seq)} != {2 * half}"

    if strand == "-":
        seq = reverse_complement(seq)
    return seq


def extract_enformer_window(chrom_seq: str, tss_0: int, strand: str) -> str:
    """Extract ENFORMER_LEN bp centered on tss_0."""
    half = ENFORMER_LEN // 2
    return extract_window(chrom_seq, tss_0, half, strand)


def extract_evo2_window(chrom_seq: str, tss_0: int, strand: str) -> str:
    """Extract ±EVO2_WINDOW (4000 bp total) centered on tss_0."""
    return extract_window(chrom_seq, tss_0, EVO2_WINDOW, strand)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if OUT_CSV.exists():
        print(f"Output already exists: {OUT_CSV}. Delete to re-run.")
        df = pd.read_csv(OUT_CSV)
        print(f"Loaded {len(df)} TSS entries.")
        # Verify sequence lengths
        bad_evo2 = (df["evo2_seq"].str.len() != 4000).sum()
        bad_enf = (df["enformer_seq"].str.len() != ENFORMER_LEN).sum()
        if bad_evo2 or bad_enf:
            print(f"  WARNING: {bad_evo2} evo2_seq and {bad_enf} enformer_seq have wrong length!")
        return df

    if not PAIRS_CSV.exists():
        print(f"[ERROR] {PAIRS_CSV} not found. Run 01_download_data.py first.")
        sys.exit(1)

    pairs = pd.read_csv(PAIRS_CSV)
    target_genes = set(pairs["gene_A"]) | set(pairs["gene_B"])
    print(f"\nTarget genes to find TSS for: {len(target_genes)}")

    # --- Download and parse GENCODE GTF ---
    print("\n[Step 1] GENCODE v38 GTF...")
    download_gtf()
    gene_transcripts = parse_gtf_for_genes(target_genes)

    # --- Select canonical TSS per gene ---
    print("\n[Step 2] Selecting canonical TSS per gene (APPRIS principal isoform)...")
    canonical = select_canonical_tss(gene_transcripts)
    still_missing = target_genes - set(canonical.keys())
    if still_missing:
        print(f"  Genes with NO TSS found: {sorted(still_missing)}")

    # --- Determine which chromosomes to download ---
    chroms_needed = {v["chrom"] for v in canonical.values()}
    print(f"\n[Step 3] Chromosomes needed: {sorted(chroms_needed)}")

    # --- Extract sequences ---
    print("\n[Step 4] Extracting sequences from hg38...")
    records = []
    errors = []

    for gene, info in sorted(canonical.items()):
        chrom = info["chrom"]
        tss_1based = info["tss"]
        tss_0based = tss_1based - 1  # convert to 0-based
        strand = info["strand"]
        tid = info["transcript_id"]

        try:
            chrom_seq = get_chrom_seq(chrom)
        except Exception as e:
            print(f"  [WARN] Could not load {chrom}: {e}")
            errors.append(gene)
            continue

        evo2_seq = extract_evo2_window(chrom_seq, tss_0based, strand)
        enformer_seq = extract_enformer_window(chrom_seq, tss_0based, strand)

        assert len(evo2_seq) == 4000, f"{gene}: evo2_seq length {len(evo2_seq)}"
        assert len(enformer_seq) == ENFORMER_LEN, f"{gene}: enformer_seq length {len(enformer_seq)}"

        records.append({
            "gene": gene,
            "chrom": chrom,
            "tss_1based": tss_1based,
            "strand": strand,
            "transcript_id": tid,
            "evo2_seq": evo2_seq,
            "enformer_seq": enformer_seq,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} TSS entries to {OUT_CSV}")
    if errors:
        print(f"Failed for {len(errors)} genes: {errors}")

    # Verify
    not_in_tss = target_genes - set(df["gene"])
    if not_in_tss:
        print(f"Genes still missing from output ({len(not_in_tss)}): {sorted(not_in_tss)[:20]}")

    return df


if __name__ == "__main__":
    main()
