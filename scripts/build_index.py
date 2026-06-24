#!/usr/bin/env python3
"""
epGram index builder
====================
Downloads parquet datasets from data.jmail.world, extracts word-year
frequency counts, and outputs a compact JSON index.

Output: ../data/epgram_index.json (~5-15 MB, to be committed)

Usage:
  pip install pandas pyarrow requests
  python scripts/build_index.py
"""

import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / ".cache"

DATASETS = {
    "emails-slim": "https://data.jmail.world/v1/emails-slim.parquet",
    "documents": "https://data.jmail.world/v1/documents.parquet",
}

# Words to ALWAYS index (common search terms for this dataset)
FORCE_WORDS = {
    "flight", "flights", "island", "palm", "beach", "settlement",
    "clinton", "trump", "andrew", "maxwell", "giuffre", "epstein",
    "jeffrey", "victim", "minor", "underage", "massage", "girl",
    "girls", "new", "york", "florida", "virgin", "islands",
    "plea", "agreement", "non-prosecution", "npa", "investigation",
    "fbi", "police", "arrest", "indictment", "attorney",
    "acosta", "dershowitz", "witness", "deposition", "testimony",
    "lawsuit", "settlement", "payment", "paid", "million",
    "billion", "financier", "bank", "account", "transfer",
    "private", "jet", "gulfstream", "pilot", "manifest",
    "prince", "royal", "british", "les", "wexner", "victoria",
    "secret", "recruit", "recruited", "traffic", "trafficking",
    "prosecution", "defense", "judge", "court", "sentence",
    "registry", "offender", "registered", "conviction",
    "miami", "herald", "journalist", "reporter", "article",
    "lawyer", "counsel", "client", "firm", "partner",
    "email", "phone", "call", "meeting", "visit", "travel",
    "hotel", "mansion", "house", "property", "estate",
    "evidence", "document", "sealed", "unsealed", "release",
    "transparency", "act", "efta", "foia", "disclosure",
    "doj", "department", "justice", "sdny", "southern", "district",
    "statement", "press", "media", "report", "reporting",
    "blackmail", "kompromat", "tape", "recorded", "video",
    "photograph", "photo", "picture", "image", "footage",
    "abuse", "assault", "rape", "molest", "trafficker",
    "teen", "teenage", "adolescent", "child", "children",
    "escort", "prostitute", "prostitution", "madam",
    "ghislaine", "maxwell", "roberts", "virginia", "jane", "doe",
    "caribbean", "st", "thomas", "little", "st", "james",
    "lolita", "express", "helicopter", "boat", "yacht",
    "science", "foundation", "donation", "charity", "philanthropy",
    "harvard", "mit", "conference", "dinner", "party", "event",
    "contact", "list", "black", "book", "address", "rolodex",
}

MIN_COUNT = 3  # minimum occurrences for a word to be included


def download_file(url, dest):
    """Download with progress. Skip if cached."""
    import requests
    if dest.exists():
        print(f"  Using cached: {dest.name}")
        return
    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1_048_576):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"    {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB", end="\r")
    print()


def tokenize(text):
    """Split text into lowercase word tokens. Returns set of unique tokens."""
    if not isinstance(text, str) or not text.strip():
        return set()
    # Normalize
    text = text.lower()
    # Extract words (2+ chars)
    words = re.findall(r'[a-z][a-z0-9_]{1,}', text)
    # Also capture 2-word bigrams for common phrases
    bigrams = set()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")
    return set(words) | bigrams


def build_index():
    """Main pipeline."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("epGram index builder\n" + "=" * 50)
    
    # Track word counts per year
    word_year_counts = defaultdict(lambda: defaultdict(int))  # word -> {year: count}
    total_per_year = defaultdict(int)  # year -> total emails
    all_years = set()
    
    # ── Process emails ──
    emails_path = CACHE_DIR / "emails-slim.parquet"
    download_file(DATASETS["emails-slim"], emails_path)
    
    print("\nProcessing emails...")
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(emails_path)
        df = table.to_pandas()
    except ImportError:
        import pandas as pd
        df = pd.read_parquet(emails_path)
    
    print(f"  {len(df):,} emails loaded")
    
    for i, row in df.iterrows():
        if i % 100000 == 0:
            print(f"  Processing email {i:,} / {len(df):,} ...", end="\r")
        
        # Get year
        year = None
        sent_at = row.get("sent_at")
        if sent_at is not None:
            try:
                year = int(pd.Timestamp(sent_at).year)
            except Exception:
                pass
        
        if year is None or year < 1980 or year > 2026:
            year = None  # skip undated
        
        # Get text
        subject = str(row.get("subject", "") or "")
        sender = str(row.get("sender", "") or "")
        text = f"{subject} {sender}"
        
        tokens = tokenize(text)
        
        if year is not None:
            total_per_year[year] += 1
            all_years.add(year)
            for token in tokens:
                word_year_counts[token][year] += 1
    
    print(f"\n  {len(df):,} emails processed. {len(word_year_counts):,} unique terms found.")
    
    # ── Process documents ──
    docs_path = CACHE_DIR / "documents.parquet"
    download_file(DATASETS["documents"], docs_path)
    
    print("\nProcessing documents...")
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(docs_path)
        df = table.to_pandas()
    except ImportError:
        import pandas as pd
        df = pd.read_parquet(docs_path)
    
    print(f"  {len(df):,} documents loaded")
    
    for i, row in df.iterrows():
        if i % 100000 == 0:
            print(f"  Processing doc {i:,} / {len(df):,} ...", end="\r")
        
        # Try to extract year from filename or description
        year = None
        filename = str(row.get("original_filename", "") or "")
        desc = str(row.get("document_description", "") or "")
        
        for field in [filename, desc]:
            m = re.search(r'\b((?:19|20)\d{2})\b', str(field))
            if m:
                y = int(m.group(1))
                if 1980 <= y <= 2026:
                    year = y
                    break
        
        text = f"{filename} {desc}"
        tokens = tokenize(text)
        
        if year is not None:
            total_per_year[year] += 1
            all_years.add(year)
            for token in tokens:
                word_year_counts[token][year] += 1
            # Count toward "undated" for docs without year in filename
        # We skip adding to word counts for undated docs since we can't place them
    
    print(f"\n  {len(df):,} documents processed.")
    
    # ── Filter & build final index ──
    print("\nBuilding index...")
    
    # Include all FORCE_WORDS, plus any word that appears >= MIN_COUNT times
    included = set(FORCE_WORDS)
    for word, year_counts in word_year_counts.items():
        total = sum(year_counts.values())
        if total >= MIN_COUNT:
            included.add(word)
    
    print(f"  Indexing {len(included):,} terms (out of {len(word_year_counts):,} total)")
    
    # Build compact output
    word_index = {}
    for word in sorted(included):
        year_counts = word_year_counts.get(word, {})
        if not year_counts:
            continue
        # Convert to sorted list of [year, count] pairs for compactness
        entries = [[int(y), int(c)] for y, c in sorted(year_counts.items())]
        word_index[word] = entries
    
    # ── Also build reverse index: top words per year ──
    top_per_year = {}
    for year in sorted(all_years):
        year_words = []
        for word in word_index:
            for entry in word_index[word]:
                if entry[0] == year:
                    year_words.append((word, entry[1]))
                    break
        year_words.sort(key=lambda x: -x[1])
        top_per_year[str(year)] = year_words[:500]  # top 500 per year
    
    years_list = sorted(all_years)
    
    output = {
        "meta": {
            "source": "data.jmail.world",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "years": [int(y) for y in years_list],
            "year_range": [years_list[0], years_list[-1]] if years_list else [1980, 2026],
            "total_terms_indexed": len(word_index),
            "total_per_year": {str(y): total_per_year.get(y, 0) for y in years_list},
        },
        "word_index": word_index,
    }
    
    # ── Write output ──
    # Uncompressed for debugging
    json_path = DATA_DIR / "epgram_index.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))
    
    uncompressed_mb = json_path.stat().st_size / 1024 / 1024
    
    # Compressed for production
    gz_path = DATA_DIR / "epgram_index.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(output, f, separators=(",", ":"))
    
    compressed_mb = gz_path.stat().st_size / 1024 / 1024
    
    print(f"\n{'=' * 50}")
    print(f"Index built successfully!")
    print(f"  {DATA_DIR / 'epgram_index.json'}: {uncompressed_mb:.1f} MB")
    print(f"  {DATA_DIR / 'epgram_index.json.gz'}: {compressed_mb:.1f} MB (gzipped)")
    print(f"\nTo use: commit {DATA_DIR / 'epgram_index.json.gz'} and deploy.")


if __name__ == "__main__":
    build_index()
