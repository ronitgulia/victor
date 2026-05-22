# feature_engineering.py
"""
Feature Engineering — Victor Bot Detection
Reads raw traffic logs from SQLite, engineers 12 ML-ready features,
and writes data/features.csv.

Feature set (Step 5 original 8 + Step 6 new 4):
  Original: ua_is_suspicious, has_referer, has_accept_lang, hit_secret_page,
            ua_length, time_gap_seconds, unique_pages_visited, total_requests_from_ip
  New:      is_datacenter_ip, header_count, missing_common_headers, accept_encoding_score

Run with: python feature_engineering.py
"""

import pandas as pd
import numpy as np
import ipaddress
import json
import os

from database import TrafficDatabase

os.makedirs("data", exist_ok=True)

BOT_KEYWORDS = [
    "python", "scrapy", "curl", "go-http", "wget",
    "bot", "crawl", "spider"
]

FEATURE_COLS = [
    # Original 8
    "ua_is_suspicious", "has_referer", "has_accept_lang",
    "hit_secret_page", "ua_length", "time_gap_seconds",
    "unique_pages_visited", "total_requests_from_ip",
    # New 4 (Step 6)
    "is_datacenter_ip", "header_count",
    "missing_common_headers", "accept_encoding_score",
]


# ──────────────────────────────────────────────────────────────────
# DATACENTER HELPERS
# ──────────────────────────────────────────────────────────────────
def _load_datacenter_ranges(path: str = "data/datacenter_ranges.json") -> list:
    """Load CIDR networks from the ranges JSON file."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found — is_datacenter_ip will be 0 for all")
        return []
    with open(path) as f:
        data = json.load(f)
    networks = []
    for cidr in data.get("ranges", []):
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            pass
    print(f"  Loaded {len(networks)} datacenter CIDR ranges")
    return networks


def _is_datacenter(ip: str, networks: list) -> int:
    """Return 1 if ip belongs to a known datacenter range, else 0."""
    try:
        addr = ipaddress.ip_address(ip)
        return int(any(addr in net for net in networks))
    except ValueError:
        return 0


# ──────────────────────────────────────────────────────────────────
# ACCEPT-ENCODING SCORE HELPER
# ──────────────────────────────────────────────────────────────────
def _accept_encoding_score(enc: str) -> int:
    """
    0 = no encoding / plain text (very bot-like)
    1 = gzip only (common in simple HTTP clients)
    2 = gzip + brotli (real browsers)
    """
    enc = str(enc).lower()
    if "br" in enc or "brotli" in enc:
        return 2
    if "gzip" in enc:
        return 1
    return 0


# ──────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────
def engineer_features() -> pd.DataFrame:
    """Load raw SQLite logs and compute all 12 ML features."""

    print("Loading traffic logs from database...")
    db  = TrafficDatabase()
    df  = db.get_all_logs()
    db.close()

    if len(df) == 0:
        print("ERROR: No data found. Run honeypot.py + simulate_traffic.py first.")
        return pd.DataFrame()

    bots   = int(df["label"].sum())
    humans = int((df["label"] == 0).sum())
    print(f"  {len(df)} rows | {bots} bots | {humans} humans")

    # ── Original 8 features ───────────────────────────────────────
    df["ua_is_suspicious"] = df["user_agent"].str.lower().apply(
        lambda ua: int(any(kw in str(ua) for kw in BOT_KEYWORDS))
    )
    df["has_referer"]     = (df["referer"].str.strip().str.lower()     != "none").astype(int)
    df["has_accept_lang"] = (df["accept_lang"].str.strip().str.lower() != "none").astype(int)
    df["hit_secret_page"] = df["path"].str.contains("secret", na=False).astype(int)
    df["ua_length"]       = df["user_agent"].str.len().fillna(0).astype(int)

    # Time gap between consecutive requests per IP
    # ISSUE 9: We use fillna(0) instead of an arbitrary value like 5. 
    # The first request from any IP has no preceding request, so the time gap 
    # is mathematically undefined. Filling with 0 establishes a baseline without 
    # introducing artificial separation between bots and humans that a value 
    # like 5 seconds might create.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["ip", "timestamp"]).reset_index(drop=True)
    df["time_gap_seconds"] = (
        df.groupby("ip")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
          .clip(lower=0)
    )

    # IP-level aggregates merged back per row
    ip_agg = df.groupby("ip").agg(
        unique_pages_visited   = ("path", "nunique"),
        total_requests_from_ip = ("ip",   "count"),
    ).reset_index()
    df = df.merge(ip_agg, on="ip", how="left")

    # ── New 4 features (Step 6) ───────────────────────────────────
    print("Computing new features (Step 6)...")

    # 1. is_datacenter_ip
    dc_networks = _load_datacenter_ranges()
    df["is_datacenter_ip"] = df["ip"].apply(
        lambda ip: _is_datacenter(ip, dc_networks)
    )

    # 2. header_count (stored in DB; 0 for old rows without this column)
    if "header_count" in df.columns:
        df["header_count"] = df["header_count"].fillna(0).astype(int)
    else:
        df["header_count"] = 0

    # 3. missing_common_headers
    #    = (1 - has_referer) + (1 - has_accept_lang) + (1 - has_accept)
    #    Range 0–3; higher = more bot-like
    has_accept_col = (
        df["has_accept"].fillna(0).astype(int)
        if "has_accept" in df.columns
        else pd.Series(0, index=df.index)
    )
    df["missing_common_headers"] = (
        (1 - df["has_referer"]) +
        (1 - df["has_accept_lang"]) +
        (1 - has_accept_col)
    ).astype(int)

    # 4. accept_encoding_score (0 / 1 / 2)
    accept_enc_col = (
        df["accept_encoding"].fillna("none")
        if "accept_encoding" in df.columns
        else pd.Series("none", index=df.index)
    )
    df["accept_encoding_score"] = accept_enc_col.apply(_accept_encoding_score)

    # ── Select output columns ─────────────────────────────────────
    keep = ["ip", "timestamp"] + FEATURE_COLS + ["label"]
    out  = df[[c for c in keep if c in df.columns]].copy()

    out.to_csv("data/features.csv", index=False)
    print(f"\nSaved → data/features.csv")
    print(f"  Rows: {len(out)} | Features: {len(FEATURE_COLS)} | "
          f"Bots: {int(out['label'].sum())} | Humans: {int((out['label']==0).sum())}")
    return out


if __name__ == "__main__":
    print("=" * 58)
    print("  VICTOR — Feature Engineering  (12 features)")
    print("=" * 58)
    result = engineer_features()
    if len(result) > 0:
        print("\nNext step: python train_model.py")
    print("=" * 58)