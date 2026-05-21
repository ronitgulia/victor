# feature_engineering.py
"""
Feature Engineering — Victor Bot Detection
Reads raw traffic logs from SQLite, computes per-request and
IP-level ML features, and writes data/features.csv.

Run with: python feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
from database import TrafficDatabase

os.makedirs("data", exist_ok=True)

BOT_KEYWORDS = [
    "python", "scrapy", "curl", "go-http", "wget",
    "bot", "crawl", "spider"
]

FEATURE_COLS = [
    "ua_is_suspicious", "has_referer", "has_accept_lang",
    "hit_secret_page", "ua_length", "time_gap_seconds",
    "unique_pages_visited", "total_requests_from_ip"
]


def engineer_features() -> pd.DataFrame:
    """Load raw SQLite logs, compute features, return feature DataFrame."""

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

    # ── per-request features ──────────────────────────────────────────
    df["ua_is_suspicious"] = df["user_agent"].str.lower().apply(
        lambda ua: int(any(kw in str(ua) for kw in BOT_KEYWORDS))
    )
    df["has_referer"]     = (df["referer"].str.strip().str.lower()     != "none").astype(int)
    df["has_accept_lang"] = (df["accept_lang"].str.strip().str.lower() != "none").astype(int)
    df["hit_secret_page"] = df["path"].str.contains("secret", na=False).astype(int)
    df["ua_length"]       = df["user_agent"].str.len().fillna(0).astype(int)

    # ── time gap between consecutive requests per IP ──────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["ip", "timestamp"]).reset_index(drop=True)
    df["time_gap_seconds"] = (
        df.groupby("ip")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
          .clip(lower=0)
    )

    # ── IP-level aggregates merged back per row ───────────────────────
    ip_agg = df.groupby("ip").agg(
        unique_pages_visited   = ("path", "nunique"),
        total_requests_from_ip = ("ip",   "count"),
    ).reset_index()
    df = df.merge(ip_agg, on="ip", how="left")

    # ── select output columns ─────────────────────────────────────────
    keep = ["ip", "timestamp"] + FEATURE_COLS + ["label"]
    out  = df[[c for c in keep if c in df.columns]].copy()

    out.to_csv("data/features.csv", index=False)
    print(f"Saved → data/features.csv  ({len(out)} rows)")
    return out


if __name__ == "__main__":
    print("=" * 55)
    print("  VICTOR — Feature Engineering")
    print("=" * 55)
    result = engineer_features()
    if len(result) > 0:
        print("\nNext step: python train_model.py")
    print("=" * 55)