# database.py
import sqlite3
import pandas as pd
import os
import re
from config_loader import Config

from config import Paths
from logger import get_logger

logger = get_logger(__name__)

DB_PATH = Paths.TRAFFIC_DB


class TrafficDatabase:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_table()
        self._migrate_schema()

    def _create_table(self):
        """Create the traffic_logs table with all current columns."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT,
                ip              TEXT,
                user_agent      TEXT,
                referer         TEXT,
                accept_lang     TEXT,
                path            TEXT,
                method          TEXT,
                status_code     INTEGER,
                session_id      TEXT,
                header_count    INTEGER DEFAULT 0,
                accept_encoding TEXT    DEFAULT 'none',
                has_accept      INTEGER DEFAULT 0,
                bot_score       REAL    DEFAULT -1.0,
                is_blocked      INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def _migrate_schema(self):
        """
        Safely add new columns to existing databases without data loss.
        ALTER TABLE ADD COLUMN is a no-op if the column already exists
        (we catch the exception and move on).
        """
        migrations = [
            ("header_count",    "INTEGER DEFAULT 0"),
            ("accept_encoding", "TEXT    DEFAULT 'none'"),
            ("has_accept",      "INTEGER DEFAULT 0"),
            ("bot_score",       "REAL    DEFAULT -1.0"),
            ("is_blocked",      "INTEGER DEFAULT 0"),
        ]
        for col, definition in migrations:
            try:
                self.conn.execute(
                    f"ALTER TABLE traffic_logs ADD COLUMN {col} {definition}"
                )
                self.conn.commit()
            except Exception:
                pass  # Column already exists — safe to ignore

    def log_request(self, timestamp, ip, user_agent, referer,
                    accept_lang, path, method, status_code, session_id,
                    header_count=0, accept_encoding="none", has_accept=0,
                    bot_score=-1.0, is_blocked=0):
        self.conn.execute("""
            INSERT INTO traffic_logs
              (timestamp, ip, user_agent, referer, accept_lang,
               path, method, status_code, session_id,
               header_count, accept_encoding, has_accept,
               bot_score, is_blocked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, ip, user_agent, referer, accept_lang,
              path, method, status_code, session_id,
              header_count, accept_encoding, has_accept,
              bot_score, is_blocked))
        self.conn.commit()

    def get_all_logs(self):
        """Returns all logs as a DataFrame with a label column."""
        df = pd.read_sql_query("SELECT * FROM traffic_logs", self.conn)
        if len(df) == 0:
            return df

        # Auto-label: bots = suspicious user agents OR hit secret page
        BOT_KEYWORDS = ["python", "scrapy", "curl", "go-http", "wget",
                        "bot", "crawl", "spider"]
        ua_is_bot = df["user_agent"].str.lower().apply(
            lambda ua: any(kw in str(ua) for kw in BOT_KEYWORDS)
        )
        hit_secret = df["path"].str.contains("secret", na=False)
        df["label"] = ((ua_is_bot) | (hit_secret)).astype(int)
        return df

    def get_record_count(self):
        cur = self.conn.execute("SELECT COUNT(*) FROM traffic_logs")
        return cur.fetchone()[0]

    def get_unique_ips(self):
        cur = self.conn.execute("SELECT COUNT(DISTINCT ip) FROM traffic_logs")
        return cur.fetchone()[0]

    def get_blocked_count(self):
        """Returns number of requests that were blocked in real-time."""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM traffic_logs WHERE is_blocked = 1"
        )
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()