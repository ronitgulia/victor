"""
Victor Honeypot Server  —  with Real-Time Bot Detection

Every incoming request is:
  1. Logged to SQLite
  2. Scored by the trained XGBoost model in real-time
  3. Blocked (HTTP 403) if the ensemble score exceeds the configured threshold
  4. Alerted via Slack / Discord if the score >= the high-confidence threshold

Run with: python honeypot.py
"""

from flask import Flask, request, jsonify, abort
from datetime import datetime
import uuid
import os
import re
from config import Paths
import json
import atexit
import threading
import ipaddress
from collections import deque

from database import TrafficDatabase
from config_loader import Config
from alerts import AlertManager

Config()  # load singleton

app = Flask(__name__)
db  = TrafficDatabase()
atexit.register(db.close)

import hashlib
from logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# DATACENTER IP CHECKER
# ──────────────────────────────────────────────────────────────────
class DatacenterChecker:
    """
    Checks whether an IP belongs to a known cloud / datacenter provider.
    Uses Python's built-in ipaddress module — no external API calls.
    """

    def __init__(self, ranges_path: str = Paths.DATACENTER_RANGES):
        self.networks: list = []
        self._load(ranges_path)

    def _load(self, path: str):
        if not os.path.exists(path):
            logger.info(f"[DatacenterChecker] CIDR file not found: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        for cidr in data.get("ranges", []):
            try:
                self.networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                pass
        logger.info(f"[DatacenterChecker] {len(self.networks)} CIDR ranges loaded")

    def is_datacenter(self, ip: str) -> int:
        try:
            addr = ipaddress.ip_address(ip)
            return int(any(addr in net for net in self.networks))
        except ValueError:
            return 0


# ──────────────────────────────────────────────────────────────────
# REAL-TIME SCORER
# ──────────────────────────────────────────────────────────────────
class RealTimeScorer:
    """
    Loads the trained XGBoost model and scores every incoming request.

    Maintains a per-IP session history (thread-safe deque) so that
    aggregate features like time_gap_seconds and unique_pages_visited
    can be computed on-the-fly without hitting the database.

    Falls back gracefully if no model is found — all requests pass through.
    """



    def __init__(self):
        self.model        = None
        self.feature_cols = []
        self.threshold    = Config.get("detection.default_threshold", 0.5)
        self.blocking     = Config.get("detection.realtime_blocking",  True)

        self._sessions: dict  = {}      # ip -> deque[dict]
        self._lock            = threading.Lock()
        self._scored          = 0
        self._blocked         = 0
        self._started         = datetime.utcnow().isoformat()

        self._load_model()

    def _load_model(self):
        try:
            import joblib

            # Feature list is saved alongside the model so they stay in sync
            feat_path  = Config.get("paths.feature_cols",   Paths.FEATURE_COLS)
            model_path = Config.get("paths.xgboost_model",  Paths.XGB_MODEL)

            if not os.path.exists(model_path):
                logger.info("[RealTimeScorer] No model found — run train_model.py first")
                logger.info("[RealTimeScorer] Real-time scoring disabled until model is available")
                return

            self.model = joblib.load(model_path)

            if os.path.exists(feat_path):
                with open(feat_path) as f:
                    self.feature_cols = json.load(f)
            else:
                # Fallback to config if feature_cols.json doesn't exist yet
                self.feature_cols = Config.get("features.columns", [])

            # Load tuned threshold from ensemble_weights.json
            weight_path = Config.get("paths.ensemble_weights", Paths.ENSEMBLE_WEIGHTS)
            if os.path.exists(weight_path):
                try:
                    with open(weight_path) as f:
                        weights = json.load(f)
                        if "best_threshold" in weights:
                            self.threshold = weights["best_threshold"]
                except Exception:
                    pass

            logger.info(f"[RealTimeScorer] Model loaded  ({len(self.feature_cols)} features)")
            logger.info(f"[RealTimeScorer] Threshold: {self.threshold}  |  "
                  f"Blocking: {'ON' if self.blocking else 'OFF (log-only)'}")

        except Exception as e:
            logger.info(f"[RealTimeScorer] Load error: {e}")

    def _push(self, ip: str, record: dict) -> list:
        """Append record to IP session history and return a snapshot."""
        with self._lock:
            if ip not in self._sessions:
                self._sessions[ip] = deque(maxlen=500)
            self._sessions[ip].append(record)
            return list(self._sessions[ip])

    def _features(self, record: dict, history: list) -> dict:
        """Compute all ML features from the current request + IP history."""
        ua = record.get("user_agent", "")

        ua_is_suspicious = int(bool(self.bot_regex.search(ua)))
        has_referer      = int(record.get("referer",     "none").lower() not in ("none", "", "-"))
        has_accept_lang  = int(record.get("accept_lang", "none").lower() not in ("none", "", "-"))
        hit_secret_page  = int("secret" in record.get("path", ""))
        ua_length        = len(ua)

        # Time gap between this and the previous request from the same IP
        if len(history) >= 2:
            try:
                t1 = datetime.fromisoformat(history[-2]["timestamp"])
                t2 = datetime.fromisoformat(history[-1]["timestamp"])
                time_gap = max(0.0, (t2 - t1).total_seconds())
            except Exception:
                time_gap = 0.0
        else:
            time_gap = 0.0

        unique_pages   = len(set(r.get("path", "") for r in history))
        total_requests = len(history)

        # New features (Step 6)
        is_datacenter_ip = record.get("is_datacenter_ip", 0)
        header_count     = record.get("header_count", 0)
        has_accept       = record.get("has_accept", 0)
        missing_common   = (1 - has_referer) + (1 - has_accept_lang) + (1 - has_accept)

        enc = record.get("accept_encoding", "").lower()
        if "br" in enc or "brotli" in enc:
            accept_encoding_score = 2
        elif "gzip" in enc:
            accept_encoding_score = 1
        else:
            accept_encoding_score = 0

        # Burst count (rolling 10s)
        try:
            t_now = datetime.fromisoformat(record["timestamp"])
            burst_count_10s = sum(1 for r in history if (t_now - datetime.fromisoformat(r["timestamp"])).total_seconds() <= 10)
        except Exception:
            burst_count_10s = 1

        return {
            "ua_is_suspicious":       ua_is_suspicious,
            "has_referer":            has_referer,
            "has_accept_lang":        has_accept_lang,
            "hit_secret_page":        hit_secret_page,
            "ua_length":              ua_length,
            "time_gap_seconds":       time_gap,
            "unique_pages_visited":   unique_pages,
            "total_requests_from_ip": total_requests,
            "is_datacenter_ip":       is_datacenter_ip,
            "header_count":           header_count,
            "missing_common_headers": missing_common,
            "accept_encoding_score":  accept_encoding_score,
            "burst_count_10s":        burst_count_10s,
        }

    def score(self, record: dict) -> tuple:
        """
        Score a single request.
        Returns (score: float, is_bot: bool).
        score = -1.0 if model is not loaded.
        """
        history = self._push(record["ip"], record)

        if self.model is None:
            return -1.0, False

        try:
            import numpy as np
            feat_map = self._features(record, history)
            X        = [[feat_map.get(c, 0) for c in self.feature_cols]]
            score    = float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            logger.info(f"[RealTimeScorer] Scoring error: {e}")
            return -1.0, False

        is_bot = score >= self.threshold

        with self._lock:
            self._scored += 1
            if is_bot:
                self._blocked += 1
            # Write stats to disk every 50 requests for dashboard
            if self._scored % 50 == 0:
                self._flush_stats()

        return score, is_bot

    def _flush_stats(self):
        stats = {
            "model_loaded":    self.model is not None,
            "scored":          self._scored,
            "blocked":         self._blocked,
            "threshold":       self.threshold,
            "blocking_mode":   self.blocking,
            "session_started": self._started,
            "last_updated":    datetime.utcnow().isoformat(),
        }
        try:
            
            with open(Paths.REALTIME_STATS, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {
            "model_loaded":    self.model is not None,
            "scored":          self._scored,
            "blocked":         self._blocked,
            "threshold":       self.threshold,
            "blocking_mode":   self.blocking,
            "session_started": self._started,
        }


# ──────────────────────────────────────────────────────────────────
# GLOBAL INSTANCES
# ──────────────────────────────────────────────────────────────────
dc_checker    = DatacenterChecker()
scorer        = RealTimeScorer()
alert_manager = AlertManager()


# ──────────────────────────────────────────────────────────────────
# REQUEST HOOKS
# ──────────────────────────────────────────────────────────────────
@app.before_request
def log_and_score_request():
    """Capture request metadata, score it, and optionally block it."""
    timestamp = datetime.utcnow().isoformat()
    ip        = request.remote_addr

    user_agent      = request.headers.get("User-Agent",       "")
    referer         = request.headers.get("Referer",          "none")
    accept_lang     = request.headers.get("Accept-Language",  "none")
    accept_encoding = request.headers.get("Accept-Encoding",  "none")
    has_accept      = 1 if request.headers.get("Accept") else 0
    header_count    = len(list(request.headers))

    is_dc = dc_checker.is_datacenter(ip)

    record = {
        "timestamp":        timestamp,
        "ip":               ip,
        "user_agent":       user_agent,
        "referer":          referer,
        "accept_lang":      accept_lang,
        "accept_encoding":  accept_encoding,
        "has_accept":       has_accept,
        "header_count":     header_count,
        "path":             request.path,
        "method":           request.method,
        "session_id":       hashlib.md5(f"{ip}-{user_agent}".encode()).hexdigest()[:16],
        "is_datacenter_ip": is_dc,
    }

    # Real-time scoring
    bot_score, is_bot = scorer.score(record)
    record["bot_score"]  = round(bot_score, 4)
    record["is_blocked"] = 0

    # Webhook alert — fires in background thread, zero latency impact
    alert_manager.fire(record, bot_score)

    # Always persist log_data so after_request can save it
    request.log_data = record

    # Block if bot and blocking mode is on
    if is_bot and scorer.blocking:
        record["is_blocked"] = 1
        abort(403)


@app.after_request
def save_request(response):
    """Save the request (including score and blocked flag) to SQLite."""
    log_data = getattr(request, "log_data", {})
    if log_data:
        try:
            db.log_request(
                timestamp       = log_data["timestamp"],
                ip              = log_data["ip"],
                user_agent      = log_data["user_agent"],
                referer         = log_data["referer"],
                accept_lang     = log_data["accept_lang"],
                path            = log_data["path"],
                method          = log_data["method"],
                status_code     = response.status_code,
                session_id      = log_data["session_id"],
                header_count    = log_data.get("header_count",    0),
                accept_encoding = log_data.get("accept_encoding", "none"),
                has_accept      = log_data.get("has_accept",      0),
                bot_score       = log_data.get("bot_score",      -1.0),
                is_blocked      = log_data.get("is_blocked",      0),
            )
        except Exception as e:
            logger.error(f"[DB Error] Failed to log request: {e}")
    return response


# ──────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    """Homepage"""
    return jsonify({
        "status":    "ok",
        "message":   "Victor Honeypot Server",
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


@app.route("/articles", methods=["GET"])
def articles():
    """Articles page"""
    return jsonify({
        "status":   "ok",
        "page":     "articles",
        "articles": [
            {"id": 1, "title": "Article 1"},
            {"id": 2, "title": "Article 2"},
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


@app.route("/about", methods=["GET"])
def about():
    """About page"""
    return jsonify({
        "status":    "ok",
        "page":      "about",
        "content":   "This is the about page",
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


@app.route("/secret-data", methods=["GET"])
def secret():
    """
    Honeypot endpoint — legitimate users won't know about this.
    Bots will find it through aggressive scanning.
    """
    return jsonify({
        "status":    "error",
        "message":   "Forbidden",
        "page":      "secret-data",
        "warning":   "This endpoint should not be public",
        "timestamp": datetime.utcnow().isoformat(),
    }), 403


@app.route("/api/status", methods=["GET"])
def api_status():
    """Server status and basic traffic statistics."""
    return jsonify({
        "status":          "running",
        "total_requests":  db.get_record_count(),
        "unique_ips":      db.get_unique_ips(),
        "blocked_total":   db.get_blocked_count(),
        "timestamp":       datetime.utcnow().isoformat(),
    }), 200


@app.route("/api/realtime-stats", methods=["GET"])
def realtime_stats():
    """Real-time scoring statistics from the in-memory scorer."""
    scorer._flush_stats()   # ensure file is up-to-date
    return jsonify(scorer.stats), 200


# ──────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ──────────────────────────────────────────────────────────────────
@app.errorhandler(403)
def bot_blocked(error):
    return jsonify({
        "status":  "blocked",
        "message": "Access denied: bot traffic detected",
        "code":    403,
    }), 403


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status":  "error",
        "message": "Not found",
        "path":    request.path,
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "status":  "error",
        "message": "Internal server error",
    }), 500


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 62)
    logger.info("  VICTOR HONEYPOT SERVER  —  Real-Time Bot Detection")
    logger.info("=" * 62)
    logger.info(f"  URL:              http://127.0.0.1:5000")
    logger.info(f"  Model loaded:     {'Yes ✓' if scorer.model is not None else 'No — run train_model.py'}")
    logger.info(f"  Blocking mode:    {'ON  — bots get 403' if scorer.blocking else 'OFF — log only'}")
    logger.info(f"  Threshold:        {scorer.threshold}")
    logger.info(f"  Datacenter CIDRs: {len(dc_checker.networks)}")

    # Webhook alert status
    _slack   = "✓ Active" if alert_manager.slack_url   else "✗ Not configured"
    _discord = "✓ Active" if alert_manager.discord_url else "✗ Not configured"
    logger.info(f"  Alert threshold:  ≥ {alert_manager.threshold:.0%} confidence")
    logger.info(f"  Slack alerts:     {_slack}")
    logger.info(f"  Discord alerts:   {_discord}")

    logger.info()
    logger.info("  Endpoints:")
    logger.info("    GET /                    Homepage")
    logger.info("    GET /articles            Articles")
    logger.info("    GET /about               About")
    logger.info("    GET /secret-data         Honeypot trap")
    logger.info("    GET /api/status          Server status")
    logger.info("    GET /api/realtime-stats  Scoring stats")
    logger.info()
    logger.info("  Traffic logged to: data/victor_traffic.db")
    logger.info("=" * 62)

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=False, host=host, port=port, threaded=True)
