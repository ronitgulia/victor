"""
alerts.py — Victor Bot Detection: Webhook Alert Manager

Sends real-time Slack and/or Discord alerts when a bot is detected
with high confidence (≥ configured threshold).

Features:
  - Supports Slack Incoming Webhooks (Block Kit formatting)
  - Supports Discord Webhooks (rich Embed formatting)
  - Per-IP cooldown to prevent alert spam during sustained attacks
  - Global rate limiter (max N alerts per minute)
  - Non-blocking: all HTTP calls run in a background thread
  - Graceful degradation: silently skips if no URL is configured

Configuration:
  Set environment variables (never commit these to Git):
    SLACK_WEBHOOK_URL   = https://hooks.slack.com/services/...
    DISCORD_WEBHOOK_URL = https://discord.com/api/webhooks/...

  Tune behaviour in config.yaml under the `alerts:` section.

Self-test:
  python alerts.py
"""

import os
import json
import threading
import time
import logging
from datetime import datetime, timezone
from collections import deque

import requests

from config_loader import Config

Config()  # ensure singleton is loaded

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# ALERT MANAGER
# ──────────────────────────────────────────────────────────────────
class AlertManager:
    """
    Fires webhook alerts to Slack and/or Discord when a high-confidence
    bot is detected by the honeypot server.

    Thread-safety:
        All shared state (_cooldown_map, _alert_timestamps) is
        protected by _lock.  alert() and fire() may be called from
        concurrent Flask request threads without issue.
    """

    def __init__(self):
        self.slack_url   = os.environ.get("SLACK_WEBHOOK_URL",   "").strip()
        self.discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

        self.threshold      = float(Config.get("alerts.high_confidence_threshold", 0.90))
        self.cooldown_secs  = int(Config.get("alerts.cooldown_seconds",            300))
        self.max_per_minute = int(Config.get("alerts.max_alerts_per_minute",       10))

        self._lock              = threading.Lock()
        self._cooldown_map: dict = {}          # ip -> last_alert_epoch
        self._alert_timestamps  = deque()      # epoch times of recent alerts (for rate limit)

        enabled = []
        if self.slack_url:
            enabled.append("Slack")
        if self.discord_url:
            enabled.append("Discord")

        if enabled:
            logger.info(f"[AlertManager] Active channels: {', '.join(enabled)}")
            logger.info(f"[AlertManager] Threshold: {self.threshold} | "
                  f"Cooldown: {self.cooldown_secs}s | "
                  f"Rate limit: {self.max_per_minute}/min")
        else:
            logger.info("[AlertManager] No webhook URLs configured — alerts disabled.")
            logger.info("[AlertManager] Set SLACK_WEBHOOK_URL or DISCORD_WEBHOOK_URL "
                  "environment variables to enable.")

    # ── Public API ──────────────────────────────────────────────────

    def fire(self, record: dict, score: float) -> None:
        """
        Called by honeypot.py on every scored request.
        Fires an alert if score >= threshold AND the IP is not in cooldown
        AND the global rate limit has not been reached.

        This method returns immediately — the HTTP call runs in a
        background thread.
        """
        if not (self.slack_url or self.discord_url):
            return
        if score < self.threshold:
            return

        ip = record.get("ip", "unknown")

        with self._lock:
            now = time.time()

            # Per-IP cooldown check
            last = self._cooldown_map.get(ip, 0)
            if now - last < self.cooldown_secs:
                return  # already alerted for this IP recently

            # Global rate limit check
            cutoff = now - 60
            while self._alert_timestamps and self._alert_timestamps[0] < cutoff:
                self._alert_timestamps.popleft()
            if len(self._alert_timestamps) >= self.max_per_minute:
                logger.warning("[AlertManager] Global rate limit reached — alert suppressed")
                return

            # Reserve the slot
            self._cooldown_map[ip] = now
            self._alert_timestamps.append(now)

        # Dispatch in background so request latency is unaffected
        t = threading.Thread(
            target=self._dispatch,
            args=(record, score),
            daemon=True,
        )
        t.start()

    # ── Internal helpers ────────────────────────────────────────────

    def _dispatch(self, record: dict, score: float) -> None:
        """Send alert to all configured channels (runs in background thread)."""
        if self.slack_url:
            try:
                self._send_slack(record, score)
            except Exception as exc:
                logger.error(f"[AlertManager] Slack send failed: {exc}")

        if self.discord_url:
            try:
                self._send_discord(record, score)
            except Exception as exc:
                logger.error(f"[AlertManager] Discord send failed: {exc}")

    def _build_fields(self, record: dict, score: float) -> dict:
        """Extract display fields from the request record."""
        ts  = record.get("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            dt  = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts_display = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            ts_display = ts

        path    = record.get("path",        "/")
        ua      = record.get("user_agent",  "—") or "—"
        ip      = record.get("ip",          "unknown")
        is_dc   = record.get("is_datacenter_ip", 0)

        return {
            "ip":        ip,
            "score_pct": f"{score:.1%}",
            "score_raw": f"{score:.4f}",
            "path":      path,
            "ua":        ua[:120] + ("…" if len(ua) > 120 else ""),
            "datacenter": "Yes ⚠️" if is_dc else "No",
            "threshold": f"{self.threshold:.0%}",
            "timestamp": ts_display,
        }

    # ── Slack ───────────────────────────────────────────────────────

    def _send_slack(self, record: dict, score: float) -> None:
        """Send a formatted Block Kit message to Slack."""
        f = self._build_fields(record, score)

        payload = {
            "text": f"🚨 High-Confidence Bot Detected — {f['ip']} scored {f['score_pct']}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 High-Confidence Bot Detected",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*IP Address*\n`{f['ip']}`"},
                        {"type": "mrkdwn", "text": f"*Confidence Score*\n`{f['score_pct']}` (threshold: {f['threshold']})"},
                        {"type": "mrkdwn", "text": f"*Endpoint Hit*\n`{f['path']}`"},
                        {"type": "mrkdwn", "text": f"*Datacenter IP*\n{f['datacenter']}"},
                        {"type": "mrkdwn", "text": f"*User-Agent*\n`{f['ua']}`"},
                        {"type": "mrkdwn", "text": f"*Detected At*\n{f['timestamp']}"},
                    ],
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "⚔ *Victor Bot Detection* — real-time ensemble ML shield",
                        }
                    ],
                },
                {"type": "divider"},
            ],
        }

        resp = requests.post(
            self.slack_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        logger.info(f"[AlertManager] Slack alert sent for {f['ip']} (score={f['score_raw']})")

    # ── Discord ─────────────────────────────────────────────────────

    def _send_discord(self, record: dict, score: float) -> None:
        """Send a rich Embed message to Discord."""
        f = self._build_fields(record, score)

        payload = {
            "username": "Victor — Bot Detection",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2092/2092663.png",
            "embeds": [
                {
                    "title": "🚨 High-Confidence Bot Detected",
                    "color": 0xD63031,   # red
                    "fields": [
                        {"name": "IP Address",        "value": f"`{f['ip']}`",          "inline": True},
                        {"name": "Confidence Score",  "value": f"`{f['score_pct']}`",   "inline": True},
                        {"name": "Threshold",         "value": f"`{f['threshold']}`",   "inline": True},
                        {"name": "Endpoint",          "value": f"`{f['path']}`",        "inline": True},
                        {"name": "Datacenter IP",     "value": f['datacenter'],         "inline": True},
                        {"name": "Detected At",       "value": f['timestamp'],          "inline": False},
                        {"name": "User-Agent",        "value": f"`{f['ua']}`",          "inline": False},
                    ],
                    "footer": {
                        "text": "⚔ Victor Bot Detection — ensemble ML shield",
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

        resp = requests.post(
            self.discord_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        logger.info(f"[AlertManager] Discord alert sent for {f['ip']} (score={f['score_raw']})")


# ──────────────────────────────────────────────────────────────────
# SELF-TEST
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Victor — AlertManager Self-Test")
    logger.info("=" * 60)

    slack_url   = os.environ.get("SLACK_WEBHOOK_URL",   "")
    discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not slack_url and not discord_url:
        logger.warning("\n⚠  No webhook URLs found in environment.")
        logger.warning("   Set one or both before running the self-test:\n")
        logger.info("   Windows PowerShell:")
        logger.info('     $env:SLACK_WEBHOOK_URL   = "https://hooks.slack.com/services/..."')
        logger.info('     $env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."')
        logger.info("\n   Linux / macOS / Docker:")
        logger.info('     export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."')
        logger.info('     export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."\n')
    else:
        am = AlertManager()

        # Fake a high-confidence bot request record
        fake_record = {
            "ip":               "192.0.2.42",
            "user_agent":       "python-requests/2.28.0",
            "path":             "/secret-data",
            "referer":          "none",
            "accept_lang":      "none",
            "accept_encoding":  "none",
            "is_datacenter_ip": 1,
            "header_count":     3,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }

        logger.info("\nSending test alert with score = 0.97 …\n")
        am.fire(fake_record, score=0.97)

        # Give background thread time to complete
        time.sleep(3)
        logger.info("\nSelf-test complete. Check your Slack / Discord channel.")

    logger.info("=" * 60)
