"""
Telegram morning-brief alerts (Phase D2). Composes a FACTUAL, numeric daily
brief from the live system state (portfolio.system.run_full_system) and sends
it via the Telegram Bot API when credentials are present.

HONESTY / OPS: this is a monitoring tool, not a marketing one -- the brief
states the current regime, book exposure, circuit-breaker status and risk
metrics as numbers, and explicitly reminds the reader the system's edge is
drawdown control, not stock-picking (the measured truth). Actual SENDING
requires a bot token + chat id (config.TELEGRAM_BOT_TOKEN / _CHAT_ID or the
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars). Without them, send_brief() is
a no-op that returns the composed text -- it never invents a delivery it
didn't make, and it never trades or authorizes anything.
"""

import logging
import os

import pandas as pd

import config

logger = logging.getLogger("esis_quant.alerts.telegram")


def compose_daily_brief(result: dict) -> str:
    """Build the morning-brief text from a run_full_system result dict."""
    latest = result["latest"]
    m = result["metrics"]["full_system"]
    te = result["attribution"]["timing"]

    regime = latest["regime"]
    exposure = latest["exposure"]
    breaker = "ARMED (in cash)" if latest["breaker_active"] else "OK"
    posture = "FULLY INVESTED" if exposure >= 0.999 else ("IN CASH" if exposure <= 1e-6 else f"{exposure*100:.0f}% INVESTED")

    lines = [
        f"ESIS-QUANT Morning Brief - {pd.Timestamp(latest['date']).date()}",
        "-" * 30,
        f"REGIME: {regime}   |   POSTURE: {posture}",
        f"Book exposure: {exposure*100:.0f}%   Circuit breaker: {breaker}",
        f"Current drawdown: {latest['current_drawdown']*100:.1f}%",
        f"Regime param vintage: {pd.Timestamp(latest['param_vintage']).date() if latest.get('param_vintage') is not None else 'n/a'}",
        "-" * 30,
        "System (walk-forward, net of 0.55% costs):",
        f"  Sharpe {m['sharpe']:.2f} | Sortino {m['sortino']:.2f} | MaxDD {m['max_drawdown']*100:.0f}%",
        f"  Calmar {m['calmar']:.2f} | CF-VaR95 {m['cornish_fisher_var95']*100:.2f}% | AnnRet {m['ann_return']*100:.0f}%",
        f"  Regime-timing contribution to date: {te['timing_contrib']*100:+.0f}%",
        "-" * 30,
        "Note: edge is DRAWDOWN CONTROL (regime de-risking), not stock selection.",
        "Not investment advice.",
    ]
    return "\n".join(lines)


def _credentials():
    token = getattr(config, "TELEGRAM_BOT_TOKEN", None) or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = getattr(config, "TELEGRAM_CHAT_ID", None) or os.environ.get("TELEGRAM_CHAT_ID")
    return token, chat


def send_brief(result: dict = None, dry_run: bool = None) -> dict:
    """Compose and (if credentialed and not dry_run) send the brief. Returns
    {'text', 'sent', 'reason'}. Defaults to dry_run=True unless credentials
    exist -- never sends silently.
    """
    if result is None:
        from portfolio.system import run_full_system
        result = run_full_system()
    text = compose_daily_brief(result)
    token, chat = _credentials()
    if dry_run is None:
        dry_run = not (token and chat)
    if dry_run or not (token and chat):
        logger.info("send_brief: dry-run (no Telegram credentials configured); brief composed only")
        return {"text": text, "sent": False, "reason": "no_credentials_or_dry_run"}

    import requests
    resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         data={"chat_id": chat, "text": text}, timeout=10)
    ok = resp.status_code == 200
    logger.info("send_brief: telegram send status=%s", resp.status_code)
    return {"text": text, "sent": ok, "reason": f"http_{resp.status_code}"}
