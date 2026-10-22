"""
Nightly database backup, verified, on two physical disks and in the cloud.

WHY. Everything that cannot be re-downloaded -- your trade ledger, invested
capital, cash flows, decisions, the forward paper trades, the label work, the
agent findings -- lives in ONE SQLite file (esis_quant.db, ~365 MB) on drive D.
Prices can be fetched again; that history cannot. One failed disk or one bad
write and it is gone.

WHERE.
  * LOCAL, drive D (D:\\Stock\\backups): one zip a day (~75 MB, deflate), the last
    KEEP_DAILY days plus one per week for KEEP_WEEKLY weeks. Protects against a
    bad write, a bad migration, a mistaken delete: roll back to yesterday.
  * OFFSITE, OneDrive on drive C (a DIFFERENT physical disk, and synced to the
    cloud): ONE copy, LZMA (~45 MB), replaced nightly. Protects against drive D
    failing, theft, or the laptop dying. C has ~1 GB free, so only one copy is
    kept there and the offsite step is SKIPPED (with a finding) when C has less
    than OFFSITE_MIN_FREE_MB free.

HOW.
  1. sqlite3 backup API -> a consistent snapshot even while a job holds the WAL.
  2. VERIFY the snapshot before trusting it: PRAGMA quick_check, and row counts of
     the irreplaceable tables equal to the live database's.
  3. Zip; replace the offsite copy atomically (write temp, then rename).
  4. Log to backup_log; the sentinel raises a finding if the last good backup is
     more than 2 days old.

RESTORE (by hand, never automatic):
    py -3.11 -m ops.backup list
    py -3.11 -m ops.backup restore D:\\Stock\\backups\\esis_quant_2026-10-04.zip
  restore unzips next to the live file as esis_quant.restored.db; you swap it in
  yourself after closing every job, so nothing is overwritten by accident.
"""

import logging
import os
import shutil
import sqlite3
import sys
import time
import zipfile

import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.backup")

LOCAL_DIR = r"D:\Stock\backups"
OFFSITE_DIR = os.path.join(os.environ.get("OneDrive", r"C:\Users\marc amgad\OneDrive"), "ESIS-QUANT backup")
KEEP_DAILY = 7
KEEP_WEEKLY = 4
OFFSITE_MIN_FREE_MB = 300
# Irreplaceable: a backup is only good if these match the live database.
CRITICAL_TABLES = ["user_trade_ledger", "account_capital", "cash_flows", "user_daily_snapshot",
                   "decision_log", "trade_reviews", "index_paper_trades", "index_paper_reviews",
                   "label_queue", "agent_findings", "system_state"]


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS backup_log (
        finished_at TEXT, local_path TEXT, local_mb REAL, offsite_path TEXT, offsite_mb REAL,
        verified INTEGER, seconds REAL, note TEXT)""")


def _counts(conn) -> dict:
    out = {}
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in CRITICAL_TABLES:
        if t in have:
            out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    return out


def snapshot(dest: str) -> dict:
    """Consistent copy via the backup API, then verified against the live file."""
    src = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
        live = _counts(src)
    finally:
        src.close()
    try:
        ok = dst.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        copy = _counts(dst)
    finally:
        dst.close()
    # the live side may have grown during the copy (a job appending); the copy may
    # be BEHIND by a few rows but never ahead and never empty where the live is not
    short = {t: (copy.get(t), n) for t, n in live.items() if copy.get(t) is None or (n and not copy.get(t))
             or copy.get(t) > n}
    return {"quick_check": ok, "counts": copy, "mismatch": short, "verified": ok and not short}


def _zip(path: str, out: str, method) -> float:
    tmp = out + ".part"
    with zipfile.ZipFile(tmp, "w", method) as z:
        z.write(path, "esis_quant.db")
    os.replace(tmp, out)
    return os.path.getsize(out) / 1e6


def _prune(today: pd.Timestamp) -> list:
    removed = []
    files = sorted(f for f in os.listdir(LOCAL_DIR) if f.startswith("esis_quant_") and f.endswith(".zip"))
    for f in files:
        try:
            d = pd.Timestamp(f[len("esis_quant_"):-4])
        except ValueError:
            continue
        age = (today - d).days
        weekly_keep = d.dayofweek == 6 and age < 7 * KEEP_WEEKLY      # Sundays
        if age >= KEEP_DAILY and not weekly_keep:
            os.remove(os.path.join(LOCAL_DIR, f))
            removed.append(f)
    return removed


def run(offsite: bool = True) -> dict:
    from ops.keep_awake import keep_awake
    t0 = time.time()
    today = pd.Timestamp.now().normalize()
    os.makedirs(LOCAL_DIR, exist_ok=True)
    raw = os.path.join(LOCAL_DIR, "_snapshot.db")
    note = []
    with keep_awake():
        snap = snapshot(raw)
        if not snap["verified"]:
            os.remove(raw)
            raise RuntimeError(f"backup snapshot failed verification: {snap}")
        local = os.path.join(LOCAL_DIR, f"esis_quant_{today.date()}.zip")
        local_mb = _zip(raw, local, zipfile.ZIP_DEFLATED)
        off_path, off_mb = None, None
        if offsite:
            free_mb = shutil.disk_usage(os.path.splitdrive(OFFSITE_DIR)[0] + "\\").free / 1e6
            if free_mb < OFFSITE_MIN_FREE_MB:
                note.append(f"offsite SKIPPED: C has {free_mb:.0f} MB free (< {OFFSITE_MIN_FREE_MB})")
            else:
                os.makedirs(OFFSITE_DIR, exist_ok=True)
                off_path = os.path.join(OFFSITE_DIR, "esis_quant_latest.zip")
                lz = os.path.join(LOCAL_DIR, "_offsite.zip")
                _zip(raw, lz, zipfile.ZIP_LZMA)
                shutil.copyfile(lz, off_path + ".part")
                os.replace(off_path + ".part", off_path)
                os.remove(lz)
                off_mb = os.path.getsize(off_path) / 1e6
        os.remove(raw)
        removed = _prune(today)
    secs = time.time() - t0
    with get_connection() as conn:
        _ensure(conn)
        conn.execute("INSERT INTO backup_log VALUES (?,?,?,?,?,?,?,?)",
                     (pd.Timestamp.now().isoformat(timespec="seconds"), local, local_mb, off_path, off_mb,
                      1, round(secs, 1), "; ".join(note) or None))
        # always recorded, so a finding clears itself on the first good night
        from ops.agent_guard import ensure_tables, record_findings
        ensure_tables(conn)
        record_findings(conn, "backup", [{"severity": "high", "kind": "offsite_backup",
                                          "subject": "offsite backup", "summary": n,
                                          "fix_hint": "free space on C (or point OFFSITE_DIR at "
                                                      "another drive / USB stick)"} for n in note])
    return {"local": local, "local_mb": round(local_mb, 1), "offsite": off_path,
            "offsite_mb": None if off_mb is None else round(off_mb, 1), "seconds": round(secs, 1),
            "pruned": removed, "counts": snap["counts"], "note": note}


def list_backups() -> pd.DataFrame:
    rows = []
    for d in (LOCAL_DIR, OFFSITE_DIR):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".zip"):
                    p = os.path.join(d, f)
                    rows.append({"file": p, "mb": round(os.path.getsize(p) / 1e6, 1),
                                 "modified": pd.Timestamp(os.path.getmtime(p), unit="s").floor("s")})
    return pd.DataFrame(rows)


def restore(zip_path: str) -> str:
    """Unzip next to the live database as esis_quant.restored.db. Never overwrites."""
    out = os.path.join(os.path.dirname(config.DB_PATH), "esis_quant.restored.db")
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists; move it away first")
    with zipfile.ZipFile(zip_path) as z, z.open("esis_quant.db") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    con = sqlite3.connect(out)
    try:
        ok = con.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        con.close()
    return f"restored to {out} (quick_check: {ok}). Close every job, then swap it for esis_quant.db yourself."


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    if argv[:1] == ["list"]:
        print(list_backups().to_string(index=False))
    elif argv[:1] == ["restore"] and len(argv) > 1:
        print(restore(argv[1]))
    else:
        print(run(offsite="--local-only" not in argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
