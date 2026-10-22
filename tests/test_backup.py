"""Backup: verified snapshot, retention, restore never overwrites."""

import os
import zipfile

import pandas as pd
import pytest

from ops import backup


def test_snapshot_is_verified():
    from tests.conftest import _COPY_DIR                      # ~365 MB: keep it off drive C
    dest = os.path.join(_COPY_DIR, "snap_test.db")
    try:
        r = backup.snapshot(dest)
    finally:
        if os.path.exists(dest):
            os.remove(dest)
    assert r["quick_check"] and r["verified"] and not r["mismatch"]
    assert "user_trade_ledger" in r["counts"]


def test_retention_keeps_last_week_and_sundays(tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "LOCAL_DIR", str(tmp_path))
    today = pd.Timestamp("2026-10-04")                          # a Sunday
    for k in range(40):
        (tmp_path / f"esis_quant_{(today - pd.Timedelta(days=k)).date()}.zip").write_bytes(b"x")
    backup._prune(today)
    left = sorted(f[11:21] for f in os.listdir(tmp_path))
    days = [pd.Timestamp(d) for d in left]
    assert sum((today - d).days < backup.KEEP_DAILY for d in days) == backup.KEEP_DAILY
    old = [d for d in days if (today - d).days >= backup.KEEP_DAILY]
    assert old and all(d.dayofweek == 6 for d in old)          # only Sundays survive
    assert all((today - d).days < 7 * backup.KEEP_WEEKLY for d in old)


def test_restore_refuses_to_overwrite(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "esis_quant.db"))
    z = tmp_path / "b.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("esis_quant.db", b"")
    (tmp_path / "esis_quant.restored.db").write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        backup.restore(str(z))
    assert (tmp_path / "esis_quant.restored.db").read_bytes() == b"keep me"
