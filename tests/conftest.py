"""
No test touches the real database (Stage 0, 2026-10-03).

Before this, any test that did not set up its own temp DB read -- and some
wrote -- esis_quant.db: the live book, the ledger, the regime signal. Four
pretrade tests failed whenever the real account was in drawdown, and a guard
test that regressed would have run its attack against real prices.

Now the whole session runs on a COPY of the database, taken once with SQLite's
backup API (consistent even while the daily job holds a WAL). Tests that build
their own temp DB still do; when they finish, config.DB_PATH falls back to the
copy, never to the original.
"""

import os
import sqlite3

import pytest

import config

_REAL_DB = config.DB_PATH


# The copy is ~365 MB. Pytest's temp root is on C, which has ~1 GB free and keeps
# the last three runs -- on 2026-10-04 that filled C completely. Keep the copy on
# the data drive and overwrite it each session.
_COPY_DIR = os.path.join(os.path.dirname(_REAL_DB) or ".", ".pytest_db")


@pytest.fixture(scope="session", autouse=True)
def _isolated_database(tmp_path_factory):
    os.makedirs(_COPY_DIR, exist_ok=True)
    copy = os.path.join(_COPY_DIR, "esis_quant_test_copy.db")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(copy + ext):
            os.remove(copy + ext)
    if os.path.exists(_REAL_DB):
        src = sqlite3.connect(f"file:{_REAL_DB}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        src.backup(dst)
        dst.close()
        src.close()
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DB_PATH", copy)
    yield copy
    mp.undo()


REAL_DB = _REAL_DB
