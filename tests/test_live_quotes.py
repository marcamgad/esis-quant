"""Delayed-quote parser and session clock. Offline: the page text is a fixture."""

import pandas as pd

from data import live_quotes as lq

PAGE = ('...,quote:{c:1.39,e:false,h:128.15,l:126.25,o:126.3,p:127.69,'
        'u:"At close: Oct 1, 2026",v:3413347,cl:126.3,cp:1.1005,ex:"Egyptian Stock Exchange",'
        'ms:"closed",pd:127.69,td:"2026-10-01",ts:1790812800000,cdr:1},archived:false')


def test_parse_quote_reads_price_fields_and_timestamp():
    q = lq.parse_quote(PAGE)
    assert q["price"] == 127.69 and q["open"] == 126.3 and q["high"] == 128.15
    assert q["prev_close"] == 126.3 and q["volume"] == 3413347
    assert q["market_state"] == "closed"
    assert q["quote_time"].startswith("2026-10-01")


def test_parse_quote_without_a_quote_is_empty():
    assert lq.parse_quote("<html>nothing</html>") == {}
    assert lq.parse_quote('quote:{p:null,c:1}') == {}


def test_market_open_follows_the_egx_week():
    tz = lq.CAIRO
    assert lq.market_open(pd.Timestamp("2026-10-04 11:00", tz=tz))       # Sunday
    assert not lq.market_open(pd.Timestamp("2026-10-02 11:00", tz=tz))   # Friday
    assert not lq.market_open(pd.Timestamp("2026-10-04 09:30", tz=tz))   # before open
    assert not lq.market_open(pd.Timestamp("2026-10-04 15:00", tz=tz))   # after close
