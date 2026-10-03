"""Ingest every EGX ticker on the public listing that is not already in the DB."""
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import yfinance as yf, pandas as pd
from data.database import get_connection

LISTED = """COMI SWDY TMGH ETEL EGAL QNBE EAST MFPC ABUK ALCN HDBK EFIH ADIB ORAS FWRY EMFD
SCTS ORHD EFID PHDC CANA GPPL JUFO BIOC VLMR VLMRA HRHO OCDI GBCO BTFH HELI FERC RAYA IRON
CIEB FAIT FAITA EXPA EGCH CLHO VALU ARCC PHAR CIRA CCAP MTIE SCEM TAQA EFIC EGTS MCQE SKPC
POUL NIPH ORWE MASR EGSA SAUD MOIL UBEE AMES EGBE ISPH TALM MBSC MHOT CICH RMDA ATQA AMOC
CSAG BINV IFAP MPRC OLFI PRDC MOIN MIPH ISMQ BONY OIH EGAS PHTV DOMT SPHT AFMC KORA MPCI
ZMID ELEC CPCI NINH SUGR ENGC ACAP AMIA NAPR AXPH GOUR CNFN ARAB OCPH DSCW SPIN AMER MICH
GSSC KABO MFSC SVCE WCDF GDWA UNIT OFH UEFM AJWA SDTI ADCI SAIB INFI ELSH ELKA ASCM ACTF
ACGC ACAMD ISMA LCSW GGCC KZPC SMFR ZEOT DAPH CRST ALRA CFGH ETRS EDFM ADPC NARE ATLC MILS
MPCO PHGC IDRE GPIM GGRN RACC CEFM UEGC EHDR EALR NAHO AALR ECAP MOSC WKOL SNFC PRCL MAAL
ODIN GTWL MENA SCFM DTPP NCCW CAED CERA DEIN NHPS SEIG SEIGA OBRI MEPA SIPC RREI NDRL AFDI
AIDC ALUM AMII LUTS COSG ASPI EBSC POCO PRMH RTVC UNIP TANM GTEX MCRO KRDI APSW ICID ROTO
AIHC SPMD MEGM TYCN ICLE RUBX EASB KWIN MOED RAKT AREH EEII CCRS EPCO GRCA GIHD ELWA ELNA
DGTZ DCCC MMAT NEDA TRTO EPPK GMCI EOSB CPME COPR""".split()

with get_connection() as c:
    have = {r[0] for r in c.execute("SELECT DISTINCT ticker FROM daily_ohlcv")}
todo = sorted(set(LISTED) - have)
print(f"listed={len(set(LISTED))}  already have={len(have)}  fetching={len(todo)}", flush=True)

ok = fail = 0
for i, t in enumerate(todo, 1):
    try:
        h = yf.Ticker(t + ".CA").history(start="2014-01-01", auto_adjust=False)
        if h.empty:
            fail += 1
            continue
        h = h.reset_index()
        rows = []
        for r in h.itertuples():
            close = float(r.Close); vol = float(r.Volume or 0)
            rows.append((t, pd.Timestamp(r.Date).strftime("%Y-%m-%d"), float(r.Open),
                         float(r.High), float(r.Low), close, close, vol, close * vol))
        with get_connection() as c:
            c.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, open, high, low, "
                          "close, adjusted_close, volume, value_traded_egp) VALUES (?,?,?,?,?,?,?,?,?)",
                          rows)
        ok += 1
    except Exception:
        fail += 1
    if i % 15 == 0:
        print(f"  {i}/{len(todo)} ok={ok} fail={fail}", flush=True)

print(f"DONE ingested={ok} no-data={fail}", flush=True)
with get_connection() as c:
    tot = c.execute("SELECT COUNT(DISTINCT ticker) FROM daily_ohlcv").fetchone()[0]
    live = c.execute("SELECT COUNT(DISTINCT ticker) FROM daily_ohlcv WHERE date>='2026-07-01'").fetchone()[0]
print(f"DB now: {tot} tickers total, {live} with data since 2026-07", flush=True)
