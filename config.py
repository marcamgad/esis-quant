"""
ESIS-QUANT configuration.

Layer 1 (data infrastructure) is the only layer implemented so far.
All parameters live here; nothing below should be hardcoded elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "esis_quant.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "esis_quant.log")
LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB rotation
LOG_BACKUP_COUNT = 5
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------
HISTORY_START_DATE = "2015-01-01"
# Default recent-snapshot window for run_layer1 (short so a routine run
# isn't a full multi-year Yahoo backfill); Layer 2 backfills from
# HISTORY_START_DATE explicitly.
LAYER1_DEFAULT_HISTORY_DAYS = 90

# ---------------------------------------------------------------------------
# EGX ticker universe
# ---------------------------------------------------------------------------
# Verified against Yahoo Finance (each resolved to real OHLCV history as of
# 2026-07-07). name_english is filled ONLY where yfinance's own metadata
# confirmed it independently -- everything else is left as None rather than
# guessed, since this feeds a database used for real trading decisions.
#
# THIS IS A SEED LIST, NOT A COMPLETE EGX100 UNIVERSE. Sector, sub_sector,
# egx30_member, egx70_member, is_soe, float_pct and listing_date are NOT
# populated here -- they require an authoritative source (EGX official
# disclosures, Mubasher, or manual entry) and must be filled in before this
# data is used for factor construction (Layer 3) or live trading.
YFINANCE_SUFFIX = ".CA"

# Data-fetch reliability (audit fix FETCH-01). yfinance is a scraped, rate-
# limited, best-effort source; a transient failure must be retried and, if it
# still fails, reported as a FAILURE -- never silently treated as "no data".
FETCH_RETRIES = 3
FETCH_BACKOFF_SECONDS = 2.0

EGX_SEED_TICKERS = [
    {"ticker": "COMI", "name_english": "Commercial International Bank Egypt (CIB)"},
    {"ticker": "HRHO", "name_english": "EFG Holding Company"},
    {"ticker": "SWDY", "name_english": None},
    {"ticker": "TMGH", "name_english": None},
    {"ticker": "EAST", "name_english": "Eastern Company"},
    {"ticker": "ORHD", "name_english": None},
    {"ticker": "EFIH", "name_english": None},
    {"ticker": "ABUK", "name_english": "Abu Qir Fertilizers & Chemical Industries"},
    {"ticker": "ETEL", "name_english": None},
    {"ticker": "ORWE", "name_english": None},
    {"ticker": "MFPC", "name_english": None},
    {"ticker": "AMOC", "name_english": None},
    {"ticker": "SKPC", "name_english": None},
    {"ticker": "EKHO", "name_english": "Valmore Holding"},
    {"ticker": "PHDC", "name_english": None},
    {"ticker": "CIRA", "name_english": "CIRA Education"},
    {"ticker": "ISPH", "name_english": None},
    {"ticker": "OLFI", "name_english": None},
    {"ticker": "JUFO", "name_english": None},
    {"ticker": "EGAL", "name_english": None},
    {"ticker": "ADIB", "name_english": None},
    {"ticker": "FWRY", "name_english": None},
    {"ticker": "CLHO", "name_english": None},
    {"ticker": "EFID", "name_english": None},
    {"ticker": "SUGR", "name_english": None},
    {"ticker": "ORAS", "name_english": None},
    {"ticker": "ELSH", "name_english": None},
    {"ticker": "CCAP", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "HELI", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "OIH", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "DSCW", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "ISMQ", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "EMFD", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "GBCO", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "MTIE", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "BTFH", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "OCDI", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "OFH", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "IRON", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "SPMD", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "ACGC", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "ZMID", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "ATQA", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "CIEB", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "PHAR", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "AMER", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
    {"ticker": "ETRS", "name_english": None},   # added 2026-07-28 (universe expansion, top-20 by liquidity)
]

# Tickers tried but NOT resolvable on Yahoo Finance as of 2026-07-07 --
# kept here so nobody re-wastes time on them without checking first.
EGX_KNOWN_BAD_TICKERS = ["ESRS", "MNHD", "PALTIN"]

# Maps Arabic name/abbreviation -> EGX ticker, for matching company
# mentions in Arabic-language news text (English name substring matching
# never fires on Arabic text -- see data/collectors.py _match_tickers).
# Multiple entries per ticker (different name variants) are correct and
# intentional -- Arabic press uses different names for the same company
# across sources.
#
# Coverage is partial by design: entries below are only added where the
# company identity and Arabic phrasing could be confirmed from training
# data. For OLFI, EGAL, and ELSH that confidence bar wasn't met, so no
# entries are included for them (marked UNCONFIRMED below) -- a missing
# entry just means those tickers won't be matched yet; a wrong one would
# silently mislabel training data for Layer 4, which is worse.
ARABIC_COMPANY_NAMES: dict = {
    # COMI -- Commercial International Bank (CIB)
    "البنك التجاري الدولي": "COMI",
    "سي آي بي": "COMI",
    "CIB": "COMI",
    "التجاري الدولي": "COMI",
    # HRHO -- EFG Hermes / EFG Holding
    "هيرميس": "HRHO",
    "إي إف جي هيرميس": "HRHO",
    "EFG": "HRHO",
    "هيرمس القابضة": "HRHO",
    "إي إف جي القابضة": "HRHO",
    # SWDY -- Elsewedy Electric
    "السويدي إليكتريك": "SWDY",
    "السويدي للكابلات": "SWDY",
    "السويدي": "SWDY",
    "إليكتريك السويدي": "SWDY",
    # TMGH -- Talaat Moustafa Group
    "طلعت مصطفى": "TMGH",
    "مجموعة طلعت مصطفى": "TMGH",
    "TMG": "TMGH",
    # EAST -- Eastern Company
    "الشركة الشرقية للدخان": "EAST",
    "الشرقية للدخان": "EAST",
    "الشرقية": "EAST",
    # ORHD -- Orascom Development / Orascom Holding
    "أوراسكوم": "ORHD",
    "اوراسكوم القابضة": "ORHD",
    "Orascom": "ORHD",
    # EFIH -- EFG Finance / EFIH
    "إي إف جي": "EFIH",
    "EFG Finance": "EFIH",
    # ABUK -- Abu Qir Fertilizers
    "أبو قير للأسمدة": "ABUK",
    "أبو قير": "ABUK",
    "أبوقير للأسمدة والكيماويات": "ABUK",
    # ETEL -- Telecom Egypt
    "المصرية للاتصالات": "ETEL",
    "تيليكوم مصر": "ETEL",
    "Telecom Egypt": "ETEL",
    "TE": "ETEL",
    # ORWE -- Oriental Weavers (carpets)
    "الشرقية للسجاد": "ORWE",
    # MFPC -- Misr Fertilizers Production Company (MOPCO)
    "موبكو": "MFPC",
    "مصر لإنتاج الأسمدة": "MFPC",
    # OLFI -- UNCONFIRMED, needs manual lookup (Obour Land for Food
    # Industries is the likely company but exact Arabic press phrasing
    # could not be confirmed from training data)
    # AMOC -- Alexandria Mineral Oils Company
    "أموك": "AMOC",
    "الإسكندرية للزيوت المعدنية": "AMOC",
    "AMOC": "AMOC",
    # SKPC -- Sidi Kerir Petrochemicals
    "سيدي كرير": "SKPC",
    "سيدي كرير للبتروكيماويات": "SKPC",
    "SKPC": "SKPC",
    # EKHO -- Valmore Holding (formerly EK Holding)
    "فالمور": "EKHO",
    "إي كيه القابضة": "EKHO",
    "EK Holding": "EKHO",
    # PHDC -- Palm Hills Development
    "بالم هيلز": "PHDC",
    "بالم هيلز للتعمير": "PHDC",
    "Palm Hills": "PHDC",
    # CIRA -- CIRA Education
    "سيرا للتعليم": "CIRA",
    "CIRA": "CIRA",
    "سيرا التعليمية": "CIRA",
    # ISPH -- Integrated Diagnostics Holdings (IDH)
    "إيدي هولدينغ": "ISPH",
    "التشخيص المتكامل": "ISPH",
    "IDH": "ISPH",
    # JUFO -- Juhayna Food Industries
    "جهينة": "JUFO",
    "جهينة للصناعات الغذائية": "JUFO",
    # EGAL -- UNCONFIRMED, needs manual lookup (possibly Egyptalum/Egypt
    # Aluminium Company, but identity and Arabic phrasing could not be
    # confirmed with enough confidence from training data)
    # ADIB -- Abu Dhabi Islamic Bank Egypt
    "أبوظبي الإسلامي": "ADIB",
    "بنك أبوظبي الإسلامي مصر": "ADIB",
    "ADIB": "ADIB",
    # FWRY -- Fawry for Banking and Payment Technology Services
    "فوري": "FWRY",
    # CLHO -- Cleopatra Hospitals Group
    "كليوباترا للمستشفيات": "CLHO",
    "مجموعة كليوباترا": "CLHO",
    # EFID -- Edita Food Industries
    "إيديتا": "EFID",
    "إيديتا للصناعات الغذائية": "EFID",
    # SUGR -- Delta Sugar
    "دلتا للسكر": "SUGR",
    "شركة دلتا للسكر": "SUGR",
    # ORAS -- Orascom Construction
    "أوراسكوم للإنشاء": "ORAS",
    "اوراسكوم للانشاء والصناعة": "ORAS",
    # ELSH -- UNCONFIRMED, needs manual lookup (company identity could not
    # be confirmed with confidence from training data)
}

# ---------------------------------------------------------------------------
# Macro data sources
# ---------------------------------------------------------------------------
# All confirmed working via yfinance on 2026-07-07.
MACRO_YFINANCE_TICKERS = {
    "usd_egp": "EGP=X",
    "brent_oil_usd": "BZ=F",
    "gold_usd": "GC=F",
    "natural_gas_usd": "NG=F",
    "us_10y_yield": "^TNX",
}

# EGX30 index: Yahoo has NO reliable multi-day history under any symbol we
# tried (^EGX30, ^EGX30.CA, EGX30.CA, CASE30.CA all return empty; ^CASE30
# returns only a same-day snapshot). Treat egx30_level/egx30_volume in
# macro_daily as best-effort/incomplete until a better source is wired in
# (e.g. reconstructing a proxy index from EGX_SEED_TICKERS, or a paid feed).
EGX30_PROXY_TICKER = "^CASE30"

# Free tier, no API key required for the "latest" endpoint. Used as a
# cross-check against EGP=X; not used for historical backfill (exchangerate
# -api.com free tier has no history endpoint).
EXCHANGERATE_API_LATEST_URL = "https://open.er-api.com/v6/latest/USD"

# ---------------------------------------------------------------------------
# News scraping
# ---------------------------------------------------------------------------
# Confirmed working (2026-07-07): headline links found via h1/h2/h3 > a on
# listing pages, article body under .article-content, published time in
# <meta property="article:published_time">.
ALMAL_BASE_URL = "https://almalnews.com"
ALMAL_CATEGORY_PATH = "/category/stocks/"
SCRAPE_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ESIS-QUANT-research/0.1"
SCRAPE_TIMEOUT_SECONDS = 10
SCRAPE_REQUEST_DELAY_SECONDS = 1.0  # politeness delay between requests

# CBE (cbe.org.eg) sits behind a WAF that rejects plain requests library
# calls outright ("Request Rejected" / support-ID page, not the real
# content) -- confirmed 2026-07-07. Scraping it needs a real browser
# (Playwright/Selenium) at minimum, and may still be blocked. Not
# implemented in Layer 1; see data/collectors.py scrape_cbe_decisions().
CBE_MPC_DECISIONS_URL = "https://www.cbe.org.eg/en/monetary-policy/mpc-meeting-decisions"

# ---------------------------------------------------------------------------
# Transaction costs (used by later backtest/portfolio layers)
# ---------------------------------------------------------------------------
# THERE IS ONE COMMISSION MODEL AND IT LIVES IN backtest/fees.py.
#
# THNDR_COMMISSION_PCT (0.175%/side) and ROUND_TRIP_COST_PCT used to live here
# and were what every backtest actually charged. They are WRONG in structure,
# not just in level: Thndr charges a FLAT EGP 4.95 per order plus 0.112%
# (MEASURED 2026-08-04 from 8 real contract notes), and a percentage-only
# constant cannot express a flat fee. The 2026-08-05 audit found the real
# schedule in backtest/fees.py was never wired into any headline number, so
# every published CAGR/Sharpe was computed at ~0.55% round trip against a
# measured 1.5% (10k book) to 8.2% (1k book).
#
# Both constants are now REMOVED. Accessing either raises (see __getattr__
# below) rather than silently returning a number that flatters results. Code
# that needs a percentage-equivalent must DERIVE it from the schedule at a
# stated book size via backtest.fees.book_pct_per_side -- so the one source of
# truth stays the source of truth.
SLIPPAGE_PCT = 0.0010

# A flat per-order fee is meaningless without a capital base, so every backtest
# that charges the real schedule must state the book it is charging against.
# This is the LIVE book size, chosen so headline numbers describe the account
# that actually exists -- not a scale-free abstraction in which flat fees
# conveniently vanish. Changing it changes every reported number, deliberately.
BACKTEST_DEFAULT_AUM_EGP = 10_000.0

_REMOVED_COST_CONSTANTS = {
    "THNDR_COMMISSION_PCT": "a flat EGP-per-order fee cannot be expressed as a percentage",
    "ROUND_TRIP_COST_PCT": "derived from the above, so equally wrong",
}


def __getattr__(name):
    """Fail loudly on the removed percentage-only cost constants (PEP 562).

    A silent fallback here would reintroduce exactly the bug the audit found:
    two cost models coexisting, with the flattering one winning by default.
    """
    if name in _REMOVED_COST_CONSTANTS:
        raise AttributeError(
            f"config.{name} was REMOVED: {_REMOVED_COST_CONSTANTS[name]}. "
            "Use backtest.fees (one_side_cost / book_pct_per_side) -- it is the "
            "single cost model. See the 2026-08-05 audit, P1."
        )
    raise AttributeError(f"module 'config' has no attribute '{name}'")

# Tiered / liquidity-aware slippage (backtest.costs). The flat 0.10% slippage
# above is optimistic for illiquid EGX names traded at size. Real slippage
# grows with PARTICIPATION = trade_value / average-daily-value-traded, via a
# square-root market-impact law slippage = base + IMPACT_COEF * sqrt(part).
# IMPACT_COEF is a conservative literature default (Almgren-style); it is NOT
# EGX-calibrated (we have no execution data), so it is a stress assumption, not
# a measurement -- stated honestly so nobody treats it as ground truth.
COST_IMPACT_COEF = 0.10          # slippage adder = 0.10 * sqrt(participation)
COST_ADV_LOOKBACK_DAYS = 20      # trailing sessions to average value-traded over
COST_MAX_PARTICIPATION = 0.20    # flag any trade above 20% of ADV as high-impact

# ---------------------------------------------------------------------------
# PRE-TRADE RISK LIMITS THAT ACTUALLY BIND  (audit 2026-08-05, P3)
# ---------------------------------------------------------------------------
# The audit found risk/ was reporting-only: COST_MAX_PARTICIPATION appeared as
# an advisory string in the order ticket's `flags` column and never stopped or
# resized anything, and the kill switch was the only control able to alter a
# decision -- and only at whole-book level. These limits are enforced per ORDER.
#
# REJECT vs RESIZE: rejecting is the default because a resized order silently
# changes the book the optimizer solved for, whereas a rejection is visible and
# leaves the decision with the operator. Set ADV_LIMIT_ACTION = "resize" to trim
# the order to the limit instead, or "warn" to restore the old advisory-only
# behaviour (which is what the audit scored 🔴).
ADV_LIMIT_ACTION = "reject"          # "reject" | "resize" | "warn"
MAX_POSITION_WEIGHT = 0.20           # no single name above this share of the book
# Daily-loss brake, deliberately DISTINCT from the kill switch: the kill switch
# flattens the whole book on a hard breach and imposes a cooldown; this only
# refuses to ADD risk on a day the book is already down badly. Softer, binds
# earlier, and cannot force a sale.
MAX_DAILY_LOSS_BLOCK_NEW = -0.05     # book down this much today -> no new buys

# ---------------------------------------------------------------------------
# LIVE-TRADING MODE (audit Tier-1 #1)
# ---------------------------------------------------------------------------
# The signal-based cross-sectional SELECTION path (BacktestEngine.run over
# Layer-3/4/5/6 alpha) measured a NEGATIVE selection contribution (~-18% in the
# Brinson attribution). Live and paper trading must run ONLY the parts that
# survived validation: the min-variance core (mu=0) + the BEAR-regime overlay.
# This flag is enforced in ops.trade_list.generate_trade_list -- if anything
# ever tries to route signal selection into the live book it must fail loudly.
LIVE_MIN_VAR_ONLY = True

# Trade the FULL EGX listing rather than the 47-name hand-picked seed.
# Set True 2026-08-10 at the operator's request. This REMOVES the selection bias
# the seed list carried (measured at ~10pp of CAGR by data/pit_universe.py) and
# will therefore REDUCE reported backtest performance -- the old numbers were
# inflated by the bias, so the drop is the correction working.
# Names are still date-gated by the liquidity/staleness filter, so thin and
# delisted names are excluded on the days they are untradable.
LIVE_UNIVERSE_FULL_EGX = True

# STALENESS GATE FOR THE TRADABLE UNIVERSE.
#
# WHY THIS IS SEPARATE FROM TFT_MAX_STALE_FRAC (0.50). That threshold was set
# for a 47-name hand-picked large-cap list, where nothing came close to it. On
# the full 202-name EGX listing it admits names that do not move on HALF their
# sessions -- and those are precisely the names a covariance estimator is most
# easily fooled by: a stale price shows near-zero correlation with everything
# else, so min-variance reads it as a perfect diversifier and piles in.
#
# Observed 2026-08-10: widening the universe produced a 9-name book of WCDF
# (49% zero-move days), GMCI (47%), NEDA (39%) and similar -- none of which had
# LOW volatility (34-38% annualised, vs SUGR at 19%). The optimizer was not
# finding stable stocks; it was finding untraded ones.
#
# 0.20 is chosen on a stated principle, not by tuning: a name that fails to move
# on more than one session in five has a covariance estimate built substantially
# on zeros, and cannot be relied on. It is not chosen to produce a particular
# book, and it removes names on both sides of any preference.
TRADABLE_MAX_STALE_FRAC = 0.20

# ---------------------------------------------------------------------------
# PRE-COMMITTED GO-LIVE GATE (audit Tier-2 #9)
# ---------------------------------------------------------------------------
# Written DOWN NOW, before any capital or ego is on the line. ops.go_live_gate
# checks the FORWARD paper record (never the backtest) against every one of
# these. All must pass before real money is justified. These are deliberately
# strict: the whole edge is one fragile mechanism (BEAR->cash) that has never
# fired live, so the gate insists on seeing it fire.
GO_LIVE_MIN_PAPER_DAYS = 189        # ~9 trading months of genuine forward record
GO_LIVE_MIN_BEAR_DAYS = 20          # must have lived through a real BEAR stretch
GO_LIVE_MIN_SHARPE = 0.80           # forward (not backtested) Sharpe floor
GO_LIVE_MAX_DRAWDOWN = -0.25        # forward MaxDD not worse than this
GO_LIVE_MIN_REAL_CAGR = 0.0         # USD/real-terms CAGR must be positive, not just nominal

# ---------------------------------------------------------------------------
# IDLE-CASH YIELD REMINDER
# ---------------------------------------------------------------------------
# When the regime overlay de-risks, freed capital sits in the brokerage account
# earning ~nothing, while data/rates.py shows Egyptian short rates running
# 8-27%. Over the backtest ~26% of days are spent in cash, so this is not a
# rounding error -- but it is only ACTIONABLE above real product minimums:
#
#   * Egyptian T-BILLS are sold in minimum subscriptions of EGP 25,000 and
#     multiples (source: Emirates NBD Egypt / standard across Egyptian banks).
#     Below that you cannot buy one at all -- a hard product constraint.
#   * MONEY MARKET FUNDS (e.g. Banque Misr "Youm b Youm", HSBC "Kol Yom") hold
#     T-bills for you with daily liquidity and far lower minimums, at a slightly
#     lower net yield after fund fees.
#
# Thresholds apply to IDLE CASH, not to book size: a large book that de-risks
# has enough idle cash to qualify; a small fully-invested one does not.
# The reminder stays SILENT below the lower tier so it never nags about sums
# too small to move.
IDLE_CASH_TBILL_MIN = 25_000.0    # direct T-bill minimum subscription
IDLE_CASH_MMF_MIN = 5_000.0       # below this, fund fees + friction dominate
IDLE_CASH_MIN_DAYS = 14           # don't suggest moving cash for a brief dip
# Relative materiality, applied ALONGSIDE the absolute product minimums above.
# Those minimums answer "which instrument can I buy"; these answer "does this
# matter". Idle cash can be too small for any cash product and still be the
# largest single drag on the book -- 3,167 EGP was 32% of a 9,987 EGP book
# costing 5.5%/yr, more than the strategy's whole expected return, and the
# absolute test kept quiet about it.
IDLE_CASH_MATERIAL_PCT = 0.20      # this share of the book sitting idle...
IDLE_CASH_MATERIAL_COST_PCT = 0.01  # ...costing at least this much of it per year

# ---------------------------------------------------------------------------
# SETTLEMENT (T+2)
# ---------------------------------------------------------------------------
# EGX equities settle T+2. Whether unsettled sale proceeds may be REINVESTED
# same-day (as opposed to withdrawn) is a broker policy that has NOT been
# verified for Thndr. Default False = conservative: the allocator may only spend
# settled cash. Set True only after confirming the broker permits it.
ALLOW_UNSETTLED_REINVESTMENT = False
SETTLEMENT_DAYS = 2          # trading sessions, not calendar days

# ---------------------------------------------------------------------------
# HARD KILL-SWITCH (audit Tier-2 #10)
# ---------------------------------------------------------------------------
# A model-independent backstop: if realized loss breaches these, flatten the
# book regardless of what the regime model believes. This is NOT the regime
# overlay (which can be wrong); it's the "the model is wrong RIGHT NOW" circuit.
KILL_SWITCH_MAX_DAILY_LOSS = -0.07     # -7% in a single session -> flatten
KILL_SWITCH_MAX_WEEKLY_LOSS = -0.12    # -12% over a rolling 5 sessions -> flatten
KILL_SWITCH_COOLDOWN_DAYS = 5          # stay flat this many sessions after a trip

# ---------------------------------------------------------------------------
# Layer 4: Arabic NLP sentiment engine
# ---------------------------------------------------------------------------
# Tokenizer
SENTIMENT_VOCAB_SIZE = 8000
SENTIMENT_SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
SENTIMENT_MAX_SEQ_LEN = 128

# Transformer architecture
SENTIMENT_D_MODEL = 256
SENTIMENT_N_HEADS = 8
SENTIMENT_N_LAYERS = 6
SENTIMENT_D_FF = 1024
SENTIMENT_DROPOUT = 0.1
SENTIMENT_N_CLASSES = 3  # Negative=0, Neutral=1, Positive=2

# Distant-supervision labeling thresholds
SENTIMENT_LABEL_RETURN_THRESHOLD = 0.015  # +/-1.5% same-day or next-day return

# Source credibility weights (article-level, multiplies into training sample weight)
SENTIMENT_SOURCE_CREDIBILITY = {
    "egx_official": 1.0,
    "almalnews": 0.85,  # matches data/collectors.py source string exactly
    "mubasher": 0.80,
    "other": 0.60,
}
SENTIMENT_MANUAL_EXAMPLE_WEIGHT = 2.0

# Training procedure
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_EPOCHS = 30
SENTIMENT_LR = 3e-4
SENTIMENT_WEIGHT_DECAY = 0.01
SENTIMENT_WARMUP_RATIO = 0.10
SENTIMENT_GRAD_CLIP_NORM = 1.0
SENTIMENT_VAL_FRACTION = 0.15  # last 15% of dates, time-based (never random)

# Daily sentiment pipeline
SENTIMENT_NEWS_LOOKBACK_HOURS = 48
SENTIMENT_ROLLING_WINDOW_DAYS = 5
SENTIMENT_SURPRISE_STD_THRESHOLD = 2.0  # |z-score| above this vs trailing window = "surprise"

# Model checkpoint (populated once a Kaggle-trained checkpoint is downloaded
# and placed here -- see models/sentiment/ and the Kaggle notebook this
# project ships for training instructions)
SENTIMENT_MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "sentiment", "checkpoints", "best_sentiment_model.pt")

# ---------------------------------------------------------------------------
# Layer 5: Causal signal extraction (Double ML + Lasso + 2SLS IV)
# ---------------------------------------------------------------------------
# Lasso coordinate descent + time-series CV over a log-spaced lambda grid
CAUSAL_LASSO_LAMBDA_MIN = 1e-4
CAUSAL_LASSO_LAMBDA_MAX = 1.0
CAUSAL_LASSO_N_LAMBDAS = 50
CAUSAL_LASSO_MAX_ITER = 1000
CAUSAL_LASSO_TOL = 1e-6

# Cross-fitting / cross-validation fold counts (time-ordered, never random)
CAUSAL_DML_N_FOLDS = 5
CAUSAL_CV_N_FOLDS = 5

# Minimum observations below which a causal estimate is flagged unreliable
CAUSAL_MIN_OBS = 100

# Weak-instrument threshold (Stock-Yogo rule of thumb) for first-stage F
CAUSAL_WEAK_IV_F_THRESHOLD = 10.0

# Signal-construction parameters
CAUSAL_FORWARD_RETURN_HORIZON = 5      # Y = 5-day forward return (spec)
CAUSAL_MOMENTUM_LOOKBACK_DAYS = 252    # 12-month formation window (Jegadeesh-Titman)
CAUSAL_MOMENTUM_SKIP_DAYS = 21         # skip most-recent month to avoid short-term reversal
CAUSAL_IV_LAG_DAYS = 5                 # lag used to build the lagged-momentum instrument
CAUSAL_SIGNAL_SENTIMENT_WEIGHT = 0.5   # causal_signal = 0.5*causal_sentiment + 0.5*causal_momentum
CAUSAL_SIGNAL_MOMENTUM_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Layer 6: Temporal Fusion Transformer (multi-horizon quantile forecasting)
# ---------------------------------------------------------------------------
# Sequence / horizon geometry
TFT_SEQUENCE_LENGTH = 60               # past observation window (trading days)
TFT_FORECAST_HORIZONS = [1, 5, 10, 20] # forward trading-day horizons to predict
TFT_QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# Architecture (hidden_size is the TFT "d_model" that every GRN/VSN/LSTM/
# attention block shares). Kept modest so it trains on a single T4 GPU.
TFT_HIDDEN_SIZE = 64
TFT_LSTM_LAYERS = 1
TFT_ATTENTION_HEADS = 4
TFT_DROPOUT = 0.1

# Training
TFT_BATCH_SIZE = 64
TFT_LR = 1e-3
TFT_WEIGHT_DECAY = 1e-4
TFT_EPOCHS = 50
TFT_EARLY_STOP_PATIENCE = 10
TFT_GRAD_CLIP_NORM = 1.0

# Walk-forward split boundaries (strict temporal ordering, never shuffled)
TFT_TRAIN_END = "2022-12-31"
TFT_VAL_END = "2023-12-31"           # test = everything after TFT_VAL_END

# Feature groups. NOTE (data honesty, mirrors Layers 3/5): the spec lists 29
# temporal / 20 static / 19 known-future features assuming full fundamental +
# options data. Only the subset actually present in the DB is assembled here;
# the model reads its input dimensions from the data at build time rather than
# hardcoding the spec counts, so it stays correct as coverage grows.
# Data-quality guards for the feature build. EGX enforces daily price-move
# limits (~10-20%), so any single-day adjusted return beyond
# TFT_RETURN_ARTIFACT_THRESHOLD is a split/data artifact (yfinance's Adj Close
# misses some EGX splits and corporate_actions is empty) -- it is neutralized
# so it can't poison returns/targets. Factor betas beyond the winsor
# percentiles are rolling-OLS artifacts on short-history stocks.
TFT_RETURN_ARTIFACT_THRESHOLD = 0.30
TFT_BETA_WINSOR_PCT = 0.005  # clip betas at 0.5% / 99.5%
# Drop a window if more than this fraction of its 60 days are zero-return
# (a halted/stale/illiquid stretch -- e.g. ORAS trades ~0.7% of test days,
# so its forward returns are pure noise and its direction is unlearnable).
# Normal liquid stocks sit ~16% zero-return days, so 0.5 only removes genuinely
# dead stretches.
TFT_MAX_STALE_FRAC = 0.5
# After train-fit z-scoring, clip standardized feature values to +/- this many
# sigmas. A single real event (e.g. the Nov-2016 EGP float, a genuine +72% FX
# day) otherwise lands at ~40 sigma and dominates the input scale; an 8-sigma
# input already says "extreme, saturate here" and anything beyond adds no
# useful gradient. Uniform across all features, applied to inputs only (never
# to the forward-return targets). (An earlier attempt at robust median/IQR
# scaling for the macro columns backfired: USD/EGP is pegged-flat most days so
# its IQR ~ 0, which amplified the float day to ~430 sigma instead of taming
# it -- clipping is the robust, non-exploding choice here.)
TFT_ZSCORE_CLIP = 8.0

TFT_MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "tft", "checkpoints", "best_tft_model.pt")
# Windowed feature tensors are exported as a compressed .npz (numpy-native, no
# parquet engine needed, keeps the (N,T,F) shapes) for upload to Kaggle.
TFT_FEATURES_NPZ_PATH = os.path.join(BASE_DIR, "kaggle_notebooks", "tft_features.npz")
