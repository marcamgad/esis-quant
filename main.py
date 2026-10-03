"""
ESIS-QUANT entry point -- Layers 1 (data infrastructure), 2 (regime
detection), 3 (factor model), 5 (causal signal extraction) implemented
locally; Layer 4 (Arabic NLP sentiment) has its architecture/training
loop/pipeline built locally, but actual model training runs on Kaggle GPU
(see kaggle_notebooks/ and run_layer4_export_dataset below) -- until a
trained checkpoint is placed at config.SENTIMENT_MODEL_CHECKPOINT_PATH,
Layer 4 scoring is a no-op. Layer 6 (Temporal Fusion Transformer) mirrors
Layer 4: the architecture + feature pipeline + charts are local, GPU
training runs on Kaggle (kaggle_notebooks/esis_tft_training.ipynb), and
inference/charts activate once a checkpoint is at
config.TFT_MODEL_CHECKPOINT_PATH. Layers 7-8 (portfolio construction,
risk/attribution) are not built yet.
"""

import logging
import logging.handlers
import os
import sys

import config
from data.database import init_db, seed_stocks, list_tables
from data.collectors import DataCollector


def init_directories():
    """Create LOG_DIR and CHARTS_DIR if they do not already exist.
    Called before any file handler or chart writer is set up.
    Uses exist_ok=True so re-running is always safe (idempotent).
    """
    import os
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHARTS_DIR, exist_ok=True)


def setup_logging():
    init_directories()
    logger = logging.getLogger("esis_quant")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        config.LOG_PATH, maxBytes=config.LOG_MAX_BYTES, backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def run_layer1(history_days: int = None):
    """Initialize the DB, seed the ticker universe, and pull a recent
    price/macro/news snapshot. `history_days` is deliberately short by
    default (a full 2015- backfill for ~27 tickers is a lot of Yahoo Finance
    calls) -- pass a larger window or config.HISTORY_START_DATE explicitly
    for a real backfill.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")
    history_days = history_days if history_days is not None else config.LAYER1_DEFAULT_HISTORY_DAYS

    init_db()
    logger.info("Tables present: %s", list_tables())

    seed_stocks()

    collector = DataCollector()

    import pandas as pd
    start = (pd.Timestamp.today() - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")

    price_result = collector.fetch_all_historical(start_date=start)
    logger.info("Price fetch result: %s", price_result)

    macro_result = collector.fetch_macro_daily(start_date=start)
    logger.info("Macro fetch result: %s", macro_result)

    news_result = collector.scrape_arabic_news(max_articles=10)
    logger.info("News scrape result: %s", news_result)

    tbill_result = collector.scan_news_for_tbill_rates()
    logger.info("T-bill discount rate scan result: %s", tbill_result)

    return {"prices": price_result, "macro": macro_result, "news": news_result, "tbill_events": tbill_result}


def run_layer2(backfill_start_date: str = None, make_chart: bool = True):
    """Backfill enough history for the regime HMM to be meaningfully
    identified (a 4-state, 6-feature Gaussian HMM has a lot of covariance
    parameters per state -- the ~90-day Layer 1 default would leave almost
    nothing after warm-up windows), then fit+decode regimes and persist
    them. Backfilling is idempotent (INSERT OR REPLACE / upsert), so
    re-running this is safe.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")
    backfill_start_date = backfill_start_date or config.HISTORY_START_DATE

    init_db()
    seed_stocks()

    collector = DataCollector()
    price_result = collector.fetch_all_historical(start_date=backfill_start_date)
    logger.info("Layer 2 backfill -- price fetch result: %s", price_result)
    macro_result = collector.fetch_macro_daily(start_date=backfill_start_date)
    logger.info("Layer 2 backfill -- macro fetch result: %s", macro_result)

    from models.regime_pipeline import run_regime_detection
    result = run_regime_detection()
    logger.info(
        "Layer 2 regime detection: %d days, regime counts=%s",
        result["n_days"], result["regime_counts"],
    )

    if make_chart:
        from visualization.regime_charts import plot_regime_analysis
        chart_path = plot_regime_analysis(result)
        logger.info("Regime chart saved to %s", chart_path)
        result["chart_path"] = chart_path

    return result


def run_layer3(make_chart: bool = True):
    """Factor model: per-stock rolling exposures against the 5 available
    factors (models/factor_returns.py documents why only 5 of the spec's 9
    factors exist), Fama-MacBeth cross-sectional risk premia, and a
    regime-conditioned composite alpha score. Assumes Layer 1 history and
    Layer 2 regime_history are already populated (run_layer2 does both) --
    doesn't re-backfill on its own.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")

    from data.database import get_connection
    from models.factor_model import compute_factor_exposures, store_factor_exposures
    from models.factor_returns import build_factor_returns
    from models.fama_macbeth import run_fama_macbeth
    from models.alpha_score import compute_alpha_scores, store_alpha_scores

    exposures = compute_factor_exposures()
    store_factor_exposures(exposures)
    logger.info("Layer 3: stored %d factor_exposures rows across %d tickers",
                len(exposures), exposures["ticker"].nunique())

    fm_result = run_fama_macbeth()
    logger.info("Layer 3: Fama-MacBeth premia:\n%s", fm_result["premia"].to_string())

    alpha_scores = compute_alpha_scores(fm_result["premia"]["premium"])
    store_alpha_scores(alpha_scores)
    logger.info("Layer 3: stored %d alpha_score rows", len(alpha_scores))

    result = {
        "exposures": exposures,
        "premia": fm_result["premia"],
        "monthly_premia": fm_result["monthly"],
        "alpha_scores": alpha_scores,
    }

    if make_chart:
        from visualization.factor_charts import plot_factor_analysis
        factor_returns = build_factor_returns()
        with get_connection() as conn:
            import pandas as pd
            regime_history = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
        chart_path = plot_factor_analysis(factor_returns, exposures, fm_result["premia"], regime_history, alpha_scores)
        logger.info("Factor analysis chart saved to %s", chart_path)
        result["chart_path"] = chart_path

    return result


def run_layer4_export_dataset(csv_path: str = None):
    """Layer 4, the part that runs locally: re-tag any previously-scraped
    articles with the current Arabic ticker matcher (in case it's
    improved since they were scraped -- see data/collectors.py
    retag_existing_articles), build the combined distant-supervision +
    manual-bootstrap training dataset, and export it to CSV for upload to
    Kaggle. Actual model training happens in the Kaggle notebook this
    project ships (models/sentiment/ has the identical architecture
    locally for the daily scoring pipeline once a trained checkpoint
    comes back) -- see kaggle_notebooks/esis_sentiment_training.ipynb.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")

    from data.collectors import DataCollector
    from models.sentiment.dataset import build_training_dataset, export_training_csv

    retag_result = DataCollector().retag_existing_articles()
    logger.info("Layer 4: retag_existing_articles: %s", retag_result)

    df = build_training_dataset()
    logger.info("Layer 4: training dataset assembled, %d rows", len(df))

    csv_path = csv_path or os.path.join(config.BASE_DIR, "kaggle_notebooks", "sentiment_training_data.csv")
    export_training_csv(df, csv_path)
    logger.info("Layer 4: exported training dataset to %s -- upload this to Kaggle as a Dataset", csv_path)

    return {"retag": retag_result, "n_rows": len(df), "csv_path": csv_path}


def run_layer4_score_and_chart(scrape_pages: int = 0, make_chart: bool = True):
    """Layer 4 operational pass, run AFTER a trained checkpoint is placed at
    config.SENTIMENT_MODEL_CHECKPOINT_PATH:

      1. (optional) scrape `scrape_pages` more listing pages of Arabic news
         to keep growing the corpus (0 = skip network, score what's stored),
      2. re-tag stored articles against the current ticker dictionaries,
      3. backfill_daily_sentiment -> score every ticker-tagged article at its
         historical publish date into daily_sentiment (the full series the
         charts and Layer 5 DML need, not just a live 48h window),
      4. render the sentiment chart from the real scored data.

    No-ops gracefully (logs a warning, no crash) if no checkpoint exists yet.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")

    from data.collectors import DataCollector
    from models.sentiment.pipeline import SentimentPipeline

    collector = DataCollector()
    if scrape_pages > 0:
        scrape_result = collector.scrape_arabic_news(max_articles=scrape_pages * 15, pages=scrape_pages)
        logger.info("Layer 4: scrape result: %s", scrape_result)
    retag = collector.retag_existing_articles()
    logger.info("Layer 4: retag result: %s", retag)

    pipeline = SentimentPipeline()
    backfilled = pipeline.backfill_daily_sentiment()
    logger.info("Layer 4: backfilled %d daily_sentiment rows", len(backfilled))

    result = {"retag": retag, "daily_sentiment_rows": len(backfilled)}
    if make_chart and not backfilled.empty:
        from visualization.sentiment_charts import plot_sentiment_analysis
        chart_path = plot_sentiment_analysis()
        logger.info("Layer 4: sentiment chart saved to %s", chart_path)
        result["chart_path"] = chart_path
    return result


def run_layer5(make_chart: bool = True):
    """Causal signal extraction: build the (ticker, date) panel, estimate
    the causal effect of sentiment (Double ML) and momentum (2SLS IV) on
    5-day forward returns net of factor/macro confounders, construct and
    store the per-stock causal_signal. Assumes Layer 1 prices, Layer 2
    regime_history and Layer 3 factor_exposures are populated. When daily
    sentiment isn't available yet the sentiment effect degrades to 0 (logged)
    and the signal is carried by the momentum-IV component -- see
    models/causal/signals.py.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")

    from models.causal.signals import run as run_causal

    result = run_causal(store=True)
    iv = result["results"]["iv_momentum"]
    logger.info(
        "Layer 5: stored %d causal_signal rows; momentum IV theta=%.5f (first-stage F=%.1f)",
        len(result["signals"]), iv.get("theta", float("nan")), iv.get("first_stage_f", {}).get("f_stat", float("nan")),
    )

    if make_chart:
        from visualization.causal_charts import generate_causal_charts
        chart_path = generate_causal_charts(run_output=result)
        logger.info("Causal analysis chart saved to %s", chart_path)
        result["chart_path"] = chart_path

    return result


def run_layer6_export_features():
    """Layer 6, local part: assemble the real 60-day feature windows from the
    DB (Layers 1/3/4/5) and export them to config.TFT_FEATURES_NPZ_PATH for
    upload to Kaggle. Actual GPU training runs in the Kaggle notebook this
    project ships (kaggle_notebooks/esis_tft_training.ipynb); the identical
    TFT architecture lives locally for inference/charts once a trained
    checkpoint is downloaded to config.TFT_MODEL_CHECKPOINT_PATH.
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")
    from models.tft.features import export_features_npz
    path = export_features_npz()
    logger.info("Layer 6: exported TFT feature windows to %s -- upload to Kaggle", path)
    return {"npz_path": path}


def run_layer6_charts(make_chart: bool = True):
    """Layer 6 inference/charts, run AFTER a trained checkpoint is placed at
    config.TFT_MODEL_CHECKPOINT_PATH: load the TFT and render the probabilistic
    forecast / interpretability figure over the real test split. No-ops with a
    warning if no checkpoint exists yet.
    """
    import os
    init_directories()
    logger = logging.getLogger("esis_quant.main")
    if not os.path.exists(config.TFT_MODEL_CHECKPOINT_PATH):
        logger.warning("Layer 6: no TFT checkpoint at %s -- train on Kaggle first "
                       "(kaggle_notebooks/esis_tft_training.ipynb)", config.TFT_MODEL_CHECKPOINT_PATH)
        return {"chart_path": None}
    result = {}
    if make_chart:
        from visualization.tft_charts import generate_tft_charts
        result["chart_path"] = generate_tft_charts()
        logger.info("Layer 6: TFT chart saved to %s", result["chart_path"])
    return result


def run_daily_job(send_alert: bool = False, refresh_days: int = 30):
    """PRODUCTION DAILY JOB (audit Part 2.2). Pull fresh data, rebuild the
    tradable universe, re-run regimes, emit + log a trade list, and mark the
    forward paper record. HARD-FAILS (raises) rather than emitting a trade
    list from stale/degraded data.

    Order matters and every step can abort the run:
      1. validate_startup()            -- config/DB sane, or stop.
      2. fetch prices                  -- FetchError -> failures reported; if
                                          ANY ticker failed we do not trade.
      3. rebuild tradable_universe     -- the shared eligibility definition.
      4. re-run regimes (walk-forward) -- the overlay's input.
      5. assert_data_fit_to_trade()    -- staleness/universe gate, or stop.
      6. generate_trade_list()         -- logged to trade_list_log.
      7. update_paper_nav()            -- independent forward record.

    Schedule (Windows Task Scheduler / cron), after the EGX close:
        py -3.11 -c "import main; main.setup_logging(); main.run_daily_job()"
    e.g. cron:  0 17 * * 0-4  (Sun-Thu, EGX trading week)
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")

    from ops.validate import validate_startup
    from ops.trade_list import generate_trade_list, format_trade_list
    from ops.paper_trading import update_paper_nav, paper_track_record
    from data.quality import data_quality_report, format_report
    from data.universe import build_tradable_universe
    from data.collectors import DataCollector

    validate_startup(require_trading=True)

    import pandas as pd
    start = (pd.Timestamp.today() - pd.Timedelta(days=refresh_days)).strftime("%Y-%m-%d")
    # Refresh the LIVE universe, not config.EGX_SEED_TICKERS. The seed list is 47
    # names and stopped being the universe when the live set widened to 203; the
    # default argument silently kept refreshing the old 47 while
    # build_tradable_universe() below evaluated all 203. The 156 unrefreshed names
    # went stale, the staleness gate fired, and no trade list was produced for
    # eleven days -- a data-plumbing bug that presented as a scheduler failure.
    from data.universe import live_universe_tickers
    fetch = DataCollector().fetch_all_historical(tickers=live_universe_tickers(),
                                                 start_date=start)
    if not fetch.get("ok", True):
        raise RuntimeError(f"run_daily_job: price fetch FAILED for {fetch['failed']} -- "
                           "refusing to trade on an incomplete universe")

    # Second reference source. Advisory only -- it never writes prices into
    # daily_ohlcv (see data.stockanalysis for why an untimestamped scrape must
    # not become a mark). It supplies company names and market cap, which Yahoo
    # does not carry at all, and cross-checks Yahoo's closes so a vendor error
    # is distinguishable from a market move. Never fatal: a second opinion going
    # missing must not stop the primary pipeline.
    try:
        from data.stockanalysis import refresh as _ref_refresh
        _r = _ref_refresh()
        logger.info("reference refresh: %d rows, %d/%d prices disagree >5%% with Yahoo",
                    _r["rows"], _r["flagged"], _r["compared"])
    except Exception as exc:
        logger.warning("reference refresh skipped (non-fatal): %s", exc)

    build_tradable_universe()
    from models.regime_pipeline import run_regime_detection
    # Frozen vintages: the daily run only filters forward (seconds). The
    # descriptive full-history fit behind regime_statistics runs on Sundays.
    run_regime_detection(mode="walkforward", describe=pd.Timestamp.today().dayofweek == 6)

    # The LIVE exposure comes from the ensemble, and nothing used to refresh it:
    # it froze at 2026-07-30 and every ticket since said exposure 100%.
    from models.regime_ensemble import refresh_if_stale
    ens = refresh_if_stale()
    logger.info("regime ensemble: rebuilt=%s last_date=%s fits=%s",
                ens.get("rebuilt"), ens.get("last_date"), ens.get("n_fits"))

    quality = data_quality_report()
    logger.info("Data quality:\n%s", format_report(quality))
    if not quality["ok"]:
        raise RuntimeError("run_daily_job: data quality gate FAILED: " + "; ".join(quality["fatal"]))

    trade = generate_trade_list()
    logger.info("Trade list:\n%s", format_trade_list(trade))
    update_paper_nav()
    record = paper_track_record()
    logger.info("Forward paper record: %s", record)

    if send_alert:
        from alerts.telegram_bot import send_brief
        send_brief()
    return {"trade_list": trade, "quality": quality, "paper": record, "fetch": fetch}


def run_delivery(send_alert: bool = False):
    """Phase D: run the full risk-managed system (min-variance core + walk-
    forward BEAR overlay), then produce the master dashboard, the markdown
    performance report, and the Telegram morning brief. `send_alert` only
    sends if Telegram credentials are configured; otherwise it composes the
    brief and logs it (never sends silently).
    """
    init_directories()
    logger = logging.getLogger("esis_quant.main")
    from portfolio.system import run_full_system
    from visualization.master_dashboard import generate_master_dashboard
    from reports.generate_report import generate_final_report
    from alerts.telegram_bot import send_brief

    result = run_full_system()
    fs = result["metrics"]["full_system"]
    logger.info("Full system: Sharpe=%.2f MaxDD=%.1f%% Calmar=%.2f",
                fs["sharpe"], fs["max_drawdown"] * 100, fs["calmar"])
    dash = generate_master_dashboard(result)
    report = generate_final_report(result)
    brief = send_brief(result, dry_run=not send_alert)
    logger.info("Delivery: dashboard=%s report=%s brief_sent=%s", dash, report, brief["sent"])
    return {"dashboard": dash, "report": report, "brief_sent": brief["sent"], "metrics": fs}


if __name__ == "__main__":
    setup_logging()
    # Full daily signal path, in dependency order. Layer 4 SCORING (populating
    # daily_sentiment) runs right after Layer 1 so sentiment shares the same
    # freshly-pulled price/news dates the downstream merges need; it is the
    # scoring pass, distinct from run_layer4_export_dataset (a manual Kaggle
    # training-set maintenance command, deliberately NOT on this path).
    summary = run_layer1()
    print(summary)
    layer4_scores = run_layer4_score_and_chart()
    print(layer4_scores)
    layer2_summary = run_layer2()
    print({k: v for k, v in layer2_summary.items() if k not in ("obs_df", "gamma", "hmm", "stats_df")})
    layer3_summary = run_layer3()
    print({k: v for k, v in layer3_summary.items() if k not in ("exposures", "alpha_scores", "monthly_premia")})
    layer5_summary = run_layer5()
    print({k: v for k, v in layer5_summary.items() if k not in ("panel", "results", "signals")})
    # Layer 6: always regenerate the feature export + chart from CURRENT data,
    # never ship a stale npz/checkpoint left over from an earlier session.
    layer6_features = run_layer6_export_features()
    print(layer6_features)
    layer6_charts = run_layer6_charts()
    print(layer6_charts)
