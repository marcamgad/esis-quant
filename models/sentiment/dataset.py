"""
Training data creation for Layer 4 (spec section 4.3): distant supervision
from real news_articles + daily_ohlcv, plus a small manually-labeled
bootstrap set, combined with source-credibility/manual-label weighting
and class-imbalance weights for the loss function.

HONESTY NOTE ON DATA VOLUME: as of this build, news_articles has only a
handful of real scraped rows (Layer 1's scraper pulls ~10-20 articles per
run and this is a new project) -- nowhere near enough real examples to
train a 6-layer Transformer to the spec's target performance (macro F1
0.65-0.75). The distant-supervision function below is fully real and will
scale automatically as more articles accumulate from repeated Layer 1
runs; the 50 manual examples are a real, if small, supplement. Training on
what exists today is a proof-of-concept/pipeline-correctness exercise, not
a claim of hitting the target metrics yet.
"""

import logging
import os

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.sentiment.dataset")

NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2


def label_from_returns(r_t: float, r_t1: float, threshold: float = None) -> int:
    """Distant-supervision labeling rule (spec 4.3):
    POSITIVE if R_T > threshold or R_{T+1} > threshold
    NEGATIVE if R_T < -threshold or R_{T+1} < -threshold
    NEUTRAL otherwise
    """
    threshold = threshold if threshold is not None else config.SENTIMENT_LABEL_RETURN_THRESHOLD
    if r_t > threshold or r_t1 > threshold:
        return POSITIVE
    if r_t < -threshold or r_t1 < -threshold:
        return NEGATIVE
    return NEUTRAL


def build_distant_supervision_dataset(tickers=None) -> pd.DataFrame:
    """For each news_articles row with at least one ticker mention, for
    each mentioned ticker: find the first daily_ohlcv close on/after the
    article's publish date (date T) and the following trading day (T+1),
    plus the trading day immediately before T (T-1, needed to compute
    R_T itself as a return). Apply label_from_returns; weight by source
    credibility (config.SENTIMENT_SOURCE_CREDIBILITY).

    Returns columns: text, label, weight, date, ticker, source. Empty
    (not raising) if no articles have ticker mentions yet -- a cold-start
    database is a valid, expected state, not an error.
    """
    with get_connection() as conn:
        articles = pd.read_sql(
            "SELECT headline_arabic, body_arabic, url, published_at, tickers_mentioned, source "
            "FROM news_articles WHERE tickers_mentioned IS NOT NULL AND tickers_mentioned != ''",
            conn,
        )
        # adjusted_close aliased to close: distant-supervision return labels
        # must be on the adjusted series so an ex-div/split day is not
        # mislabeled as a large negative-return (NEGATIVE) example (SYS-CRIT-01).
        prices = pd.read_sql(
            "SELECT ticker, date, adjusted_close AS close FROM daily_ohlcv ORDER BY ticker, date",
            conn, parse_dates=["date"],
        )

    if articles.empty:
        logger.warning("build_distant_supervision_dataset: no news_articles with ticker mentions found")
        return pd.DataFrame(columns=["text", "label", "weight", "date", "ticker", "source"])

    if tickers is not None:
        prices = prices[prices["ticker"].isin(tickers)]

    price_by_ticker = {t: df.set_index("date")["close"].sort_index() for t, df in prices.groupby("ticker")}

    rows = []
    for _, article in articles.iterrows():
        published = pd.to_datetime(article["published_at"]) if article["published_at"] else pd.NaT
        if pd.isna(published):
            continue
        pub_date = published.normalize()
        mentioned = [t for t in (article["tickers_mentioned"] or "").split(",") if t]
        text = ((article["headline_arabic"] or "") + " " + (article["body_arabic"] or "")).strip()
        if not text:
            continue

        for ticker in mentioned:
            close = price_by_ticker.get(ticker)
            if close is None:
                continue
            on_or_after = close.index[close.index >= pub_date]
            if len(on_or_after) < 2:
                continue
            date_t, date_t1 = on_or_after[0], on_or_after[1]
            before_t = close.index[close.index < date_t]
            if len(before_t) == 0:
                continue
            date_t_minus_1 = before_t[-1]

            r_t = close.loc[date_t] / close.loc[date_t_minus_1] - 1
            r_t1 = close.loc[date_t1] / close.loc[date_t] - 1
            label = label_from_returns(r_t, r_t1)

            credibility = config.SENTIMENT_SOURCE_CREDIBILITY.get(
                article["source"], config.SENTIMENT_SOURCE_CREDIBILITY["other"]
            )
            rows.append({
                "text": text, "label": label, "weight": credibility,
                "date": pub_date, "ticker": ticker, "source": article["source"],
            })

    result = pd.DataFrame(rows)
    logger.info(
        "build_distant_supervision_dataset: %d labeled (article, ticker) examples from %d articles",
        len(result), len(articles),
    )
    return result


# ---------------------------------------------------------------------------
# Manual bootstrap examples (spec 4.3: "50 example articles with labels in
# each direction"). These are GENERIC, TEMPLATED Arabic financial-news
# sentiment patterns -- deliberately written about "the company" (الشركة)
# rather than any specific real EGX-listed firm, and deliberately without
# specific fabricated figures/events, so nothing here could be mistaken for
# a real report about a real company. They exist only to give the model a
# few clean examples of each sentiment direction's typical phrasing before
# real distant-supervision data accumulates.
# ---------------------------------------------------------------------------
_POSITIVE_EXAMPLES = [
    "حققت الشركة أرباحا قياسية خلال الربع الحالي متجاوزة توقعات المحللين",
    "ارتفعت أرباح الشركة بشكل ملحوظ مدعومة بنمو قوي في الإيرادات",
    "أعلنت الشركة عن توزيعات أرباح نقدية مجزية للمساهمين",
    "تصدرت الشركة قائمة الأسهم الأكثر ارتفاعا في تعاملات اليوم",
    "وقعت الشركة اتفاقية استراتيجية جديدة تعزز حضورها في السوق",
    "أشاد المحللون بأداء الشركة القوي ورفعوا توقعاتهم المستقبلية",
    "سجلت إيرادات الشركة نموا قويا مقارنة بالعام الماضي",
    "حصلت الشركة على تصنيف ائتماني مرتفع من وكالة تصنيف عالمية",
    "أعلنت الشركة عن توسعات جديدة في أسواق واعدة",
    "استحوذت الشركة على حصة سوقية إضافية بفضل استراتيجيتها الناجحة",
    "أغلق سهم الشركة على ارتفاع قوي مدعوما بتفاؤل المستثمرين",
    "أعلنت الشركة عن نجاح إصدار سندات جديد بإقبال قوي من المستثمرين",
    "تجاوزت مبيعات الشركة التوقعات بفارق كبير خلال الموسم الأخير",
    "أعلن مجلس إدارة الشركة عن خطة نمو طموحة للسنوات القادمة",
    "استقبل السوق نتائج الشركة بإيجابية كبيرة مع ارتفاع حجم التداول",
    "أعلنت الشركة عن شراكة جديدة تفتح آفاقا واسعة للنمو",
    "حققت الشركة تحسنا ملحوظا في هوامش الربحية خلال العام الحالي",
]
_NEGATIVE_EXAMPLES = [
    "تراجعت أرباح الشركة بشكل حاد خلال الربع الحالي مقارنة بالعام الماضي",
    "أعلنت الشركة عن خسائر غير متوقعة أثرت على ثقة المستثمرين",
    "هبط سهم الشركة بشكل حاد بعد إعلان نتائج مخيبة للآمال",
    "خفضت وكالة تصنيف ائتماني تصنيف الشركة مع نظرة مستقبلية سلبية",
    "واجهت الشركة تحديات تشغيلية أثرت سلبا على أدائها المالي",
    "أعلنت الشركة عن تأجيل مشروع توسعي بسبب ضغوط السيولة",
    "سجلت إيرادات الشركة تراجعا ملحوظا نتيجة ضعف الطلب",
    "استقال عدد من كبار المسؤولين التنفيذيين وسط أزمة داخلية بالشركة",
    "حذر المحللون من استمرار الضغوط على أرباح الشركة خلال الفترة المقبلة",
    "أعلنت الشركة عن دعوى قضائية قد تؤثر على مركزها المالي",
    "تكبدت الشركة خسائر إضافية بسبب ارتفاع تكاليف التشغيل",
    "انخفضت حصة الشركة السوقية أمام منافسة شرسة من شركات أخرى",
    "أغلق سهم الشركة على تراجع حاد وسط موجة بيع مكثفة",
    "أعلنت الشركة عن خطة لخفض التكاليف وسط تراجع الأداء",
    "واجهت الشركة صعوبات في سداد التزاماتها المالية في موعدها",
    "خفضت الشركة توقعاتها للأرباح للعام الحالي بشكل ملحوظ",
    "تراجعت ثقة المستثمرين في الشركة بعد تسريبات عن مشاكل تشغيلية",
]
_NEUTRAL_EXAMPLES = [
    "عقدت الشركة اجتماعا للجمعية العمومية لمناقشة جدول الأعمال السنوي",
    "أعلنت الشركة عن موعد الإفصاح عن نتائجها المالية للربع القادم",
    "استقرت أسهم الشركة دون تغيير يذكر في تعاملات اليوم",
    "أصدرت الشركة بيانا توضيحيا حول أنشطتها الاعتيادية",
    "عينت الشركة عضوا جديدا في مجلس الإدارة خلفا لعضو منتهية ولايته",
    "واصلت الشركة أنشطتها التشغيلية المعتادة دون مستجدات تذكر",
    "نشرت الشركة تقريرها السنوي المعتاد للمساهمين",
    "أكدت الشركة التزامها بخطتها الاستراتيجية المعلنة سابقا",
    "شاركت الشركة في مؤتمر قطاعي دون الإعلان عن مستجدات جوهرية",
    "حافظت أسهم الشركة على مستوياتها السعرية في تداولات هادئة",
    "أعلنت الشركة عن تحديث روتيني في بيانات الإفصاح",
    "عقدت إدارة الشركة اجتماعا دوريا لمراجعة الأداء التشغيلي",
    "لم تشهد أسهم الشركة تغيرات ملحوظة رغم تداولات نشطة نسبيا",
    "جددت الشركة عقد التدقيق الخارجي مع نفس مكتب المحاسبة",
    "أصدرت الشركة توضيحا للهيئة العامة للرقابة المالية دون مستجدات",
    "استمرت الشركة في تنفيذ خطتها المعلنة دون تغييرات جوهرية",
]

MANUAL_BOOTSTRAP_EXAMPLES = (
    [{"text": t, "label": POSITIVE} for t in _POSITIVE_EXAMPLES]
    + [{"text": t, "label": NEGATIVE} for t in _NEGATIVE_EXAMPLES]
    + [{"text": t, "label": NEUTRAL} for t in _NEUTRAL_EXAMPLES]
)


def compute_class_weights(labels: np.ndarray, n_classes: int = None) -> np.ndarray:
    """w_c = N_total / (n_classes * N_c) (spec 4.3). Classes absent from
    `labels` get a denominator of 1 instead of 0 (avoids a div-by-zero;
    an absent class contributes nothing to training anyway).
    """
    n_classes = n_classes or config.SENTIMENT_N_CLASSES
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)
    counts_safe = np.where(counts == 0, 1, counts)
    n_total = len(labels)
    return n_total / (n_classes * counts_safe)


def build_training_dataset(tickers=None) -> pd.DataFrame:
    """Combine real distant-supervision examples with the manual bootstrap
    set. Manual examples get config.SENTIMENT_MANUAL_EXAMPLE_WEIGHT
    (spec: 2.0, "high-confidence"); distant-supervision examples get their
    source-credibility weight. Returns columns: text, label, weight, date,
    ticker, source.
    """
    distant = build_distant_supervision_dataset(tickers)
    manual = pd.DataFrame(MANUAL_BOOTSTRAP_EXAMPLES)
    manual["weight"] = config.SENTIMENT_MANUAL_EXAMPLE_WEIGHT
    manual["source"] = "manual_synthetic"
    manual["date"] = pd.NaT
    manual["ticker"] = None

    combined = pd.concat([distant, manual], ignore_index=True, sort=False)
    logger.info(
        "build_training_dataset: %d distant-supervision + %d manual = %d total examples",
        len(distant), len(manual), len(combined),
    )
    return combined


def export_training_csv(df: pd.DataFrame, path: str) -> str:
    """Write the training dataset to CSV for upload to Kaggle as a
    Dataset (the Kaggle notebook this project ships reads exactly this
    file/column layout: text, label, weight, date, ticker, source).
    Re-runnable any time -- as more real news accumulates via Layer 1's
    scraper, re-export to get a fresher, larger training set.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("export_training_csv: wrote %d rows to %s", len(df), path)
    return path
