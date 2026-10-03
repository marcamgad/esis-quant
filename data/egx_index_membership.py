"""
EGX index membership (EGX30 / EGX70 EWI).

WHY THIS FILE EXISTS. `stocks.egx30_member` and `stocks.egx70_member` have been
columns in the schema since the beginning and were NEVER populated. Any "pick
the best EGX70 stock" question was therefore unanswerable, and the tempting
workaround (rank by liquidity, call the top 70 "EGX70") is not just imprecise
but INVERTED: EGX70 EWI is defined as the 70 most active companies AFTER
EXCLUDING the 30 constituents of EGX30. A liquidity-ranked top-70 would hand
back large caps when the index means mid caps.

SOURCE. Constituent names from Investing.com's EGX70 component list, refreshed
2026-09-04 (i.e. AFTER the 2026-09-01 semi-annual review took effect);
ticker<->name mapping from stockanalysis.com's EGX listing. Names are matched to
tickers explicitly below rather than fuzzily, because a silent mis-match here
would mislabel a stock's index membership with no visible error. Anything that
could not be matched with confidence is in UNMATCHED and is deliberately NOT
guessed.

STALENESS. EGX rebalances semi-annually (March and September). The 2026-09-01
review has been APPLIED to the list below -- it moved 12 names, most notably
OIH out of EGX30 and into EGX70. The next review is 2027-03-01, after which this
file is wrong. `assert_fresh()` exists so that expiry raises instead of being
silently believed: the previous version of this file sat one week past its own
expiry date while being used to answer index questions, because the date was a
comment rather than a check.
"""

import datetime
import logging

logger = logging.getLogger("esis_quant.index_membership")

# The review reflected in the list below.
REVIEW_APPLIED = "2026-09-01"
# When this list goes stale. Past this date the membership below is WRONG.
NEXT_REVIEW = "2027-03-01"

# EGX70 EWI constituents as of the 2026-09-01 review. name -> ticker.
EGX70_EWI = {
    "Egyptian Tourism Resorts": "EGTS",
    "Electro Cable": "ELEC",                        # added 2026-09-01
    "El Kahera Housing": "ELKA",                    # added 2026-09-01
    "Egyptian Media Production City": "MPRC",
    "SODIC": "OCDI",
    "Aspire Capital Holding for Financial Investments": "ASPI",
    "South Valley Cement": "SVCE",
    "El Sewedy Electric": "SWDY",
    "El Saeed Contracting & RE Inv": "UEGC",
    "ASEC Co for Mining": "ASCM",
    "Cairo Oils & Soap": "COSG",
    "Canal Shipping Agencies": "CSAG",
    "DAPH": "DAPH",
    "Egyptians Housing Develop": "EHDR",
    "El Shams": "ELSH",
    "Industrial Engineering": "ENGC",
    "Egyptian Transport & Commercial Services": "ETRS",
    "Housing&Develop Bank": "HDBK",
    "Ismailia Misr Poultry": "ISMA",
    "El Nasr Clothing & Textiles": "KABO",
    "Lecico Egypt": "LCSW",
    "Madinet Nasr for Housing and Development": "MASR",
    "Mansourah Poultry": "MPCO",
    "Nasr Co. for Civil Works": "NCCW",
    "Oriental Weavers": "ORWE",                     # added 2026-09-01 (left EGX30)
    "Cairo Poultry": "POUL",
    "Ceramic&Porcelain": "PRCL",
    "Universal Co Packaging": "UNIP",
    "Extracted Oils": "ZEOT",
    "Zahraa Maadi": "ZMID",
    "Misr Beni Suef Cement": "MBSC",                # added 2026-09-01
    "Credit Agricole Egypt": "CIEB",
    "Nile Pharma": "NIPH",
    "EIPICO": "PHAR",
    "Kima": "EGCH",                                 # added 2026-09-01 (left EGX30)
    "Sinai Cement": "SCEM",
    "EDBE": "EXPA",
    "Arabia Invest": "AIHC",
    "Amer Group Holding": "AMER",
    "Arab Moltaka Invest": "AMIA",
    "Misr National Steel": "ATQA",
    "Dice Sport&Casual Wear": "DSCW",
    "GPI for Urban Growth": "GPIM",
    "New lsmailia for Urban Dev": "IDRE",
    "Medical Packaging": "MEPA",
    "Memphis Pharma": "MPCI",
    "El Obour RE Inv": "OBRI",
    "Orascom Invest": "OIH",                        # added 2026-09-01 (DEMOTED from EGX30)
    "Sharm Dreams Co.": "SDTI",
    "Arabian Cement Co SAE": "ARCC",                # added 2026-09-01 (left EGX30)
    "Arab Developers Holding": "ARAB",
    "Sabaa Intl Pharma and Chemical Industries": "SIPC",
    "MM Group": "MTIE",
    "The Egyptian Modern Education": "MOED",
    "Al Tawfeek Leasing": "ATLC",
    "Contact Financial Holding": "CNFN",
    "Iron Steel for Mines Quarries": "ISMQ",
    "Gadwa for Industrial Development": "GDWA",     # added 2026-09-01
    "Pioneer Properties for Urban Development": "PRDC",  # added 2026-09-01
    "Macro Pharmaceutical": "MCRO",
    "TAQA Arabia": "TAQA",
    "Act Financial": "ACTF",
    "U Consumer Finance": "VALU",
    "Bonyan Dev Trade": "BONY",                     # added 2026-09-01
    "Gourmet Egyptcom Foods": "GOUR",               # added 2026-09-01
}

# Present in the source list but NOT mapped -- no confident ticker match.
# Guessing these would silently mislabel index membership.
UNMATCHED = [
    "Atlas Land Reclamation",       # source page name does not match the review's
                                    # "Atlas For Investment and Food Industries" (ALRA)
                                    # closely enough to assert; two plausible companies.
    "Industrial Engineer Enterprises",  # distinct from ENGC "Industrial Engineering"
    "ICFC",                         # abbreviation not resolvable from the EGX listing
    "Orascom Financial Holding",    # no distinct code in the EGX listing (not OIH)
    "Arabia for Investment and",    # truncated at source
]

# REMOVED by the 2026-09-01 review -- kept so that a stale cached answer that
# names one of these is recognisable as stale rather than merely unfamiliar.
REMOVED_2026_09 = {
    "IFAP": "International Agricultural Products",
    "BIOC": "Glaxo Smith Kline",
    "AFMC": "Alexandria Flour Mills",
    "RACC": "Raya Customer Experience",
    "SKPC": "Sidi Kerir Petrochemicals -- promoted to EGX30",
    "MFPC": "Misr Fertilizers (Mopco) -- promoted to EGX30",
    "TALM": "Taaleem Management Services",
    "KRDI": "Al Khair River for Development Agricultural",
    "AFDI": "El Ahli Investment and Development",
    "TANM": "Tanmiya for Real Estate Investment",
    "ALCN": "Alexandria Containers -- promoted to EGX30",
    "ECAP": "El Ezz Porcelain (Gemma)",
}

# EGX30 changes at the same review, for completeness.
EGX30_ADDED_2026_09 = ["MFPC", "ALCN", "CLHO", "SKPC"]
EGX30_REMOVED_2026_09 = ["ARCC", "EGCH", "ORWE", "OIH"]


def assert_fresh(today: str = None, raise_on_stale: bool = True) -> bool:
    """True while this list is current; raises (or warns) once it is not.

    The previous version of this module recorded its expiry in a docstring. It
    then sat a week past that date and was used to answer index questions, and
    nothing anywhere said so. A date that matters has to be checked, not
    written down.
    """
    today = today or datetime.date.today().isoformat()
    if today < NEXT_REVIEW:
        return True
    msg = (f"EGX index membership is STALE: list reflects the {REVIEW_APPLIED} review, "
           f"but the {NEXT_REVIEW} review has taken effect. Re-scrape constituents "
           f"before using egx70_tickers().")
    if raise_on_stale:
        raise RuntimeError(msg)
    logger.warning(msg)
    return False


def egx70_tickers(check: bool = True):
    """Tickers in EGX70 EWI as of the most recently applied review."""
    if check:
        assert_fresh()
    return sorted(set(EGX70_EWI.values()))
