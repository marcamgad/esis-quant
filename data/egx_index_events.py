"""
EGX30 index review events: which companies entered and left, and when.

SOURCES. Collected 2026-09-13 from the exchange's own press releases (via
mondovisione.com) and Arabic financial press that reprinted the releases in full:
Youm7, Masrawy, Hapi Journal, EnterpriseAM, Ahram Gate. Each review below names
its source. Where an article gave only some names ("the most notable excluded
companies were..."), the review is marked partial and only the named companies
are listed -- never inferred.

TIMING -- WHY TWO DATES AND WHY IT MATTERS.
  announced  the date the results were published
  effective  the first session trading on the new constituents

Until the July 2026 rule change, EGX published results only 1-4 days before the
effective date ("reviews end in January, effective 1 February"). From September
2026 results come ~2 weeks ahead (16 Aug -> 1 Sep). So historically there was
little time to trade AHEAD of a change; the testable effect is price pressure
around the effective date and any reversal afterwards, not front-running.

KNOWN GAPS. Jan 2018, Jul 2018, Jan/Jul 2019, Jan/Jul 2020 and Jul 2015/2016 are
still missing (for 2018-2020 the Al Mal archive only carries broker FORECASTS,
which are not results). Jan 2015, Jan 2016 and Jan 2017 were recovered from the
Al Mal archive on 2026-10-03 (RECOVERED_2026_10_03_B).
On 2026-10-03 Jan 2021 and Jan 2022 were added and Jul 2021, Jan 2023 and Jul
2025 completed, each from article text read in full (RECOVERED_2026_10_03). A
search-engine summary of "Jul 2018" (Edita/SKPC/ISPH/QNBE/EKHO in; EGCH, ARCC,
ADIB, AUTO, AMER, ACAMD out) was REJECTED: EGCH, AUTO and ACAMD are still in the
published 31 Jan 2019 constituent list (adelonsi.com), so it cannot be right. GB Auto (AUTO) and QNB Al Ahli (QNBE) have no price history in this
system, and Orascom Construction's history starts after its 2017 removal, so
those events cannot be measured.
"""

# (announced, effective, added, removed, source, complete)
REVIEWS = [
    ("2015-01-28", "2015-02-01",
     ["ARCC", "BTFH", "ELSH", "EHDR"],
     [],
     "Al Mal 204037, 2015-01-28 (recovered 2026-10-03; additions that map to a ticker "
     "unambiguously; 'العربية للاستثمارات' and 'المصرية للدواجن' and all removals but GB Auto "
     "left out as ambiguous, GB Auto has no prices)", False),
    ("2016-01-27", "2016-02-01",
     ["EMFD", "SVCE", "UASG"],
     ["SKPC", "MCQE", "BTFH", "EKHO"],
     "Al Mal 267596, 2016-01-27, quoting the EGX statement (recovered 2026-10-03)", True),
    ("2017-01-24", "2017-02-01",
     ["ORWE", "JUFO", "SKPC"],
     ["EKHO", "EFID", "UEGC"],
     "Al Mal 320005, 2017-01-24 (recovered 2026-10-03; 4th removal 'العربية للاستثمارات "
     "والتنمية القابضة' has no prices)", False),
    ("2017-07-26", "2017-08-01",
     ["AMOC", "EGCH", "ADIB", "EFIC"],
     ["CIEB", "JUFO", "AUTO", "ORAS"],
     "Youm7 2017-07-26", True),
    ("2021-01-27", "2021-02-01",
     ["FWRY", "CICH", "MTIE", "ABUK", "AMOC"],
     ["IRON", "JUFO", "CIRA", "DSCW", "BTFH"],
     "Economy Plus 2021-01-27 (recovered 2026-10-03)", True),
    ("2021-07-28", "2021-08-01",
     ["CIRA", "MFPC", "SPMD", "RAYA", "RMDA", "EGTS"],
     ["OCDI", "EMFD", "OFH", "OIH", "CICH", "EXPA", "EFID"],
     "Youm7 2021-07-28 (4 removals) + Business Today Egypt 2021-07-28 (full lists, recovered 2026-10-03)", True),
    ("2022-01-31", "2022-02-01",
     ["EFIH", "ADIB", "ORAS", "HDBK"],
     ["PRDC", "RAYA", "SPMD", "EGTS", "GDWA", "ASPI"],
     "CNBC Arabia 93466 + Youm7 2022-01-31 (recovered 2026-10-03)", True),
    ("2022-07-28", "2022-08-01",
     ["JUFO", "QNBE", "ALCN"],
     ["ORHD", "ADIB", "MTIE"],
     "Hapi Journal 2022-07-28", True),
    ("2023-01-30", "2023-02-01",
     ["ADIB", "EFID", "TALM"],
     ["ALCN", "HDBK", "QNBE"],
     "Youm7 2023-01-30 (additions) + Hapi Journal 2023-01-30 table (removals, recovered 2026-10-03)", True),
    ("2023-07-30", "2023-08-01",
     ["BTFH", "BINV", "ALCN", "ORHD"],
     ["TALM", "ISPH", "CLHO", "RMDA"],
     "Masrawy 2023-07-30", True),
    ("2024-01-31", "2024-02-01",
     ["PHAR", "ISPH", "SUGR"],
     ["CIRA", "CIEB", "BINV"],
     "EGX release via mondovisione 2024-01-31", True),
    ("2024-07-30", "2024-08-01",
     ["EMFD", "BINV", "CIEB", "CLHO", "FAIT"],
     ["CCAP", "ISPH", "ORHD", "SUGR", "PHAR"],
     "EnterpriseAM 2024-07-31 citing the EGX release", True),
    ("2025-01-29", "2025-02-01",
     ["ORHD", "RMDA", "EGAL", "ISPH", "CCAP", "PHAR"],
     ["ESRS", "SWDY", "HELI", "BINV", "CLHO", "FAIT"],
     "Hapi Journal 2025-01-29 + Middle East Observer 2025-02-02", True),
    ("2025-07-29", "2025-08-01",
     ["RAYA", "MCQE", "ARCC"],
     ["ALCN", "PHAR", "EFID"],
     "Youm7 headline 2025-07-29 (removals) + Alborsa News 2025-07-29 (additions, recovered 2026-10-03)", True),
    ("2026-01-27", "2026-02-01",
     ["EFID", "EGCH", "HELI", "OIH"],
     ["SKPC", "MFPC", "CIEB", "MASR"],
     "EGX release (press_release_indices_feb_2026) via mondovisione", True),
    ("2026-08-16", "2026-09-01",
     ["MFPC", "ALCN", "CLHO", "SKPC"],
     ["ARCC", "EGCH", "ORWE", "OIH"],
     "EGX release 2026-08-16 (ent.news PDF)", True),
]


# ---------------------------------------------------------------------------
# EGX70 EWI. Only reviews with the COMPLETE official lists are recorded; the
# press usually published these long lists as images, so earlier reviews were
# not recoverable. Company names were mapped to tickers by ISIN where possible
# and hand-checked: automated name matching got four wrong (ICON -> EEII,
# Orascom Financial -> ACTF, El Ahli -> AIDC twice), corrected below.
# Unmappable names are omitted, not guessed: "The Egyptian Company for
# Construction Development Lift Slab" (2024) and "Egyptians For Investment &
# Urban Development" (2026).
#
# CROSS-MOVES ARE EXCLUDED IN events70(), not here. A stock promoted to EGX30 is
# listed as an EGX70 removal but is receiving the EGX30 inclusion flow, and an
# EGX30 demotion is listed as an EGX70 addition. Counting those would re-measure
# the EGX30 effect under a different label.
# ---------------------------------------------------------------------------
EGX70_REVIEWS = [
    ("2024-01-31", "2024-02-01",
     ["CIEB", "BINV", "IFAP", "COPR", "ECAP", "CIRA", "EGTS", "ISMA", "SDTI", "ENGC",
      "TAQA", "ASPI", "PRMH", "AMER", "GGCC", "UEGC", "ACAP"],
     ["PHAR", "CSAG", "IDRE", "ARCC", "BIOC", "MOED", "ACAMD", "AMIA", "OFH", "EGBE",
      "ISPH", "ASCM", "FAITA", "SPIN", "SUGR", "LCSW"],
     "EGX release via mondovisione 2024-01-31", True),
    ("2026-01-27", "2026-02-01",
     ["MASR", "IDRE", "DAPH", "CIEB", "ZEOT", "BIOC", "AFMC", "RACC", "SIPC", "MOED",
      "SKPC", "MFPC", "TALM", "AFDI", "SDTI", "VALU", "TANM", "NCCW", "UNIP", "NIPH"],
     ["EFID", "CERA", "ODIN", "BINV", "OLFI", "UBEE", "CLHO", "ELEC", "RREI", "UNIT",
      "ELKA", "CIRA", "ADPC", "EGCH", "SUGR", "MOIL", "PRDC", "MENA", "GGCC", "EGAS",
      "ACGC", "HELI"],
     "EGX release (press_release_indices_feb_2026) via mondovisione", True),
    ("2026-08-16", "2026-09-01",
     ["ALRA", "ARCC", "ELEC", "ORWE", "OIH", "ELKA", "GDWA", "BONY", "GOUR", "MBSC",
      "EGCH", "PRDC"],
     ["IFAP", "BIOC", "AFMC", "RACC", "SKPC", "ECAP", "MFPC", "TALM", "KRDI", "AFDI",
      "TANM", "ALCN"],
     "EGX release 2026-08-16 (ent.news PDF)", True),
]


def events70():
    """Pure EGX70 events: tickers that moved to or from EGX30 in the same review
    are dropped (see the block comment above)."""
    egx30 = {(ann, eff): set(a) | set(r) for ann, eff, a, r, _s, _c in REVIEWS}
    rows = []
    for ann, eff, added, removed, src, complete in EGX70_REVIEWS:
        cross = egx30.get((ann, eff), set())
        for t in added:
            if t not in cross:
                rows.append({"ticker": t, "announced": ann, "effective": eff, "side": +1,
                             "source": src, "complete": complete})
        for t in removed:
            if t not in cross:
                rows.append({"ticker": t, "announced": ann, "effective": eff, "side": -1,
                             "source": src, "complete": complete})
    return rows


# Events recovered on 2026-10-03, AFTER the pre-registered test had been run on
# the other 73. They are the out-of-sample replication set (see
# analysis/index_event_study.replication): (announced, ticker, side).
RECOVERED_2026_10_03 = frozenset(
    [("2021-01-27", t, +1) for t in ("FWRY", "CICH", "MTIE", "ABUK", "AMOC")]
    + [("2021-01-27", t, -1) for t in ("IRON", "JUFO", "CIRA", "DSCW", "BTFH")]
    + [("2021-07-28", t, +1) for t in ("CIRA", "MFPC", "SPMD", "RAYA", "RMDA", "EGTS")]
    + [("2021-07-28", t, -1) for t in ("CICH", "EXPA", "EFID")]
    + [("2022-01-31", t, +1) for t in ("EFIH", "ADIB", "ORAS", "HDBK")]
    + [("2022-01-31", t, -1) for t in ("PRDC", "RAYA", "SPMD", "EGTS", "GDWA", "ASPI")]
    + [("2023-01-30", t, -1) for t in ("ALCN", "HDBK", "QNBE")]
    + [("2025-07-29", t, +1) for t in ("RAYA", "MCQE", "ARCC")]
)

# Second batch, recovered later the same day from the Al Mal archive (2015-2017),
# AFTER batch A had failed to replicate. Reported separately and never used to
# overturn batch A's verdict (see analysis/index_event_study.replication).
RECOVERED_2026_10_03_B = frozenset(
    [("2015-01-28", t, +1) for t in ("ARCC", "BTFH", "ELSH", "EHDR")]
    + [("2016-01-27", t, +1) for t in ("EMFD", "SVCE", "UASG")]
    + [("2016-01-27", t, -1) for t in ("SKPC", "MCQE", "BTFH", "EKHO")]
    + [("2017-01-24", t, +1) for t in ("ORWE", "JUFO", "SKPC")]
    + [("2017-01-24", t, -1) for t in ("EKHO", "EFID", "UEGC")]
)


def events():
    """One row per (ticker, review): +1 added, -1 removed."""
    rows = []
    for ann, eff, added, removed, src, complete in REVIEWS:
        for t in added:
            rows.append({"ticker": t, "announced": ann, "effective": eff, "side": +1,
                         "source": src, "complete": complete})
        for t in removed:
            rows.append({"ticker": t, "announced": ann, "effective": eff, "side": -1,
                         "source": src, "complete": complete})
    return rows
