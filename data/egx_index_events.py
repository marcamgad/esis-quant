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

KNOWN GAPS. January 2018, Jan/Jul 2020, January 2022, full July 2025 and the
additions of July 2021 / removals of January 2023 were not recoverable from free
sources. GB Auto (AUTO) and QNB Al Ahli (QNBE) have no price history in this
system, and Orascom Construction's history starts after its 2017 removal, so
those events cannot be measured.
"""

# (announced, effective, added, removed, source, complete)
REVIEWS = [
    ("2017-07-26", "2017-08-01",
     ["AMOC", "EGCH", "ADIB", "EFIC"],
     ["CIEB", "JUFO", "AUTO", "ORAS"],
     "Youm7 2017-07-26", True),
    ("2021-07-28", "2021-08-01",
     [],
     ["OCDI", "EMFD", "OFH", "OIH"],
     "Youm7 2021-07-28 (named the four most notable of 7 removals; additions not named)", False),
    ("2022-07-28", "2022-08-01",
     ["JUFO", "QNBE", "ALCN"],
     ["ORHD", "ADIB", "MTIE"],
     "Hapi Journal 2022-07-28", True),
    ("2023-01-30", "2023-02-01",
     ["ADIB", "EFID", "TALM"],
     [],
     "Youm7 2023-01-30 (additions only)", False),
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
     [],
     ["ALCN", "PHAR", "EFID"],
     "Youm7 headline 2025-07-29 (removals named in headline only)", False),
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
