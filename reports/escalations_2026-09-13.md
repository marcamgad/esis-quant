# ESIS-QUANT escalation pack — 2026-09-13

Paste each prompt into a chat assistant (ChatGPT, Claude, Gemini). Copy the
JSON it returns into a file, then run:

```
py -3.11 -m ops.escalation ingest answers.json
```

Answers without a source URL are rejected on ingest. UNKNOWN is fine.

## Security identity

### 1. KASABF

```
You are verifying reference data for an Egyptian Exchange (EGX) security: ticker KASABF, ISIN EGS696Z1C017, listed as "Certificates Of Odin Egyptian Equity Investment Fund-KASAB".

Answer ONLY from sources you can cite. For every field give a source URL.
If you cannot verify a field from a source, set it to "UNKNOWN". UNKNOWN is a
correct and acceptable answer; a guess is not. Do not infer from the name.

Return exactly this JSON and nothing else:
{
  "ticker": "KASABF",
  "security_type": "ordinary | fund | etf | rights | preferred | spac | fx_class | UNKNOWN",
  "security_type_source": "https://...",
  "listing_date": "YYYY-MM-DD or UNKNOWN",
  "listing_date_source": "https://...",
  "pricing": "continuous | nav | UNKNOWN",
  "pricing_source": "https://...",
  "holds_basket_of_equities": "yes | no | UNKNOWN",
  "notes": "one sentence, or empty"
}
```

## Price disagreements (split / share-class triage)

### 2. ORAS

```
EGX ticker ORAS shows 858.0000 EGP on stockanalysis.com and 71.0500 on Yahoo Finance on 2026-09-10
(ratio 12.076). Explain the difference using ONLY sourced facts.

Which of these is the cause?
  split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN

Give a source URL for the cause. If it is a split or bonus issue, give the ratio
and the ex-date. If you cannot verify, answer UNKNOWN -- that is acceptable.

Return exactly this JSON and nothing else:
{
  "ticker": "ORAS",
  "cause": "split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN",
  "cause_source": "https://...",
  "ratio": "e.g. 10:1, or UNKNOWN",
  "ex_date": "YYYY-MM-DD or UNKNOWN",
  "which_source_is_correct": "stockanalysis.com | Yahoo Finance | both (different instruments) | UNKNOWN",
  "notes": "one sentence, or empty"
}
```

### 3. SEIGA

```
EGX ticker SEIGA shows 0.9500 EGP on stockanalysis.com and 63.2900 on Yahoo Finance on 2026-09-10
(ratio 0.015). Explain the difference using ONLY sourced facts.

Which of these is the cause?
  split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN

Give a source URL for the cause. If it is a split or bonus issue, give the ratio
and the ex-date. If you cannot verify, answer UNKNOWN -- that is acceptable.

Return exactly this JSON and nothing else:
{
  "ticker": "SEIGA",
  "cause": "split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN",
  "cause_source": "https://...",
  "ratio": "e.g. 10:1, or UNKNOWN",
  "ex_date": "YYYY-MM-DD or UNKNOWN",
  "which_source_is_correct": "stockanalysis.com | Yahoo Finance | both (different instruments) | UNKNOWN",
  "notes": "one sentence, or empty"
}
```

### 4. DCCC

```
EGX ticker DCCC shows 1.8180 EGP on stockanalysis.com and 1.7000 on Yahoo Finance on 2026-09-10
(ratio 1.069). Explain the difference using ONLY sourced facts.

Which of these is the cause?
  split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN

Give a source URL for the cause. If it is a split or bonus issue, give the ratio
and the ex-date. If you cannot verify, answer UNKNOWN -- that is acceptable.

Return exactly this JSON and nothing else:
{
  "ticker": "DCCC",
  "cause": "split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN",
  "cause_source": "https://...",
  "ratio": "e.g. 10:1, or UNKNOWN",
  "ex_date": "YYYY-MM-DD or UNKNOWN",
  "which_source_is_correct": "stockanalysis.com | Yahoo Finance | both (different instruments) | UNKNOWN",
  "notes": "one sentence, or empty"
}
```

## Needs your broker statement (not a question for an AI)

- sell 120 ISMQ booked at 9.3500 on 2026-09-04 -- send the real fill price
- sell 84 SNFC booked at 10.4000 on 2026-09-04 -- send the real fill price
- sell 3 EFIC booked at 196.5900 on 2026-09-04 -- send the real fill price
- sell 21 OLFI booked at 22.2500 on 2026-09-04 -- send the real fill price
- sell 137 ARAB booked at 0.2600 on 2026-09-11 -- send the real fill price
