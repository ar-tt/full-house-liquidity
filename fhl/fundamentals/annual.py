"""Turn a company's raw XBRL facts into one row per fiscal year, as known on a date.

Point in time: a number only counts if it was FILED on or before the as-of
date. If a later filing restated a year (still before the as-of date), the
restated number is used, because that is what an investor could have seen.

Each fiscal year also records when it first became public (the earliest
filing date of its annual report). Nothing may use a year before that date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class FiscalYear:
    end: date
    available: date                 # first filing date of this year's annual report
    values: dict = field(default_factory=dict)
    filed: dict = field(default_factory=dict)   # concept -> filing date of the number used

    def get(self, concept: str):
        return self.values.get(concept)


@dataclass
class Financials:
    name: str
    currency: str | None
    years: list                     # FiscalYear, oldest first
    latest_shares: tuple | None     # (filed date, diluted shares) from the newest filing, any form

    def series(self, concept: str) -> list:
        return [(fy.end, fy.get(concept)) for fy in self.years]


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _split_tag(tag: str) -> tuple[str, str]:
    ns, _, name = tag.partition(":")
    return ns, name


def _facts(companyfacts: dict, tag: str, unit: str) -> list:
    ns, name = _split_tag(tag)
    return companyfacts.get("facts", {}).get(ns, {}).get(name, {}).get("units", {}).get(unit, [])


def reporting_currency(companyfacts: dict, fcfg: dict, as_of: date) -> str | None:
    """The currency the company reports in: whichever currency unit carries
    the most annual revenue / net-income facts filed by as_of."""
    counts: dict[str, int] = {}
    for concept in ("revenue", "net_income"):
        for alt in fcfg["concepts"][concept]["alternatives"]:
            ns, name = _split_tag(alt[0])
            units = companyfacts.get("facts", {}).get(ns, {}).get(name, {}).get("units", {})
            for unit, facts in units.items():
                if len(unit) == 3 and unit.isupper():
                    n = sum(1 for f in facts if f.get("form") in fcfg["annual_forms"] and _d(f["filed"]) <= as_of)
                    counts[unit] = counts.get(unit, 0) + n
    counts = {k: v for k, v in counts.items() if v}
    return max(counts, key=counts.get) if counts else None


def _unit_for(concept_cfg: dict, currency: str) -> str:
    if concept_cfg.get("per_share"):
        return f"{currency}/shares"
    if concept_cfg.get("shares"):
        return "shares"
    return currency


def _annual_values(companyfacts, tag, unit, fcfg, as_of, instant) -> dict:
    """{period end: (filed, value)} for annual facts filed by as_of, keeping
    the most recently filed version of each period (restatements)."""
    lo, hi = fcfg["annual_days"]
    out: dict[date, tuple] = {}
    for f in _facts(companyfacts, tag, unit):
        if f.get("form") not in fcfg["annual_forms"]:
            continue
        filed = _d(f["filed"])
        if filed > as_of:
            continue
        end = _d(f["end"])
        if instant:
            if "start" in f:
                continue
        else:
            if "start" not in f or not lo <= (end - _d(f["start"])).days <= hi:
                continue
        if end not in out or filed >= out[end][0]:
            out[end] = (filed, float(f["val"]))
    return out


def _near(values: dict, end: date, tol: int):
    best = None
    for e, v in values.items():
        gap = abs((e - end).days)
        if gap <= tol and (best is None or gap < best[0]):
            best = (gap, v)
    return best[1] if best else None


def build_financials(companyfacts: dict, fcfg: dict, as_of: date) -> Financials:
    name = companyfacts.get("entityName", "")
    cur = reporting_currency(companyfacts, fcfg, as_of)
    if cur is None:
        return Financials(name, None, [], None)
    tol = fcfg["year_end_tolerance_days"]
    concepts = fcfg["concepts"]

    cache: dict[tuple, dict] = {}

    def values(tag, ccfg):
        key = (tag, ccfg.get("instant", False))
        if key not in cache:
            cache[key] = _annual_values(companyfacts, tag, _unit_for(ccfg, cur), fcfg, as_of,
                                        ccfg.get("instant", False))
        return cache[key]

    # Fiscal year ends = ends of annual net-income (or revenue) periods
    ends: dict[date, date] = {}
    for concept in ("net_income", "revenue"):
        for alt in concepts[concept]["alternatives"]:
            for end, (filed, _) in values(alt[0], concepts[concept]).items():
                if not any(abs((end - e).days) <= tol for e in ends):
                    ends[end] = filed
    # "available" = the earliest filing that reported any flow for that year
    for end in ends:
        for concept in ("net_income", "revenue"):
            for alt in concepts[concept]["alternatives"]:
                for f in _facts(companyfacts, alt[0], _unit_for(concepts[concept], cur)):
                    if (f.get("form") in fcfg["annual_forms"] and "start" in f
                            and abs((_d(f["end"]) - end).days) <= tol and _d(f["filed"]) <= as_of):
                        ends[end] = min(ends[end], _d(f["filed"]))

    years = []
    for end in sorted(ends):
        vals, filed = {}, {}
        for concept, ccfg in concepts.items():
            v = None
            for alt in ccfg["alternatives"]:
                first = _near(values(alt[0], ccfg), end, tol)
                if first is None:
                    continue
                v = first[1] + sum(x[1] for x in (_near(values(t, ccfg), end, tol) for t in alt[1:]) if x)
                filed[concept] = first[0]
                break
            if v is None and concept in fcfg["default_zero"]:
                v = 0.0
            vals[concept] = v
        years.append(FiscalYear(end, ends[end], vals, filed))

    # Newest share count from ANY filing (10-Q too), for today's market value.
    # Alternatives are tried in order; one whose newest figure is stale (the
    # company stopped using that tag) is skipped in favour of the next.
    latest_shares, fallback = None, None
    for alt in concepts["diluted_shares"]["alternatives"]:
        newest = None
        for f in _facts(companyfacts, alt[0], "shares"):
            filed = _d(f["filed"])
            if filed <= as_of and (newest is None or (_d(f["end"]), filed) > newest[0]):
                newest = ((_d(f["end"]), filed), float(f["val"]))
        if newest is None:
            continue
        fallback = fallback or newest
        if (as_of - newest[0][0]).days <= fcfg["latest_shares_max_age_days"]:
            latest_shares = (newest[0][1], newest[1])
            break
    if latest_shares is None and fallback:
        latest_shares = (fallback[0][1], fallback[1])
    return Financials(name, cur, years, latest_shares)
