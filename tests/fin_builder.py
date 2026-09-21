"""Build fake SEC 'company facts' and prices for tests, in the real format."""
from datetime import date, timedelta

import numpy as np

from fhl.prices import Bar

TAGS = {
    "revenue": "Revenues",
    "operating_income": "OperatingIncomeLoss",
    "pretax_income": "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "income_tax": "IncomeTaxExpenseBenefit",
    "net_income": "NetIncomeLoss",
    "eps_diluted": "EarningsPerShareDiluted",
    "diluted_shares": "WeightedAverageNumberOfDilutedSharesOutstanding",
    "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
    "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
    "depreciation_amortization": "DepreciationDepletionAndAmortization",
    "dividends_paid": "PaymentsOfDividendsCommonStock",
    "interest_expense": "InterestExpense",
    "total_assets": "Assets",
    "equity": "StockholdersEquity",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
    "total_debt": "LongTermDebt",
}
INSTANT = {"total_assets", "equity", "cash", "total_debt"}


def add_fact(cf, tag, unit, end, val, filed, start=None, form="10-K"):
    """Add one fact to a company-facts dict (as returned by companyfacts())."""
    f = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        f["start"] = start
    ns = cf.setdefault("facts", {}).setdefault("us-gaap", {})
    ns.setdefault(tag, {"units": {}})["units"].setdefault(unit, []).append(f)


def companyfacts(rows, currency="USD", name="TestCo"):
    """rows: one dict per fiscal year (calendar years) with 'year' and any concepts.
    Optional per row: 'filed' (default Feb 15 of the next year), 'form', 'skip' (concepts to omit)."""
    cf = {"entityName": name, "facts": {}}
    for r in rows:
        y = r["year"]
        filed = r.get("filed", f"{y + 1}-02-15")
        for concept, tag in TAGS.items():
            if concept not in r or concept in r.get("skip", ()):
                continue
            unit = {"eps_diluted": f"{currency}/shares", "diluted_shares": "shares"}.get(concept, currency)
            add_fact(cf, tag, unit, f"{y}-12-31", r[concept], filed,
                     None if concept in INSTANT else f"{y}-01-01", r.get("form", "10-K"))
    return cf


def compounder(years=range(2014, 2026), growth=0.08, margin=0.25, shares=100.0, payout=0.4,
               debt=500.0, split_on=None, split_ratio=2.0):
    """A steady, high-quality business. If split_on is a date, filings made
    before it report pre-split shares and EPS (as real filings do)."""
    rows = []
    for i, y in enumerate(years):
        rev = 1000 * (1 + growth) ** i
        op = rev * margin
        interest = 0.05 * debt
        pre = op - interest
        tax = 0.21 * pre
        ni = pre - tax
        da = capex = 0.04 * rev
        filed = f"{y + 1}-02-15"
        pre_split = split_on is not None and date.fromisoformat(filed) < split_on
        sh = shares / split_ratio if pre_split else shares
        rows.append(dict(year=y, revenue=rev, operating_income=op, pretax_income=pre, income_tax=tax,
                         net_income=ni, eps_diluted=ni / sh, diluted_shares=sh,
                         operating_cash_flow=ni + da, capex=capex, depreciation_amortization=da,
                         dividends_paid=payout * ni, interest_expense=interest,
                         total_assets=3000 + 100 * i, equity=2000 + 60 * i, cash=200.0, total_debt=debt))
    return rows


def weekdays(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def price_bars(level, start=date(2014, 1, 2), end=date(2026, 9, 18), seed=3, dividends=None,
               splits=None, level_fn=None):
    """Daily bars wobbling around `level` (or level_fn(date)), same wobble as
    market_bars(seed) so beta is about 1. dividends: {date: amount}; splits: {date: ratio}."""
    rng = np.random.default_rng(seed)
    days = weekdays(start, end)
    wobble = rng.normal(0, 0.01, len(days))
    out = []
    for d, w in zip(days, wobble):
        p = (level_fn(d) if level_fn else level) * (1 + w)
        out.append(Bar(d, p, p, p, p, 1e6, (dividends or {}).get(d, 0.0), (splits or {}).get(d, 1.0)))
    return out


def market_bars(seed=3, **kw):
    return price_bars(100.0, seed=seed, **kw)


def quarterly_dividends(amount_by_year: dict):
    """Four equal payments a year, on the first weekday of Mar/Jun/Sep/Dec."""
    out = {}
    for y, amt in amount_by_year.items():
        for m in (3, 6, 9, 12):
            d = date(y, m, 1)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            out[d] = amt / 4
    return out
