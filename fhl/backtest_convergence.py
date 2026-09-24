"""BACKTEST 4: do higher Convergence scores come before better returns?

THE STOCK LIST, chosen as it looked at the time: the biggest US companies by
revenue in 2013 (SEC data for every filer), financials removed. It includes
companies later taken over or delisted, so it isn't a list of winners.

EACH APRIL 1 from 2015 to 2025, every company is scored by the live
Convergence Engine using only filings public by that day and prices up to
that day. Then we measure its total return over the next 12 months and
compare the top, middle and bottom thirds by score.

THE BIAS: free prices (Yahoo) mostly lack companies that no longer trade.
Those companies drop out of the results, and they are disproportionately
the ones that struggled (or were bought at a premium). The report counts
them. Treat every result here as an upper bound on how well the scores work.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .fundamentals.edgar import EdgarFundamentals, sec_get
from .history import CurveHistory
from .pillars.base import Context
from .pillars.convergence import ConvergenceEngine
from .prices import YahooChartPrices
from .securities import Security


def sector_for(sic: int | None, rows: list) -> str | None:
    if sic is None:
        return None
    for lo, hi, sector in rows:
        if lo <= sic <= hi:
            return sector
    return None


SUFFIXES = {"inc", "corp", "corporation", "co", "company", "holdings", "holding", "group", "the", "plc",
            "ltd", "limited", "lp", "llc", "sa", "nv", "de", "new", "incorporated"}


def name_key(name: str) -> str:
    """'WALT DISNEY CO/' and 'Walt Disney Co' -> 'waltdisney'; used to find a
    company's ticker after it reorganized under a new SEC number."""
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in name).split()
    return "".join(w for w in words if w not in SUFFIXES)


def universe(cfg: dict, offline: bool = False) -> list[dict]:
    """The biggest US non-financial companies with public shares, by revenue
    in the universe year, with today's ticker where one can be found."""
    b = cfg["backtest"]["convergence"]
    year = b["universe_year"]
    cache = Path(cfg["_path"]).parent / cfg["data"]["cache_dir"] / "sec"
    revenue: dict[int, dict] = {}
    for tag in b["revenue_tags"]:
        url = b["frames_url"].format(tag=tag, year=year)
        frame = sec_get(cfg, url, cache / f"frame_{tag}_{year}.json.gz", 3650, offline) or {"data": []}
        for row in frame["data"]:
            if not str(row.get("loc", "")).startswith("US"):
                continue
            if row["cik"] not in revenue or row["val"] > revenue[row["cik"]]["revenue"]:
                revenue[row["cik"]] = {"cik": row["cik"], "name": row["entityName"], "revenue": row["val"]}
    shares_frame = sec_get(cfg, b["shares_url"].format(year=year + 1),
                           cache / f"frame_shares_{year + 1}Q1.json.gz", 3650, offline) or {"data": []}
    shares = {r["cik"]: r["val"] for r in shares_frame["data"]}
    listing = sec_get(cfg, b["company_list_url"], cache / "company_tickers.json.gz", 7, offline) or {}
    by_cik, by_name = {}, {}
    for v in listing.values():
        by_cik.setdefault(v["cik_str"], v["ticker"])
        by_name.setdefault(name_key(v["title"]), v["ticker"])
    out = []
    for c in sorted(revenue.values(), key=lambda r: -r["revenue"])[:b["candidates"]]:
        if shares.get(c["cik"], 0) <= b["min_shares"]:
            continue
        sub = sec_get(cfg, b["submissions_url"].format(cik=c["cik"]),
                      cache / "submissions" / f"CIK{c['cik']:010d}.json.gz", 30, offline) or {}
        sic = int(sub["sic"]) if str(sub.get("sic", "")).isdigit() else None
        ticker = by_cik.get(c["cik"]) or (sub.get("tickers") or [None])[0] or by_name.get(name_key(c["name"]))
        c.update(sic=sic, sector=sector_for(sic, b["sic_sectors"]), ticker=ticker)
        if c["sector"] and c["sector"] != "Financials":
            out.append(c)
    return out[:b["universe_size"]]


def securities(companies: list[dict]) -> dict:
    out = {}
    for c in companies:
        t = c["ticker"] or f"CIK{c['cik']}"
        out[t] = Security(t, c["name"], "stock", "equity", c["name"], {c["sector"]: 1.0}, {"US": 1.0},
                          bool(c["ticker"]), c["cik"])
    return out


def forward_return(prices, ticker: str, d0: date, days: int):
    """Total return over the next `days` calendar days, or None if the price
    history doesn't cover the whole period."""
    try:
        bars = prices.bars(ticker, d0 + timedelta(days=days))
    except LookupError:
        return None
    start = [b for b in bars if b.date <= d0]
    after = [b for b in bars if b.date > d0]
    if not start or not after or after[-1].date < d0 + timedelta(days=days - 7):
        return None
    level, prev = 1.0, start[-1]
    for b in after:
        level *= (b.close + b.dividend) / prev.close
        prev = b
    return level - 1


@dataclass
class Scores:
    dates: list
    rows: list        # dicts: date, ticker, score, quality, dividend, value, mos, status, fwd


def run(cfg: dict, offline: bool = False, progress=None) -> tuple[list, Scores]:
    b = cfg["backtest"]["convergence"]
    companies = universe(cfg, offline)
    secs = securities(companies)
    dates = [date(y, b["rebalance"]["month"], b["rebalance"]["day"]) for y in range(b["first_year"], b["last_year"] + 1)]
    fundamentals = EdgarFundamentals(cfg, offline=offline, keep_raw=False)
    for k, s in enumerate(secs.values()):
        fundamentals.prebuild(s, dates)
        if progress:
            progress(k + 1, len(secs))
    prices = YahooChartPrices(cfg, offline=offline)
    curves = CurveHistory(cfg, offline=offline)
    spec = next(p for p in cfg["pillars"] if p["name"] == "convergence")
    rows = []
    for d in dates:
        ctx = Context(cfg, secs, prices, d, fundamentals=fundamentals, curve=curves.curve_on(d))
        eng = ConvergenceEngine(spec, ctx)
        for t, s in secs.items():
            a = eng.analyze(t)
            rows.append({"date": d, "ticker": t, "status": a.status, "score": a.score,
                         "quality": a.sources.get("quality"), "dividend": a.sources.get("dividend"),
                         "value": a.sources.get("value"), "mos": a.facts.get("margin_of_safety"),
                         "fwd": forward_return(prices, t, d, b["forward_days"]) if s.has_prices else None})
    return companies, Scores(dates, rows)


def group_returns(rows: list, key: str, groups: int) -> dict:
    """Average next-12-month return of each group (top/middle/bottom) by `key`,
    ranked within each date, plus the rank correlation (information coefficient)."""
    by_group = {g: [] for g in range(groups)}
    ics = []
    for d in sorted({r["date"] for r in rows}):
        rs = [r for r in rows if r["date"] == d and r[key] is not None and r["fwd"] is not None]
        if len(rs) < groups * 3:
            continue
        rs.sort(key=lambda r: r[key], reverse=True)
        for g, chunk in enumerate(np.array_split(np.arange(len(rs)), groups)):
            by_group[g].append(np.mean([rs[i]["fwd"] - np.mean([x["fwd"] for x in rs]) for i in chunk]))
        a = np.argsort(np.argsort([r[key] for r in rs]))
        f = np.argsort(np.argsort([r["fwd"] for r in rs]))
        ics.append(float(np.corrcoef(a, f)[0, 1]))
    return {"groups": {g: np.array(v) for g, v in by_group.items()}, "ic": np.array(ics)}


def report(cfg, args, table, explain):
    b = cfg["backtest"]["convergence"]
    print("BACKTEST 4: do higher Convergence scores come before better returns?   [BIASED: see below]")

    def progress(k, n):
        if k in (1, n) or k % 25 == 0:
            print(f"  reading filings: {k} of {n} companies", flush=True)
    companies, sc = run(cfg, offline=args.offline, progress=progress)
    rows = sc.rows
    no_prices = [c for c in companies if not c["ticker"]]
    scored = [r for r in rows if r["status"] == "scored"]
    usable = [r for r in scored if r["fwd"] is not None]
    lost = [r for r in scored if r["fwd"] is None]
    print(f"\n  Stock list: the {len(companies)} biggest US non-financial companies by {b['universe_year']} revenue "
          f"(SEC data), scored each April 1, {sc.dates[0].year}-{sc.dates[-1].year}")
    print(f"  Company-years scored: {len(scored):,} of {len(rows):,} (the rest lack enough filings or data)")
    print(f"  SURVIVOR BIAS: {len(no_prices)} of the {len(companies)} companies no longer trade under any ticker "
          f"(taken over, merged or delisted) and have no free prices; {len(lost)} scored company-years have no "
          f"complete next-12-month price history. All of these are left out, so results below flatter the scores.")

    names = {0: "Top third", 1: "Middle third", 2: "Bottom third"} if b["groups"] == 3 else {}
    print("\n  Next-12-month return versus the average company that year (equal weight):")
    out_rows = []
    for key, label in (("score", "Convergence score"), ("quality", "Quality source"),
                       ("dividend", "Dividend source"), ("value", "Value source")):
        g = group_returns(usable, key, b["groups"])
        top, bottom = g["groups"][0], g["groups"][b["groups"] - 1]
        out_rows.append([label] + [f"{g['groups'][k].mean():+.1%}" for k in range(b["groups"])] +
                        [f"{(top - bottom).mean():+.1%}", f"{np.mean(top > bottom):.0%} of {len(top)}",
                         f"{g['ic'].mean():+.2f}"])
    print(table(["Ranked by"] + [names.get(k, f"Group {k + 1}") for k in range(b["groups"])] +
                ["Top minus bottom", "Years top beat bottom", "Avg rank correlation"], out_rows))

    need = cfg["convergence"]["hard_rules"]["min_margin_of_safety"]
    passed = [r for r in usable if r["mos"] is not None and r["mos"] >= need]
    failed = [r for r in usable if r["mos"] is not None and r["mos"] < need]
    rel = lambda rs: np.mean([r["fwd"] - np.mean([x["fwd"] for x in usable if x["date"] == r["date"]]) for r in rs])
    print(f"\n  The {need:.0%} margin-of-safety rule (3C blend): "
          f"{len(passed)} company-years passed, next 12 months {rel(passed):+.1%} vs average; "
          f"{len(failed)} failed, {rel(failed):+.1%}" if passed and failed else
          f"\n  The {need:.0%} margin-of-safety rule: too few passes to judge ({len(passed)})")
    years = {}
    for r in passed:
        years[r["date"].year] = years.get(r["date"].year, 0) + 1
    print("  Passes by year: " + ", ".join(f"{y}: {n}" for y, n in sorted(years.items())) if years else "")
    explain(args, """
Each April the live Convergence Engine scores every company from filings
public by then, and we see how each third did over the next year compared
with the average company. 'Top minus bottom' is the payoff from favouring
high scores; the rank correlation runs from -1 to +1 (0 = no link; +0.05 is
already useful in real investing). With only about ten years, a few good or
bad years can decide the result. And because companies that stopped trading
are missing, the true payoff is probably lower than shown.""")
