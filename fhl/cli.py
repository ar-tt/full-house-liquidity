"""Command line: python -m fhl <schedule|reserve|project> [options]

  schedule   print Laura's cash-flow rows
  reserve    price the operating reserve, show what it holds and how it runs down
  project    grow the portfolio to 2033 and compute the facility contribution
  range      check the portfolio's structure, or a trade, against every pillar
             e.g.  python -m fhl range --buy MSFT --amount 5000 --explain
  score      the Convergence Engine's full breakdown for one or more stocks
             e.g.  python -m fhl score KO MSFT --explain

Add --explain to any command for a plain-English note on how each number is made.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from .cashflows import Schedule
from .combine import decide
from .config import ConfigError, load_config
from .curves import fetch_fed_curve_file, load_fed_curve
from .fundamentals.edgar import EdgarFundamentals
from .pillars.convergence import format_analysis
from .pillars.base import Context, load_pillars
from .portfolio import Portfolio, Proposal, exposures
from .prices import YahooChartPrices
from .projection import facility_contribution, project
from .reserve import PRICING_METHODS, build_ladder, reserve_curve
from .securities import load_securities


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    cells = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    align = align or "l" + "r" * (len(headers) - 1)
    def fmt(r):
        return "  ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(r, widths, align))
    lines = [fmt(headers), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]]
    return "\n".join(lines)


def explain(args, text: str) -> None:
    if args.explain:
        print("\n  WHY: " + text.strip().replace("\n", "\n       ") + "\n")


def try_fed_curve(cfg, args, required: bool):
    if args.offline:
        if required:
            raise ConfigError("this pricing method needs the Fed curve; drop --offline")
        return None, "skipped (--offline)"
    try:
        return load_fed_curve(fetch_fed_curve_file(cfg)), None
    except Exception as e:  # network down, file format changed, ...
        if required:
            raise
        return None, f"unavailable ({e.__class__.__name__}: {e})"


# ---------------------------------------------------------------------------

def cmd_schedule(cfg, args):
    s = Schedule.from_config(cfg)
    print(table(["Date", "Amount", "Kind", "Note"],
                [[r.date, money(r.amount), r.kind, r.note] for r in s.rows], "lrll"))
    print(f"\nContributed: {money(s.total_contributed())}    Paid out: {money(s.total_paid_out())}")
    explain(args, "Each line is one row in config.yaml under cash_flows. Positive = Laura puts\n"
                  "money in; negative = a residency payment going out. Add, move or delete rows\n"
                  "there and every other calculation follows.")


def cmd_reserve(cfg, args):
    s = Schedule.from_config(cfg)
    rcfg = cfg["reserve"]
    carve = cfg["decision_dates"]["reserve_carve_out"]
    method = args.pricing or rcfg["pricing"]
    fed, fed_note = try_fed_curve(cfg, args, required=method != "flat")

    # 1. Cost under every pricing assumption, side by side
    print(f"RESERVE COST ON {carve}  (funds {len(s.of_kind(*rcfg['funds_kinds']))} payments, "
          f"{money(-sum(r.amount for r in s.of_kind(*rcfg['funds_kinds'])))} total)\n")
    rows = []
    for rate in rcfg["scenario_rates"]:
        lad = build_ladder(s, rcfg, carve, reserve_curve(cfg, "flat", rate=rate))
        rows.append([f"Flat {rate:.2%} ({rcfg['compounding']})", money(lad.cost)])
    for m, label in (("curve_spot", "If 2033 curve looks like latest Fed curve"),
                     ("curve_forward", "Market-implied 2033 cost (lockable now)")):
        if fed:
            lad = build_ladder(s, rcfg, carve, reserve_curve(cfg, m, fed_curve=fed))
            rows.append([label, money(lad.cost)])
        else:
            rows.append([label, fed_note])
    print(table(["Pricing assumption", "Reserve cost"], rows))
    if fed:
        print(f"\nFed curve date: {fed.as_of}. Latest zero yields: "
              + ", ".join(f"{t}y {fed.zero_rate(t):.2%}" for t in (1, 2, 5, 7, 10, 16)))
    explain(args, """
Reserve cost = what it costs on the carve-out date to buy one zero-coupon
Treasury per payment. A zero-coupon bond pays one lump sum at maturity and
nothing before, so a bond paying $50,000 on 1 Jan 2036 exactly covers that
payment. Its price today is $50,000 x discount factor. The first payment is
made the same day, so it costs the full $50,000 (no discount).
Flat rows use one rate for every bond (like your anchors). The two curve rows
use the Federal Reserve's published Treasury curve, where each maturity has
its own rate. 'Lockable now' is the 2033 cost you could fix today by buying
the bonds early, which is why building the ladder before 2033 removes
interest-rate risk from the co-sponsor quote.""")

    # 2. What the reserve is made of on the carve-out date
    curve = reserve_curve(cfg, method, rate=args.rate, fed_curve=fed)
    lad = build_ladder(s, rcfg, carve, curve)
    print(f"\nLADDER ON {carve}  — priced with: {curve.describe()}\n")
    print(table(["Pays on", "Bond", "Face", "Years", "Yield", "Cost", "Share"],
                [[r.payment.date,
                  "Cash (paid today)" if r.years_to_maturity == 0 else "Zero-coupon Treasury",
                  money(r.face), f"{r.years_to_maturity:.2f}", f"{r.locked_yield:.2%}",
                  money(r.cost), f"{r.cost / lad.cost:.1%}"] for r in lad.rungs], "llrrrrr"))
    print(f"\nTotal: {money(lad.cost)} buys {money(lad.total_face)} of payments")
    comp = lad.composition(rcfg["maturity_buckets"])
    print("By maturity: " + "   ".join(f"{k} {v:.0%}" for k, v in comp.items()))

    # 3. How it runs down
    print("\nRUN-DOWN AS PAYMENTS ARE MADE\n")
    buckets = [b["label"] for b in rcfg["maturity_buckets"]][1:]
    rows = []
    for r in lad.runoff(rcfg["maturity_buckets"]):
        rows.append([r.date, money(r.value_before), money(r.payment), money(r.value_after),
                     r.rungs_left, f"{r.avg_years_left:.1f}", money(r.interest_next_year)]
                    + [f"{r.composition.get(b, 0):.0%}" if r.rungs_left else "-" for b in buckets])
    print(table(["Date", "Before", "Paid", "After", "Bonds", "Avg yrs", "Grows by"] + buckets,
                rows, "l" + "r" * (6 + len(buckets))))
    explain(args, """
Each year one bond matures and pays that year's $50,000. The bonds that are
left keep growing at the yield they were bought at ('Grows by'), which is
exactly what makes the next payment affordable. Values shown are book values
(steady growth at the locked-in yield). Market prices will move up and down
with interest rates in between, but because every bond is held until it
pays out, those moves never change what Laura receives. The reserve starts
long (most money in 5-10 year bonds) and gets shorter every year until it
is just the final bond.""")


def cmd_project(cfg, args):
    s = Schedule.from_config(cfg)
    rcfg, pcfg = cfg["reserve"], cfg["projection"]
    carve = cfg["decision_dates"]["reserve_carve_out"]
    growth = pcfg["growth_rate"] if args.growth is None else args.growth
    method = args.pricing or rcfg["pricing"]
    fed = try_fed_curve(cfg, args, required=method != "flat")[0] if method != "flat" else None

    p = project(s, carve, growth)
    print(f"PORTFOLIO PROJECTION AT {growth:.2%} A YEAR (deterministic, one path)\n")
    print(table(["Year", "Jan 1 value", "Cash in/out", "Invested", "Return", "Growth", "Dec 31 value"],
                [[r.year, money(r.value_start), money(r.flows), money(r.value_invested),
                  f"{r.rate:.2%}", money(r.growth), money(r.value_end)] for r in p.rows]))
    lad = build_ladder(s, rcfg, carve, reserve_curve(cfg, method, rate=args.rate, fed_curve=fed))
    fr = facility_contribution(p.final_value, lad.cost)
    print(f"\nOn {carve}:  portfolio {money(fr.portfolio_value)}"
          f"  - reserve {money(fr.reserve_cost)} ({lad.curve.describe()})")
    if fr.fully_funded:
        print(f"            = facility contribution {money(fr.contribution)}")
    else:
        print(f"            RESERVE SHORT BY {money(fr.shortfall)}: facility contribution $0")
    explain(args, """
Money goes in on Jan 1, then grows at the stated rate for the year. On the
carve-out date the reserve is bought FIRST. Only what is left can go to the
facility. If the portfolio can't cover the reserve, the facility gets
nothing and the shortfall is reported rather than hidden.
This is one straight-line path. It is useful for checking the arithmetic,
not for the co-sponsor range: that comes from the Monte Carlo in step 5.""")

    # Grid: growth scenarios x reserve rates
    print("\nFACILITY CONTRIBUTION GRID (rows: portfolio growth, columns: reserve rate)\n")
    rates = rcfg["scenario_rates"]
    costs = {r: build_ladder(s, rcfg, carve, reserve_curve(cfg, "flat", rate=r)).cost for r in rates}
    rows = []
    for g in pcfg["scenario_growth_rates"]:
        v = project(s, carve, g).final_value
        rows.append([f"{g:.0%} growth", money(v)]
                    + [money(facility_contribution(v, costs[r]).contribution) for r in rates])
    print(table(["", "Portfolio 2033"] + [f"@ {r:.0%}" for r in rates], rows))


def cmd_range(cfg, args):
    rt = cfg["range_table"]
    root = Path(cfg["_path"]).parent
    pf = Portfolio.from_file(args.portfolio or root / cfg["portfolio_file"])
    ctx = build_context(cfg, args)
    secs, as_of = ctx.securities, ctx.as_of
    pillars = load_pillars(cfg, ctx)
    range_pillar = next(p for p in pillars if p.name == "range")
    pct = lambda x: f"{x:.1%}"

    if not (args.buy or args.sell):
        print(f"PORTFOLIO STRUCTURE as of {as_of}   total {money(pf.total)}   cash {money(pf.cash)}\n")
        for dim, limits in (("asset_class", {k: v[1] for k, v in rt["asset_class_budgets"].items()}),
                            ("region", {k: v[1] for k, v in rt["region_budgets"].items()})):
            ex = exposures(pf, secs, dim)
            print(table([dim.replace("_", " ").title(), "Share", "Max"],
                        [[k, pct(ex.get(k, 0)), pct(limits[k])] for k in limits]) + "\n")
        us = rt["us_region"]
        non_us = sum(v for k, v in exposures(pf, secs, "region").items() if k != us)
        print(f"Non-US total: {pct(non_us)} (cap {rt['non_us_cap']:.0%})\n")
        sec = exposures(pf, secs, "sector", rt["sector_cap_asset_classes"])
        print(table(["Sector (equity)", "Share", "Cap"],
                    [[k, pct(v), "exempt" if k in rt["sector_cap_exempt"] else pct(rt["sector_cap"])]
                     for k, v in sorted(sec.items(), key=lambda kv: -kv[1])]))
        a = range_pillar.assess(pf)
        print(f"\nPositions: {a.positions} (target {rt['positions']['target'][0]}-{rt['positions']['target'][1]})")
        print(f"Average pairwise correlation ({rt['correlation']['window_days']} days, "
              f"{rt['correlation']['averaging']}): {a.avg_corr:.2f}  (limit {rt['correlation']['max_avg_pairwise']:.2f})")
        print(f"Effective number of bets: {a.bets:.1f} out of {len(a.corr_tickers)} holdings measured")
        print(f"Fullest cap: {a.fullest_cap[0]} at {a.fullest_cap[1]:.0%} of its limit")
        if a.excluded:
            print(f"Left out of correlation (not enough price history): {', '.join(a.excluded)}")
        print(f"\nRANGE SCORE {a.score:.0f}/100   "
              + "   ".join(f"{k} {v:.0f}" for k, v in a.subscores.items()))
        explain(args, """
Exposures count ETFs by what they hold when securities.yaml says so; broad
ETFs marked 'Diversified' are capped as a whole fund instead of by sector.
Correlation uses only daily closing prices up to the as-of date, which we
turn into daily returns ourselves (dividends added back). The effective
number of bets comes from the eigenvalues of that correlation matrix: 17
stocks that all move together count as far fewer than 17 bets.
The Range score mixes four parts (weights in config.yaml): correlation,
bets, headroom (how close the fullest cap is to its limit) and position
count against the 25-35 target.""")
        return

    ticker = (args.buy or args.sell).upper()
    if args.amount is None:
        raise ConfigError("give --amount in dollars for the trade")
    prop = Proposal(ticker, "buy" if args.buy else "sell", args.amount, as_of)
    d = decide(prop, pf, pillars, cfg)
    print(f"{prop.action.upper()} {money(prop.amount)} of {ticker} on {as_of}")
    print(f"DECISION: {d.status}" + (f"   combined score {d.score:.0f}" if d.score is not None else ""))
    print(f"Because:  {d.reason}\n")
    for r in d.results:
        print(f"[{r.pillar}] score {'n/a' if r.score is None else f'{r.score:.0f}'}"
              + ("   BLOCKED" if r.blocked else ""))
        shown = r.rules if args.explain else r.fired
        for rule in shown:
            mark = "PASS" if rule.passed else rule.severity.upper()
            print(f"   {mark:5}  {rule.code:20} {rule.message}")
        if not shown:
            print("   no rules fired (use --explain to see every check)")
        if "analysis" in r.details and r.score is not None:
            print()
            for line in format_analysis(r.details["analysis"], args.explain):
                print("   " + line)
        if "before" in r.details:
            b, a = r.details["before"], r.details["after"]
            print(f"\n   {'':22}{'before':>8}{'after':>8}")
            print(f"   {'avg correlation':22}{b.avg_corr:8.2f}{a.avg_corr:8.2f}")
            print(f"   {'effective bets':22}{b.bets:8.1f}{a.bets:8.1f}")
            print(f"   {'positions':22}{b.positions:8d}{a.positions:8d}")
            print(f"   {'fullest cap used':22}{b.fullest_cap[1]:8.0%}{a.fullest_cap[1]:8.0%}   ({a.fullest_cap[0]})")
            print(f"   {'range score':22}{b.score:8.0f}{a.score:8.0f}")
    if d.not_run:
        print(f"\nNot consulted (a structural pillar blocked first): {', '.join(d.not_run)}")
    explain(args, """
The Range Table runs first. Any BLOCK line stops the order, and the other
pillars are not asked. WARN lines are recorded but don't stop anything,
e.g. a cap already broken before this trade that the trade doesn't add to.
The Range score is the score of the portfolio AFTER the trade, so a buy
that spreads the portfolio out scores higher than one that concentrates it.""")


def build_context(cfg, args) -> Context:
    """Everything the pillars read, as of one date (today unless --as-of)."""
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    curve, _ = try_fed_curve(cfg, args, required=False)
    if curve is not None and curve.as_of > as_of:  # point in time: reload the curve as of that date
        curve = load_fed_curve(fetch_fed_curve_file(cfg), as_of)
    return Context(cfg, load_securities(cfg), YahooChartPrices(cfg, offline=args.offline), as_of,
                   fundamentals=EdgarFundamentals(cfg, offline=args.offline), curve=curve)


def cmd_score(cfg, args):
    if not args.tickers:
        raise ConfigError("name at least one ticker, e.g. python -m fhl score KO")
    ctx = build_context(cfg, args)
    conv = next((p for p in load_pillars(cfg, ctx) if p.name == "convergence"), None)
    if conv is None:
        raise ConfigError("the convergence pillar is not enabled in config.yaml")
    for t in args.tickers:
        a = conv.analyze(t.upper())
        print(f"\n{t.upper()}  —  CONVERGENCE as of {ctx.as_of}"
              + (f":  {a.score:.0f}/100" if a.score is not None else ""))
        for line in format_analysis(a, args.explain):
            print("  " + line)
        mos = a.facts.get("margin_of_safety")
        need = cfg["convergence"]["hard_rules"]["min_margin_of_safety"]
        if a.score is not None:
            print("  Buy rule: " + ("can't check the margin of safety" if mos is None else
                                    f"margin of safety {mos:.0%} {'meets' if mos >= need else 'is below'} "
                                    f"the required {need:.0%}"))
    explain(args, """
Each metric is turned into 0-100 on a straight line between a 'worst' value
(0) and a 'best' value (100) set in config.yaml. A source (quality, dividend,
value) is the weighted average of its metrics. The final score is the
weighted average of the sources, times an agreement factor: the more sources
at or above the threshold, the closer the factor is to 1. All numbers come
from the company's own filings (as they were public on the as-of date) and
daily prices; nothing is taken ready-made from a data vendor.""")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fhl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["schedule", "reserve", "project", "range", "score"])
    ap.add_argument("tickers", nargs="*", help="for 'score': the stocks to score")
    ap.add_argument("--config", help="path to config.yaml (default: the one next to fhl/)")
    ap.add_argument("--pricing", choices=PRICING_METHODS, help="override reserve.pricing")
    ap.add_argument("--rate", type=float, help="override reserve.flat_rate, e.g. 0.045")
    ap.add_argument("--growth", type=float, help="override projection.growth_rate, e.g. 0.06")
    ap.add_argument("--offline", action="store_true", help="don't download the Fed curve")
    ap.add_argument("--explain", action="store_true", help="plain-English notes on each number")
    ap.add_argument("--portfolio", help="holdings file (default: portfolio_file in config)")
    trade = ap.add_mutually_exclusive_group()
    trade.add_argument("--buy", metavar="TICKER", help="check a buy against the rules")
    trade.add_argument("--sell", metavar="TICKER", help="check a sell against the rules")
    ap.add_argument("--amount", type=float, help="trade size in dollars")
    ap.add_argument("--as-of", help="date to evaluate on, YYYY-MM-DD (default today)")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        {"schedule": cmd_schedule, "reserve": cmd_reserve, "project": cmd_project,
         "range": cmd_range, "score": cmd_score}[args.command](cfg, args)
    except ConfigError as e:
        print(f"CONFIG PROBLEM: {e}", file=sys.stderr)
        return 2
    return 0
