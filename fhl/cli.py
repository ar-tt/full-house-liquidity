"""Command line: python -m fhl <schedule|reserve|project> [options]

  schedule   print Laura's cash-flow rows
  reserve    price the operating reserve, show what it holds and how it runs down
  project    grow the portfolio to 2033 and compute the facility contribution
  range      check the portfolio's structure, or a trade, against every pillar
             e.g.  python -m fhl range --buy MSFT --amount 5000 --explain
  score      the Convergence Engine's full breakdown for one or more stocks
             e.g.  python -m fhl score KO MSFT --explain
  value-compare  margin of safety for every stock under each valuation what-if
             e.g.  python -m fhl value-compare --variants C,3 --explain
  clock      the Position Clock's entry timing and size for one or more tickers
             e.g.  python -m fhl clock MSFT VTI --explain
  wind-down  the long hand: calendar, funded status and growth target
             e.g.  python -m fhl wind-down --explain
  simulate   Monte Carlo: certainty test, what certainty costs, the 2031 range
             e.g.  python -m fhl simulate --explain     (add --paths 2000 for a quick run)
  backtest   backtests: --part plan (default) | clock | range | convergence | all
             e.g.  python -m fhl backtest --part all --explain

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
from .pillars.convergence import ConvergenceEngine, format_analysis, what_if
from .pillars.position_clock import format_plan, format_wind_down
from .wind_down import band_on, hedge_share_on
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
        for dim, limits in (("asset_class", {k: v[1] for k, v in range_pillar.ac_budgets.items()}),
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
        if "plan" in r.details:
            print()
            for line in format_plan(r.details["plan"], cfg):
                print("   " + line)
        if "wind_down" in r.details and args.explain:
            for line in format_wind_down(r.details["wind_down"], cfg):
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


def cmd_value_compare(cfg, args):
    ctx = build_context(cfg, args)
    spec = next(p for p in cfg["pillars"] if p["name"] == "convergence")
    need = cfg["convergence"]["hard_rules"]["min_margin_of_safety"]
    variants = cfg["convergence"]["valuation_variants"]
    if args.variants:
        wanted = [c.strip() for c in args.variants.split(",")]
        unknown = set(wanted) - {v["code"] for v in variants}
        if unknown:
            raise ConfigError(f"unknown variant(s) {sorted(unknown)}; see valuation_variants in config.yaml")
        variants = [v for w in wanted for v in variants if v["code"] == w]
    tickers = args.tickers and [t.upper() for t in args.tickers] or \
        [t for t, s in ctx.securities.items() if s.type == "stock"]
    results = {}
    for v in variants:
        eng = ConvergenceEngine(spec, Context(what_if(cfg, v["changes"]), ctx.securities, ctx.prices,
                                              ctx.as_of, ctx.fundamentals, ctx.curve))
        results[v["code"]] = {t: eng.analyze(t) for t in tickers}

    print(f"MARGIN OF SAFETY BY VALUATION VERSION as of {ctx.as_of}   "
          f"(buy rule: at least {need:.0%}; * = passes)\n")
    rows = []
    for t in tickers:
        cells = []
        for v in variants:
            a = results[v["code"]][t]
            mos = a.facts.get("margin_of_safety") if a.status == "scored" else None
            cells.append("n/a" if mos is None else f"{mos:+.0%}" + (" *" if mos >= need else "  "))
        rows.append([t] + cells)
    passing = []
    for v in variants:
        vals = [results[v["code"]][t].facts.get("margin_of_safety") for t in tickers
                if results[v["code"]][t].status == "scored"]
        vals = [m for m in vals if m is not None]
        passing.append(f"{sum(m >= need for m in vals)} of {len(vals)}")
    rows.append(["passing"] + passing)
    print(table(["Stock"] + [v["code"] for v in variants], rows))
    print()
    for v in variants:
        scores = sorted(a.score for a in results[v["code"]].values() if a.score is not None)
        med = scores[len(scores) // 2] if scores else float("nan")
        print(f"  {v['code']:3} {v['name']:45} median Convergence score {med:.0f}")
    print("\n  n/a = can't be valued (foreign currency, no free cash flow, bank, or missing data)")

    blend = next((v for v in variants if results[v["code"]] and any(
        a.facts.get("margin_of_safety_basis") == "blended" for a in results[v["code"]].values())), None)
    if blend:
        methods = ["dcf", "own_ev_ebit", "own_fcf_yield", "sector_ev_ebit", "blended"]
        print(f"\nVALUE PER SHARE BY METHOD (version {blend['code']})\n")
        rows = []
        for t in tickers:
            a = results[blend["code"]][t]
            vps, price = a.facts.get("value_per_share"), a.facts.get("price")
            if not vps or price is None:
                continue
            rows.append([t, f"${price:,.0f}"] + ["-" if m not in vps else f"${vps[m]:,.0f}" for m in methods])
        print(table(["Stock", "Price", "DCF", "Own EV/EBIT", "Own FCF yld", "Sector", "Blended"], rows))
        w = cfg["convergence"]["valuation"]["blend_weights"]
        print("\n  Blend weights: " + ", ".join(f"{k} {x:.0%}" for k, x in w.items())
              + " (re-weighted over the methods available)")
    explain(args, """
Every version keeps the 25% rule and the same WACC. The DCF versions change
the shape of the cash-flow forecast ('fade' = growth steps down evenly to
2.5% instead of dropping there at once). The blended versions measure the
25% against a weighted mix of four values: the DCF; the price if EV/EBIT
went back to the company's own 10-year median; the price if its free-cash-
flow yield went back to its own median; and the price at the sector's
median EV/EBIT. The history and sector methods ask 'cheap compared with
what it usually costs?'; the DCF asks 'cheap compared with its cash
flows?'. A company that has always been expensive looks fair on the first
question and expensive on the second.""")

def _load_portfolio(cfg, args) -> Portfolio:
    return Portfolio.from_file(args.portfolio or Path(cfg["_path"]).parent / cfg["portfolio_file"])


def cmd_clock(cfg, args):
    if not args.tickers:
        raise ConfigError("name at least one ticker, e.g. python -m fhl clock MSFT")
    ctx = build_context(cfg, args)
    pf = _load_portfolio(cfg, args)
    clock = next((p for p in load_pillars(cfg, ctx) if p.name == "position"), None)
    if clock is None:
        raise ConfigError("the position pillar is not enabled in config.yaml")
    for t in (t.upper() for t in args.tickers):
        if t not in ctx.securities:
            print(f"\n{t}: not in securities.yaml")
            continue
        plan = clock.plan(t, pf)
        print(f"\n{t}  —  POSITION CLOCK as of {ctx.as_of}")
        if plan is None:
            print("  no price history")
            continue
        for line in format_plan(plan, cfg):
            print("  " + line)
        d = clock.evaluate(Proposal(t, "buy", max(plan.tranche_size, 1.0), ctx.as_of), pf)
        waits = [r.message for r in d.fired if r.severity == "wait"]
        print("  Now: " + ("WAIT - " + waits[0] if waits else f"a ${plan.tranche_size:,.0f} tranche can go in today"))
    explain(args, """
The clock only decides WHEN and in what steps; Range and Convergence decide
WHAT. Trend (price vs its 200-day average, ADX for strength) sets the pace:
3 tranches 2, 3 or 4 weeks apart. RSI under 40 means the stock has sold
off, so the next tranche comes a week early; over 70 it has run up, so the
tranche waits. By 8 weeks after the first tranche, everything is in. Size
is set so calmer stocks get bigger positions than jumpy ones.""")


def cmd_wind_down(cfg, args):
    ctx = build_context(cfg, args)
    pf = _load_portfolio(cfg, args)
    clock = next((p for p in load_pillars(cfg, ctx) if p.name == "position"), None)
    if clock is None:
        raise ConfigError("the position pillar is not enabled in config.yaml")
    wd = clock.wind_down(pf)
    print(f"WIND-DOWN (LONG HAND) as of {ctx.as_of}"
          + (f"   portfolio includes contributions through {pf.contributions_through}"
             if pf.contributions_through else "") + "\n")
    for line in format_wind_down(wd, cfg):
        print("  " + line)
    print("\nCALENDAR\n")
    rows = []
    for y in range(cfg["meta"]["year_zero"] + 1, cfg["decision_dates"]["reserve_carve_out"].year + 1):
        lo, hi = band_on(cfg, date(y, 1, 1))
        rows.append([y, f"{lo:.0%}-{hi:.0%}", f"{hi:.0%}", f"{hedge_share_on(cfg, date(y, 1, 1)):.0%}"])
    print(table(["Year", "Growth band", "Ceiling", "Reserve already in Treasuries"], rows))
    explain(args, """
The calendar gives a band for the growth share and a hard ceiling. Where to
sit is decided by PROJECTED FUNDED STATUS: the portfolio projected to 2033
(following the calendar at the expected returns in config.yaml, plus
Laura's promised contributions) divided by what the reserve is projected to
cost then. Behind: stay at the ceiling, never above it and never borrowing.
On track: slide toward the bottom of the band. Far ahead: lock the whole
reserve in Treasuries now. From 2030 a rising share of the reserve must
already be in Treasuries, so 2033 never forces a sale of stocks in a slump.""")


def cmd_simulate(cfg, args):
    import numpy as np
    from . import montecarlo as mc
    from .config import with_changes
    from .projection import project

    m = cfg["monte_carlo"]
    paths = args.paths or m["paths"]
    threshold = args.threshold or m["certainty_threshold"]
    s = Schedule.from_config(cfg)
    start, carve = m["start"], cfg["decision_dates"]["reserve_carve_out"]
    qd = m["quote"]["date"]
    months = len(mc.month_starts(start, carve)) - 1
    value0 = sum(c.amount for c in s.contributions() if c.date <= start)
    curve, _ = try_fed_curve(cfg, args, required=False)
    r0 = mc.start_rate(cfg, curve)
    hist = mc.load_history(cfg, offline=args.offline) if m["growth_returns"]["model"] == "bootstrap" else None
    market = mc.make_market(cfg, months, paths, m["seed"], hist)
    g = m["growth_returns"]
    print(f"MONTE CARLO: {paths:,} paths, {start} to {carve} (seed {m['seed']})")
    if hist is not None:
        print(f"  Stock returns: {g['series']} monthly history since {g['history_start']} ({len(hist)} months), "
              f"replayed in {g['block_months']}-month blocks, re-centred to a +{g['median_annual_return']:.0%} "
              f"median year (volatility {market.long_run_vol:.1%} a year)")
    else:
        print(f"  Stock returns: bell curve, +{g['median_annual_return']:.0%} median year, "
              f"volatility {market.long_run_vol:.0%}")
    rt = m["rates"]
    print(f"  Treasury yield: starts {r0:.2%}" + (f" (Fed {rt['start_maturity_years']}y rate, {curve.as_of})" if curve else "")
          + f", drifts toward {rt['long_run']:.2%}, typical move +/-{rt['vol_per_year']:.1%} a year")
    print(f"  A path succeeds if the reserve is fully covered on {carve} AND finishing the ladder "
          f"doesn't need a stock sale in a {m['forced_sale_drawdown']:.0%}+ slump")

    runs = mc.policy_menu(cfg, s, market, start, value0, r0)
    x = mc.simulate(cfg, s, market, start, value0, r0, record=(qd,))
    live_ok = x.p(x.success) >= threshold
    dial = cfg["wind_down"]["risk_dial"]
    ramp = ", ".join(f"{r['share']:.0%} from {r['from'].year}" for r in cfg["wind_down"]["hedge_ramp"]) or "none"
    print(f"\n1. YOUR LIVE SETTINGS (hedge ramp {ramp}; risk dial {dial:.2f})")
    print(f"   All ten payments secured on {x.p(x.success):.1%} of paths -> "
          f"{'PASSES' if live_ok else 'FAILS'} the {threshold:.0%} test")
    print(f"     reserve covered {x.p(x.funded):.1%} | stock sale in a slump needed {x.p(x.forced):.1%}")
    print("   Facility contribution in 2033 (after the reserve is fully funded): "
          + " | ".join(f"{q}th ${x.pct(q):,.0f}" if q != 50 else f"median ${x.pct(q):,.0f}" for q in (5, 20, 50, 70, 95)))
    jan = [i for i, d in enumerate(x.dates[:-1]) if d.month == 1]
    print("   Median growth share each Jan: " + ", ".join(
        f"{x.dates[i].year} {np.median(x.growth_share[i]):.0%}" for i in jan))
    print(f"   Volatility target binding in {x.vol_capped.mean():.0%} of path-months; "
          f"drawdown rule active in {x.dd_capped.mean():.0%}")

    print("\n2. CERTAINTY MENU: what each level costs Laura in facility money")
    rows, base = [], None
    for th in m["thresholds_to_compare"]:
        b = mc.best_for(runs, th)
        if b is None:
            rows.append([f"{th:.0%}", "none of the policies", "-", "-", "-", "-", "-"])
            continue
        base = base if base is not None else b.median
        r = b.result
        rows.append([f"{th:.0%}", f"{b.code}: {b.name}", f"{b.dial:.2f}", f"{b.success:.1%}", f"${b.median:,.0f}",
                     f"${r.pct(20):,.0f}-${r.pct(70):,.0f}", "-" if b.median == base else f"-${base - b.median:,.0f}"])
    print(table(["Certainty", "Best policy", "Dial", "Secured", "Median facility", "20th-70th", "Cost vs first row"],
                rows, "llrrrrr"))

    print(f"\n3. 2031 CO-SPONSOR QUOTE (floor = {m['quote']['floor_percentile']}th, ceiling = "
          f"{m['quote']['ceiling_percentile']}th percentile, after the reserve is fully funded)")
    pick = mc.best_for(runs, threshold)
    if live_ok:
        qcfg, qfull, label = cfg, x, "your live settings"
    elif pick is not None:
        qcfg = mc.policy_config(cfg, next(p for p in m["policies"] if p["code"] == pick.code), pick.dial)
        qfull, label = None, f"policy {pick.code} at dial {pick.dial:.2f} (your live settings miss {threshold:.0%})"
    else:
        qcfg = None
    if qcfg is None:
        print(f"   No policy meets {threshold:.0%}, so no quote can be made at that certainty.")
    else:
        if qfull is None or qd not in qfull.recorded:
            qfull = mc.simulate(qcfg, s, market, start, value0, r0, record=(qd,))
        v31, y31 = qfull.recorded[qd]
        qmarket = mc.make_market(cfg, len(mc.month_starts(qd, carve)) - 1, paths, m["seed"] + 1, hist)
        plan_v = project(s, qd, m["growth_returns"]["median_annual_return"]).final_value
        starts = [("plan path (+7% a year)", plan_v, r0)]
        if args.quote_value:
            starts.insert(0, ("your value", args.quote_value, r0))
        starts += [(f"{lbl} ({q}th percentile)", float(np.percentile(v31, q)), float(np.median(y31)))
                   for lbl, q in (("bad 2027-30", 20), ("typical 2027-30", 50), ("good 2027-30", 80))]
        print(f"   Using {label}")
        rows, plan_q = [], None
        for lbl, v, y in starts:
            q = mc.quote(qcfg, s, qmarket, qd, v, y)
            plan_q = plan_q or (q if lbl.startswith("plan") else None)
            rows.append([lbl, f"${v:,.0f}", f"${q.floor:,.0f}", f"${q.ceiling:,.0f}", f"{q.result.p(q.result.success):.1%}"])
        print(table([f"Portfolio on {qd}", "Value", "Floor", "Ceiling", "Secured"], rows))
        if plan_q:
            print(f"   On the plan path Laura could say: \"We are {plan_q.sentence()}.\"")

    print("\n4. EVERY POLICY AND DIAL (same 10,000 futures for all)")
    rows = [[r.code, r.name, f"{r.dial:.2f}", f"{r.success:.1%}", f"{r.result.p(r.result.forced):.1%}",
             f"${r.result.pct(20):,.0f}", f"${r.median:,.0f}", f"${r.result.pct(70):,.0f}"] for r in runs]
    print(table(["", "Policy", "Dial", "Secured", "Slump sale", "20th", "Median", "70th"], rows, "llrrrrrr"))
    explain(args, """
Each path is one possible future, month by month to 2033: stock returns
replayed from real history, Treasury yields drifting at random. Along the
way the portfolio follows the engine's own rules (wind-down target, then the
volatility target, the 20% drawdown rule and the turnover cap). 'Secured'
means the reserve was fully covered on 1 Jan 2033 and finishing the ladder
didn't force a stock sale in a slump; after that the ten payments are
certain, because each is a Treasury bond held to maturity. The facility
contribution is what's left once the reserve is fully paid for. The risk
dial scales the growth share down (0.5 = half as much in stocks); the menu
picks, for each certainty level, the policy and dial that leave Laura the
most facility money while still meeting it. The 2031 quote re-runs the
last two years from a 2031 starting value and reads off the 20th and 70th
percentiles.""")


def cmd_backtest(cfg, args):
    parts = ["plan", "clock", "range", "convergence"] if args.part == "all" else [args.part or "plan"]
    for part in parts:
        {"plan": _backtest_plan, "clock": _backtest_clock, "range": _backtest_range,
         "convergence": _backtest_convergence}[part](cfg, args)
        print()


def _backtest_clock(cfg, args):
    import numpy as np
    from . import backtest_clock as bc
    from .history import CurveHistory

    curves = CurveHistory(cfg, offline=args.offline)
    b, pc = cfg["backtest"]["clock"], cfg["position_clock"]
    short, long_ = min(b["horizons_days"]), max(b["horizons_days"])
    print("BACKTEST 2: the Position Clock's timing vs buying all at once")
    print(f"  Every trading day: decide to buy $1 of the ETF. LUMP = all that day; THIRDS = a third every "
          f"{b['naive_spacing_weeks']} weeks; CLOCK = the live rules (trend-set spacing, RSI, all in by "
          f"{pc['max_window_weeks']} weeks). Signals from the day before; waiting cash earns T-bills.\n")
    rows, pooled = [], []
    for t in b["series"]:
        bars, bills = bc.load(cfg, t, curves, offline=args.offline)
        e = bc.run(bars, bills, cfg, t)
        pooled.append(e)
        L, C, T = e.lump, e.clock, e.thirds
        rows.append([t, f"{e.dates[0].year}-{e.dates[-1].year}", f"{len(e.dates):,}",
                     f"{np.mean(C[long_] - L[long_]):+.2%}", f"{np.mean(C[long_] > L[long_]):.0%}",
                     f"{np.mean(C[long_] - T[long_]):+.2%}",
                     f"{np.percentile(L[short], 5) - 1:+.1%}", f"{np.percentile(C[short], 5) - 1:+.1%}"])
    print(table(["ETF", "Decisions", "Days", "Clock vs lump (12m)", "Clock wins", "Clock vs thirds",
                 "Worst 5% lump (3m)", "Worst 5% clock (3m)"], rows))
    print("\n   Do the signals help? Clock minus plain thirds, 12 months on, all ETFs pooled:")
    reg = np.concatenate([e.regime for e in pooled])
    rsi = np.concatenate([e.rsi for e in pooled])
    diff = np.concatenate([e.clock[long_] - e.thirds[long_] for e in pooled])
    rows = []
    for lbl, m in [(g.replace("_", " "), reg == g) for g in bc.REGIMES] + [
            (f"RSI < {pc['indicators']['rsi_accelerate_below']}", rsi < pc["indicators"]["rsi_accelerate_below"]),
            (f"RSI > {pc['indicators']['rsi_delay_above']}", rsi > pc["indicators"]["rsi_delay_above"])]:
        rows.append([lbl, f"{m.mean():.0%}", f"{diff[m].mean():+.2%}", f"{np.mean(diff[m] > 0):.0%}"])
    print(table(["On decision day", "Share of days", "Clock minus thirds", "Clock better"], rows))
    explain(args, """
Markets rise more often than they fall, so on average buying everything at
once wins; spreading purchases out is insurance against buying just before a
drop. The last two columns of the first table show that insurance: the worst
5% of outcomes three months later. The second table isolates the clock's
signals by comparing it with spreading purchases blindly on a fixed schedule.""")


def _backtest_range(cfg, args):
    from . import backtest_range as br
    br.report(cfg, args, table, explain)


def _backtest_convergence(cfg, args):
    from . import backtest_convergence as bcv
    bcv.report(cfg, args, table, explain)


def _backtest_plan(cfg, args):
    import numpy as np
    from . import backtest as bk

    s = Schedule.from_config(cfg)
    starts, market = bk.load(cfg, s, offline=args.offline)
    bt = bk.calibrate(cfg, s, bk.replay(cfg, s, starts, market))
    r, b = bt.result, cfg["backtest"]
    years = (len(r.dates) - 1) // 12
    print("BACKTEST: Laura's plan replayed through real history")
    print(f"  {len(starts)} windows, day one from {starts[0]} to {starts[-1]}, each running {years} years")
    print(f"  Stocks: {b['series']} (S&P 500 index fund), actual monthly total returns, "
          f"volatility {market.long_run_vol:.1%} a year")
    print("  Ladder: the Fed Treasury curve on each date; cash: the 3-month Treasury rate on each date")
    print(f"  Windows overlap, so this is roughly {len(starts) / (12 * years):.0f} independent "
          f"{years}-year stretches of history, not {len(starts)}")

    cover = r.total / r.reserve
    close = int(np.argmin(cover))
    dial = cfg["wind_down"]["risk_dial"]
    print(f"\n1. YOUR LIVE SETTINGS (risk dial {dial:.2f})")
    print(f"   All ten payments secured in {int(r.success.sum())} of {len(starts)} windows ({r.p(r.success):.1%})")
    print(f"   Closest call: day one {starts[close]}, the portfolio covered the reserve {cover[close]:.2f} times "
          f"(facility ${r.facility[close]:,.0f})")
    print("   Facility contribution: " + " | ".join(
        f"{q}th ${r.pct(q):,.0f}" if q != 50 else f"median ${r.pct(q):,.0f}" for q in (5, 20, 50, 70, 95)))
    rows = []
    for dec in sorted({d.year // 10 * 10 for d in starts}):
        m = np.array([d.year // 10 * 10 == dec for d in starts])
        rows.append([f"{dec}s", int(m.sum()), f"{np.mean(r.success[m]):.0%}", f"${np.median(r.reserve[m]):,.0f}",
                     f"${np.median(r.facility[m]):,.0f}", f"${r.facility[m].min():,.0f}"])
    print(table(["Day one in", "Windows", "Secured", "Median reserve cost", "Median facility", "Worst facility"], rows))
    print("\n   The five hardest windows:")
    rows = []
    for i in np.argsort(r.facility)[:5]:
        rows.append([str(starts[i]), str(bt.ends[i]), f"{np.prod(1 + market.growth[i]) - 1:+.0%}",
                     f"{market.rates[i, 0]:.1%} -> {market.rates[i, -1]:.1%}", f"${r.reserve[i]:,.0f}",
                     f"${r.facility[i]:,.0f}"])
    print(table(["Day one", "Carve-out", "Stocks over window", "Ladder yield", "Reserve cost", "Facility"], rows))

    print("\n2. EVERY POLICY ON REAL HISTORY")
    rows = []
    for run in bk.policy_menu(cfg, s, starts, market):
        x = run.result
        rows.append([run.code, run.name, f"{run.dial:.2f}", f"{run.success:.1%}", f"{np.min(x.total / x.reserve):.2f}",
                     f"${x.pct(20):,.0f}", f"${x.pct(50):,.0f}", f"${x.pct(70):,.0f}"])
    print(table(["", "Policy", "Dial", "Secured", "Closest call", "20th", "Median", "70th"], rows, "llrrrrrr"))

    q = cfg["monte_carlo"]["quote"]
    fac = r.facility
    below, above = np.mean(fac < bt.floor), np.mean(fac > bt.ceiling)
    print(f"\n3. QUOTE CHECK: each window's {q['date'].year}-style quote, made with what was known then, "
          "vs what actually happened")
    print(f"   Below the floor {below:.0%} (the quote says {q['floor_percentile']}%) | inside "
          f"{1 - below - above:.0%} (says {q['ceiling_percentile'] - q['floor_percentile']}%) | above the ceiling "
          f"{above:.0%} (says {100 - q['ceiling_percentile']}%)")
    rows = []
    for dec in sorted({d.year // 10 * 10 for d in starts}):
        m = np.array([d.year // 10 * 10 == dec for d in starts])
        rows.append([f"{dec}s", int(m.sum()), f"{np.mean(fac[m] < bt.floor[m]):.0%}",
                     f"{np.mean((fac[m] >= bt.floor[m]) & (fac[m] <= bt.ceiling[m])):.0%}",
                     f"{np.mean(fac[m] > bt.ceiling[m]):.0%}"])
    print(table(["Day one in", "Windows", "Below floor", "Inside", "Above ceiling"], rows))
    explain(args, """
Every month since 1981 is tried as Laura's day one, and the plan runs six
years exactly as the live engine would: same wind-down, risk dial and risk
controls, with the S&P 500's real returns and the real Treasury curve of
each date. Nothing is invented, but windows overlap: the 2008 crash sits
inside dozens of them, so treat 476 windows as about seven separate
stretches of history. The quote check re-makes the co-sponsor quote in each
window using only what was known on that day, then compares it with where
the facility money really landed. If the quote were perfectly calibrated,
about 20% of windows would land below the floor and 30% above the ceiling.""")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fhl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["schedule", "reserve", "project", "range", "score", "value-compare",
                                        "dcf-compare", "clock", "wind-down", "simulate", "backtest"])
    ap.add_argument("tickers", nargs="*", help="for 'score' / 'value-compare' / 'clock': the tickers")
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
    ap.add_argument("--variants", help="for 'value-compare': codes to show, e.g. C,3,3C (default: all)")
    ap.add_argument("--paths", type=int, help="for 'simulate': number of paths (default from config)")
    ap.add_argument("--threshold", type=float, help="for 'simulate': certainty level, e.g. 0.99")
    ap.add_argument("--quote-value", type=float, help="for 'simulate': portfolio value to quote from in 2031")
    ap.add_argument("--part", choices=["plan", "clock", "range", "convergence", "all"],
                    help="for 'backtest': which backtest (default: plan)")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        {"schedule": cmd_schedule, "reserve": cmd_reserve, "project": cmd_project,
         "range": cmd_range, "score": cmd_score, "value-compare": cmd_value_compare,
         "dcf-compare": cmd_value_compare, "clock": cmd_clock,
         "wind-down": cmd_wind_down, "simulate": cmd_simulate,
         "backtest": cmd_backtest}[args.command](cfg, args)
    except ConfigError as e:
        print(f"CONFIG PROBLEM: {e}", file=sys.stderr)
        return 2
    return 0
