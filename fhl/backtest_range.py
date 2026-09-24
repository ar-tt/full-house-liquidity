"""BACKTEST 3: do the Range Table's rules actually predict smaller losses?

  A. SECTOR CAP. Each month, 400 random mixes of the 9 sector ETFs are
     grouped by their biggest sector (<=20% = within the cap, 20-40%, >40%)
     and followed for 12 months: volatility, worst drop (max drawdown), return.
  B. CORRELATION CAP. The 60-day average correlation is tracked month by
     month for the sector ETFs and for the starter stocks. When was it over
     the 0.6 cap (when the Range Table would block buys), and what did the
     market do over the next 12 months?
  C. EFFECTIVE BETS. Does a higher effective number of bets (from the
     correlation matrix) come before smaller drawdowns?
Every measure uses only the 60 trading days before the month starts.
"""
from __future__ import annotations

import copy
from datetime import date

import numpy as np

from .prices import YahooChartPrices, daily_returns
from .securities import load_securities
from .stats import average_pairwise, effective_bets


def returns_matrix(prices, tickers, as_of=date.max, start=None):
    """Daily total returns on the dates every ticker traded: (dates, matrix)."""
    rets = {t: daily_returns(prices.bars(t, as_of)) for t in tickers}
    common = sorted(set.intersection(*(set(r) for r in rets.values())))
    if start:
        common = [d for d in common if d >= start]
    return common, np.array([[rets[t][d] for t in tickers] for d in common])


def month_starts_idx(dates, back, forward):
    out, seen = [], set()
    for i, d in enumerate(dates):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            if i >= back and i + forward < len(dates):
                out.append(i)
    return out


def mix(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Portfolio returns = asset returns x weights. (numpy 2.0 with Apple's
    maths library raises false floating-point warnings on this; silenced.)"""
    with np.errstate(all="ignore"):
        return returns @ weights


def years_text(years) -> str:
    """{2002, 2003, 2004, 2008} -> '2002-04, 2008'."""
    ys, out = sorted(years), []
    for y in ys:
        if out and y == out[-1][1] + 1:
            out[-1][1] = y
        else:
            out.append([y, y])
    return ", ".join(f"{a}" if a == b else f"{a}-{str(b)[2:]}" for a, b in out) or "never"


def drawdown(paths: np.ndarray) -> np.ndarray:
    """Max drawdown of each column of daily returns (days x portfolios)."""
    level = np.cumprod(1 + paths, axis=0)
    level = np.vstack([np.ones(level.shape[1]), level])
    return 1 - (level / np.maximum.accumulate(level, axis=0)).min(axis=0)


def corr_path(dates, R, back, forward, weights=None):
    """(month index, average pairwise correlation, effective bets) per month start."""
    out = []
    for i in month_starts_idx(dates, back, forward):
        c = np.corrcoef(R[i - back:i].T)
        out.append((i, average_pairwise(c, weights), effective_bets(c)))
    return out


def sector_cap_test(dates, R, cfg) -> dict:
    b = cfg["backtest"]["range"]
    back, fwd = cfg["range_table"]["correlation"]["window_days"], b["forward_days"]
    rng = np.random.default_rng(b["seed"])
    lo, hi = b["weight_bins"]
    bins = {f"<= {lo:.0%}": [], f"{lo:.0%}-{hi:.0%}": [], f"> {hi:.0%}": []}
    names = list(bins)
    for i in month_starts_idx(dates, back, fwd):
        w = rng.dirichlet(np.full(R.shape[1], b["dirichlet_alpha"]), size=b["random_portfolios"])
        path = mix(R[i:i + fwd], w.T)
        vol = path.std(axis=0, ddof=1) * np.sqrt(252)
        dd = drawdown(path)
        ret = np.prod(1 + path, axis=0) - 1
        top = w.max(axis=1)
        for k, m in enumerate([top <= lo, (top > lo) & (top <= hi), top > hi]):
            bins[names[k]].append(np.column_stack([vol[m], dd[m], ret[m]]))
    return {k: np.vstack(v) for k, v in bins.items() if v}


def load(cfg, offline=False):
    b = cfg["backtest"]
    c = copy.deepcopy(cfg)
    c["prices"]["history_start"] = b["history_start"]
    c["prices"]["cache_subdir"] = b["cache_subdir"]
    long_prices = YahooChartPrices(c, offline=offline)
    sec_dates, sec_R = returns_matrix(long_prices, b["range"]["sector_etfs"])
    one_per_company = {}
    for t, s in load_securities(cfg).items():
        if s.type == "stock":
            one_per_company.setdefault(s.issuer, t)      # two share classes = one company
    stock_prices = YahooChartPrices(cfg, offline=offline)
    st_dates, st_R = returns_matrix(stock_prices, sorted(one_per_company.values()))
    spy_dates, spy_R = returns_matrix(long_prices, ["SPY"])
    return (sec_dates, sec_R), (st_dates, st_R), dict(zip(spy_dates, spy_R[:, 0]))


def forward_market(spy: dict, d0, days: int):
    ds = sorted(k for k in spy if k >= d0)[:days]
    r = np.array([spy[k] for k in ds])
    return np.prod(1 + r) - 1, float(drawdown(r[:, None])[0])


def report(cfg, args, table, explain):
    b = cfg["backtest"]["range"]
    cap = cfg["range_table"]["correlation"]["max_avg_pairwise"]
    back, fwd = cfg["range_table"]["correlation"]["window_days"], b["forward_days"]
    (sd, sR), (td, tR), spy = load(cfg, offline=args.offline)
    print("BACKTEST 3: do the Range Table's rules predict smaller losses?")
    print(f"  Sector ETFs {', '.join(b['sector_etfs'])}: {sd[0]} to {sd[-1]}; "
          f"starter stocks: {td[0]} to {td[-1]} (survivors, fine for correlation)\n")

    res = sector_cap_test(sd, sR, cfg)
    print(f"A. SECTOR CAP: {b['random_portfolios']} random sector mixes a month, followed for 12 months")
    rows = [[k, f"{len(v):,}", f"{np.median(v[:, 0]):.1%}", f"{np.median(v[:, 1]):.1%}",
             f"{np.percentile(v[:, 1], 95):.1%}", f"{np.median(v[:, 2]):+.1%}"] for k, v in res.items()]
    print(table(["Biggest sector", "Portfolio-years", "Median volatility", "Median worst drop",
                 "Worst drop, bad 5%", "Median return"], rows))

    print(f"\nB. CORRELATION CAP ({cap:.1f}): months when the {back}-day average correlation was above it")
    rows = []
    for lbl, (dates, R) in (("Sector ETFs", (sd, sR)), ("Starter stocks", (td, tR))):
        path = corr_path(dates, R, back, fwd)
        over = [(dates[i], c) for i, c, _ in path if c > cap]
        fr_over = [forward_market(spy, d, fwd) for d, _ in over]
        fr_all = [forward_market(spy, dates[i], fwd) for i, _, _ in path]
        rows.append([lbl, f"{len(over)} of {len(path)}",
                     years_text({d.year for d, _ in over}),
                     f"{np.mean([x[0] for x in fr_over]):+.1%}" if fr_over else "-",
                     f"{np.mean([x[0] for x in fr_all]):+.1%}",
                     f"{np.mean([x[1] for x in fr_over]):.1%}" if fr_over else "-",
                     f"{np.mean([x[1] for x in fr_all]):.1%}"])
    print(table(["Holdings", "Months over the cap", "Years it happened", "S&P next 12m (over)",
                 "S&P next 12m (all)", "Worst drop next 12m (over)", "(all)"], rows))

    print("\nC. DO THE MEASURES PREDICT TROUBLE? Rank correlation with the next 12 months (sector ETFs, equal weight)")
    path = corr_path(sd, sR, back, fwd)
    ew = np.full(sR.shape[1], 1 / sR.shape[1])
    fwd_vol = np.array([mix(sR[i:i + fwd], ew).std(ddof=1) * np.sqrt(252) for i, _, _ in path])
    fwd_dd = np.array([drawdown(mix(sR[i:i + fwd], ew)[:, None])[0] for i, _, _ in path])
    corr_now = np.array([c for _, c, _ in path])
    bets_now = np.array([e for _, _, e in path])
    past_vol = np.array([mix(sR[i - back:i], ew).std(ddof=1) * np.sqrt(252) for i, _, _ in path])

    def rank_corr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rows = [[lbl, f"{rank_corr(x, fwd_vol):+.2f}", f"{rank_corr(x, fwd_dd):+.2f}"] for lbl, x in (
        ("Average correlation (60 days)", corr_now), ("Effective number of bets", bets_now),
        ("Past volatility (60 days), for comparison", past_vol))]
    print(table(["Measured at the month start", "vs next-12m volatility", "vs next-12m worst drop"], rows))
    explain(args, """
A tests the 20% sector cap: portfolios whose biggest sector is within the cap
are compared with lopsided ones over the next 12 months. B shows when the
0.6 correlation cap would have blocked buys, and what the market did next:
if the cap mostly fires after crashes, it blocks buying at good prices. C
asks whether the measures carry any warning at all: a rank correlation of +1
means 'higher now, always higher later', 0 means no link. Past volatility is
included as a simple benchmark the measures should beat to earn their place.""")
