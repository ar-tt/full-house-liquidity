# Full House Liquidity — quant engine (Full House Capital)

Steps done: 1 (liability model, reserve, projection), 2 (the Range Table),
3 (the Convergence Engine), 4 (the Position Clock), 5 (Monte Carlo).

## Setup (once)
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

Then create `config.local.yaml` (kept out of git) with your contact email,
which the SEC asks automated users to send:

    fundamentals:
      user_agent: "Full House Capital WGHSIC quant engine you@example.com"

## Run
    .venv/bin/python -m fhl schedule                      # Laura's cash-flow rows
    .venv/bin/python -m fhl reserve                       # reserve cost, ladder, run-down
    .venv/bin/python -m fhl project                       # 2033 portfolio and facility contribution
    .venv/bin/python -m fhl range                         # portfolio structure vs the Range Table
    .venv/bin/python -m fhl range --buy MSFT --amount 5000   # check one trade (all pillars)
    .venv/bin/python -m fhl score KO MSFT --explain       # Convergence breakdown for stocks
    .venv/bin/python -m fhl value-compare --variants C,3C # margin of safety under valuation what-ifs
    .venv/bin/python -m fhl clock MSFT VTI --explain      # entry timing and tranche size
    .venv/bin/python -m fhl wind-down --explain           # growth target from calendar + funded status
    .venv/bin/python -m fhl simulate --explain            # certainty test, certainty menu, 2031 range
    .venv/bin/python -m fhl simulate --quote-value 590000 # 2031 range from an actual portfolio value
    .venv/bin/python -m fhl backtest --explain            # the plan replayed through real history since 1981
    .venv/bin/python -m fhl backtest --part all           # all four backtests (plan, clock, range, convergence)
    .venv/bin/python -m pytest -q                         # all tests

Options: `--explain` (plain-English notes and every rule checked),
`--pricing flat|curve_spot|curve_forward`, `--rate 0.045`, `--growth 0.06`,
`--portfolio file.yaml`, `--as-of 2026-06-30`, `--offline`, `--config other.yaml`.

Every number, date and switch lives in `config.yaml`. What each ticker is
(sector, region, type) lives in `securities.yaml`. `portfolio.yaml` is a
made-up example for exercising the rules, not a recommendation.

## Files
| File | What it does |
|---|---|
| `fhl/cashflows.py` | Cash-flow schedule, read as rows from config |
| `fhl/curves.py` | Interest-rate curves: flat, Fed Treasury zero curve, forward-implied |
| `fhl/reserve.py` | Prices the Treasury ladder, shows composition and run-down |
| `fhl/projection.py` | Grows the portfolio year by year; reserve first, facility gets the rest |
| `fhl/securities.py` | Loads and checks `securities.yaml` |
| `fhl/portfolio.py` | Holdings, trades, and exposure by asset class / sector / region / company |
| `fhl/prices.py` | Daily prices (free Yahoo feed, cached) behind a replaceable interface |
| `fhl/stats.py` | Correlation matrix, average pairwise correlation, effective number of bets |
| `fhl/pillars/base.py` | The common pillar interface and plug-in loader |
| `fhl/pillars/range_table.py` | Pillar 1: the Range Table (yes/no gate) |
| `fhl/fundamentals/edgar.py` | Downloads and caches SEC filings data |
| `fhl/fundamentals/annual.py` | Turns filings into one row per fiscal year, as known on a date |
| `fhl/fundamentals/metrics.py` | ROIC, FCF, growth, accruals, DCF, reverse DCF, beta |
| `fhl/pillars/convergence.py` | Pillar 2: the Convergence Engine (margin of safety on the 3C blend) |
| `fhl/indicators.py` | 200-day average, ADX, RSI, 60-day volatility from raw prices |
| `fhl/wind_down.py` | Long hand: calendar bands, projected funded status, hedge ramp |
| `fhl/pillars/position_clock.py` | Pillar 3: the Position Clock (entry timing, tranches, sizing) |
| `fhl/montecarlo.py` | Step 5: 10,000-path simulation, certainty menu, 2031 quote |
| `fhl/history.py` | Historical index returns and the Fed curve on every date |
| `fhl/backtest.py` | Backtest 1: the plan replayed through every 6-year window since 1981, plus the quote check |
| `fhl/backtest_clock.py` | Backtest 2: Position Clock timing vs lump sum, on 5 index ETFs |
| `fhl/backtest_range.py` | Backtest 3: do the Range Table's rules predict smaller losses? |
| `fhl/backtest_convergence.py` | Backtest 4: do Convergence scores predict returns? (survivor-biased) |
| `fhl/combine.py` | Runs pillars (structural first) and combines scores into BLOCKED / HOLD / ACT |
| `fhl/cli.py` | The commands above |

## Adding a pillar
Write `fhl/pillars/<name>.py` with a class that subclasses `Pillar` and
implements `evaluate(proposal, portfolio) -> PillarResult`, then add one
line under `pillars:` in `config.yaml`. Nothing else changes.

## How pillars combine
The Range Table is a yes/no gate: if any of its rules fails, the trade is
BLOCKED and nothing else is asked. Otherwise Convergence and the Position
Clock are averaged by weight (25/25). Convergence's hard rules (e.g. the 25%
margin of safety) can also block. The Position Clock never blocks: "not yet"
(tranche not due, RSI over 70, order bigger than the tranche, growth above
the wind-down target) puts the order on HOLD.

`portfolio.yaml` may list `entries` (positions being bought in tranches) and
`contributions_through` (the date through which Laura's contributions are
already in the values).

## Data (free tier) and its limits
- **Company filings:** SEC EDGAR "company facts" (free, official). Every
  number carries its filing date, so nothing is used before it was public,
  and delisted companies' filings remain available. The SEC asks automated
  users to put a contact email in `fundamentals.user_agent` in config.yaml.
  Limits: tagged data starts around 2009-2011 (a full 10 years exists only
  from about 2019); foreign companies filing IFRS start around 2018; figures
  a company files under its own custom tag (e.g. NextEra's capital spending)
  are not included; annual figures only (not trailing 12 months yet).
- **Foreign companies (TSM, ASML, NVO):** quality and dividend ratios work,
  but valuation needs currency conversion and ADR share ratios, which are not
  built yet. Their margin of safety can't be checked, so buys are blocked
  (setting: `convergence.hard_rules.margin_of_safety_unknown`). Their dividend
  streak is counted in US dollars, so exchange-rate moves can break it.
- **Banks and insurers:** not scored (their accounts don't fit ROIC, free
  cash flow or EBITDA).
- **Treasury curve:** Federal Reserve Gürkaynak-Sack-Wright zero-coupon curve,
  daily since 1961, point-in-time lookups supported.
- **Prices:** Yahoo's chart feed (unofficial). Fine for today's checks.
  Limits: delisted companies are mostly missing (survivor bias in
  backtests), and closes come already split-adjusted. Dividends are added
  back by our own code. Stooq was tested and now blocks automated downloads.
- **ETF look-through:** broad ETFs are marked `Diversified` and capped as a
  whole fund (25%). Their hidden sector exposure (e.g. the technology inside
  a total-market fund) is not yet counted toward sector caps. Fix: load
  issuers' published holdings files into `sector_weights`.

## Monte Carlo (step 5)
10,000 month-by-month futures to 2033. Stock returns are real VTI monthly
returns since 2001 (dividends included), replayed in 12-month blocks and
re-centred to a +7% median year; one Treasury yield drifts from today's Fed
10-year rate toward 4.5%. Along each path the portfolio follows the engine's
own wind-down rule and the risk controls (volatility target, 20% drawdown
rule, 50% turnover cap). A path succeeds if the reserve is fully covered on
1 Jan 2033 and finishing the ladder doesn't force a stock sale in a 10%+
slump. The facility contribution is what's left after the reserve is paid.

Not yet built: the risk controls as order-level hard stops in the live
engine (volatility target, drawdown rule, turnover cap, 3-day liquidity at
10% of volume, reserve never sold for the facility). The simulation obeys
them; the trade checker doesn't block on them yet.

## Backtest (plan level)
Every month from 1981 to 2020 is tried as Laura's day one; the plan runs six
years with the live rules on the S&P 500 index fund's real returns (VFINX,
dividends included), the Fed's real Treasury curve on each date for the
ladder, and the real 3-month Treasury rate for cash. An index has no survivor
bias. Windows overlap, so 476 windows are about seven independent stretches.
The quote check re-makes the 2031-style quote in each window from what was
known that day and compares it with what happened.

Live settings: full hedge ramp (33/67/100% from 2030/31/32) and risk dial 0.75,
chosen in step 5 (98.4% of simulated paths secured; 476 of 476 historical windows).

## Backtests 2-4
- **Clock** (`--part clock`): every trading day since 1993 on SPY, QQQ, IWM,
  EFA, EEM; lump sum vs plain thirds vs the live clock rules, signals from the
  day before. No survivor bias (index ETFs).
- **Range** (`--part range`): sector cap, correlation cap and effective bets
  tested on the 9 sector ETFs since 1998 (plus the starter stocks for
  correlation levels).
- **Convergence** (`--part convergence`): the 150 biggest US non-financial
  companies with public shares by 2013 revenue (SEC data, chosen as of then),
  scored every April 1 2015-2025 from filings public by that day. BIASED:
  companies that stopped trading have no free prices and drop out (the report
  counts them). First run downloads about 150 companies' filings (stored
  compressed, ~60 MB).
