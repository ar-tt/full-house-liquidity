# Full House Liquidity — quant engine (Full House Capital)

Steps done: 1 (liability model, reserve, projection), 2 (the Range Table),
3 (the Convergence Engine).

## Setup (once)
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

## Run
    .venv/bin/python -m fhl schedule                      # Laura's cash-flow rows
    .venv/bin/python -m fhl reserve                       # reserve cost, ladder, run-down
    .venv/bin/python -m fhl project                       # 2033 portfolio and facility contribution
    .venv/bin/python -m fhl range                         # portfolio structure vs the Range Table
    .venv/bin/python -m fhl range --buy MSFT --amount 5000   # check one trade (all pillars)
    .venv/bin/python -m fhl score KO MSFT --explain       # Convergence breakdown for stocks
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
| `fhl/pillars/convergence.py` | Pillar 2: the Convergence Engine |
| `fhl/combine.py` | Runs pillars (structural first) and combines scores into BLOCKED / HOLD / ACT |
| `fhl/cli.py` | The commands above |

## Adding a pillar
Write `fhl/pillars/<name>.py` with a class that subclasses `Pillar` and
implements `evaluate(proposal, portfolio) -> PillarResult`, then add one
line under `pillars:` in `config.yaml`. Nothing else changes.

## How pillars combine
The Range Table is a yes/no gate: if any of its rules fails, the trade is
blocked and nothing else is asked. Otherwise the scoring pillars
(Convergence now, the Position Clock in step 4) are averaged by weight.
Their own hard rules (e.g. the 25% margin of safety) can also block.

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
