"""Every Range Table rule, on synthetic prices (no internet needed)."""
import copy

import pytest

from fhl.config import ConfigError
from fhl.pillars.base import BLOCK, WARN, Context
from fhl.pillars.range_table import RangeTable
from fhl.portfolio import Portfolio, Proposal
from conftest import bars_from_returns, factor_prices, sec, weekdays

N_DAYS = 120
AS_OF = weekdays(N_DAYS + 1)[-1]


def universe():
    s = {}
    for prefix, sector in (("IND", "Industrials"), ("HC", "Health Care"),
                           ("FIN", "Financials"), ("UTL", "Utilities")):
        for i in range(10):
            s[f"{prefix}{i}"] = sec(f"{prefix}{i}", sector)
    for i in range(5):
        s[f"IT{i}"] = sec(f"IT{i}", "Information Technology")
        s[f"DEV{i}"] = sec(f"DEV{i}", "Health Care", region="Developed ex-US")
        s[f"EM{i}"] = sec(f"EM{i}", "Materials", region="Emerging")
    for i in range(6):
        s[f"CL{i}"] = sec(f"CL{i}", "Energy")
    s["GOOGL"] = sec("GOOGL", "Communication Services", issuer="Alphabet")
    s["GOOG"] = sec("GOOG", "Communication Services", issuer="Alphabet")
    s["BROAD"] = sec("BROAD", "Diversified", type="etf")
    s["TECHETF"] = sec("TECHETF", type="etf",
                       sector_weights={"Information Technology": 0.6, "Communication Services": 0.4})
    s["INTLETF"] = sec("INTLETF", "Diversified", region="Developed ex-US", type="etf")
    s["BONDETF"] = sec("BONDETF", "Government", type="etf", asset_class="bond")
    s["STRIP"] = sec("STRIP", "Government", type="bond", asset_class="bond",
                     issuer="US Treasury", has_prices=False)
    s["NEWCO"] = sec("NEWCO", "Industrials")
    return s


def prices():
    secs = universe()
    clones = [t for t in secs if t.startswith("CL")]
    others = [t for t in secs if t not in clones and t not in ("STRIP", "NEWCO")]
    # each "other" ticker in its own independent group -> roughly uncorrelated
    groups = {t: (0.0, [t]) for t in others}
    groups["clones"] = (0.95, clones)
    p = factor_prices(groups, n_days=N_DAYS)
    p.data["NEWCO"] = bars_from_returns([0.01, -0.012] * 10, start=weekdays(N_DAYS - 19)[-1])
    return p


PRICES = prices()


@pytest.fixture
def make(cfg):
    def _make(holdings, cash, changes=None):
        c = copy.deepcopy(cfg)
        for path, value in (changes or {}).items():
            node = c["range_table"]
            *head, last = path.split(".")
            for k in head:
                node = node[k]
            node[last] = value
        ctx = Context(c, universe(), PRICES, AS_OF)
        return RangeTable({"name": "range", "weight": 50, "structural": True}, ctx), Portfolio(holdings, cash)
    return _make


def run(pillar, pf, ticker, amount, action="buy"):
    return pillar.evaluate(Proposal(ticker, action, amount, AS_OF), pf)


def blocks(res):
    return [r.code for r in res.fired if r.severity == BLOCK]


def warns(res):
    return [r.code for r in res.fired if r.severity == WARN]


def names(*groups):
    return {t: v for prefix, count, v in groups for t in [f"{prefix}{i}" for i in range(count)]}


# 25 names x $30k = $750k in stocks, $250k cash, $1M total
BASE = names(("IND", 5, 30_000), ("HC", 5, 30_000), ("FIN", 5, 30_000), ("UTL", 5, 30_000),
             ("IT", 3, 30_000), ("DEV", 1, 30_000), ("EM", 1, 30_000))


# --- 5% single company -----------------------------------------------------------

def test_single_name_exactly_at_cap_passes(make):
    rt, pf = make(BASE, 250_000)
    assert blocks(run(rt, pf, "IND0", 20_000)) == []          # 5.0%


def test_single_name_over_cap_blocks(make):
    rt, pf = make(BASE, 250_000)
    assert blocks(run(rt, pf, "IND0", 21_000)) == ["single_name_cap"]   # 5.1%


def test_two_share_classes_count_as_one_company(make):
    rt, pf = make({**BASE, "GOOGL": 30_000}, 220_000)
    res = run(rt, pf, "GOOG", 25_000)
    assert blocks(res) == ["single_name_cap"]
    assert "Alphabet" in res.fired[0].message


def test_treasuries_are_exempt_from_single_name_cap(make):
    rt, pf = make(BASE, 250_000)
    res = run(rt, pf, "STRIP", 200_000)
    assert blocks(res) == []
    assert any(r.code == "single_name_cap" and "exempt" in r.message for r in res.rules)


# --- ETFs --------------------------------------------------------------------------

def test_etf_has_its_own_cap_not_the_5pct_one(make):
    rt, pf = make(names(("IND", 5, 30_000)), 850_000)
    assert blocks(run(rt, pf, "BROAD", 250_000)) == []          # 25%
    assert blocks(run(rt, pf, "BROAD", 260_000)) == ["etf_cap"]  # 26%


def test_etf_sector_look_through_counts_pro_rata(make):
    rt, pf = make(BASE, 250_000)
    res = run(rt, pf, "TECHETF", 50_000)                       # 5% of the portfolio
    got = {r.message.split(" ")[1] + " " + r.message.split(" ")[2]: r.value
           for r in res.rules if r.code == "sector_cap"}
    assert got == pytest.approx({"Information Technology": 0.09 + 0.6 * 0.05,
                                 "Communication Services": 0.4 * 0.05})


def test_broad_etf_is_exempt_from_sector_cap(make):
    rt, pf = make(BASE, 250_000)
    res = run(rt, pf, "BROAD", 140_000)
    assert blocks(res) == []
    assert not any(r.code == "sector_cap" for r in res.rules)


# --- 20% sector ------------------------------------------------------------------

def test_sector_at_cap_passes_and_over_cap_blocks(make):
    rt, pf = make(BASE, 250_000)                               # Health Care 15% + DEV0 3% = 18%
    assert blocks(run(rt, pf, "HC5", 20_000)) == []            # 20.0%
    assert blocks(run(rt, pf, "HC5", 25_000)) == ["sector_cap"]


def test_bonds_are_not_a_sector_bet(make):
    rt, pf = make(BASE, 250_000)
    res = run(rt, pf, "BONDETF", 240_000)
    assert blocks(res) == []
    assert not any(r.code == "sector_cap" for r in res.rules)


# --- geography -------------------------------------------------------------------

NONUS = names(("IND", 5, 30_000), ("UTL", 5, 30_000), ("FIN", 5, 30_000),
              ("DEV", 5, 30_000), ("EM", 3, 30_000))  # dev 15%, EM 9%, non-US 24%


def test_non_us_cap_blocks(make):
    rt, pf = make(NONUS, 310_000)
    assert blocks(run(rt, pf, "DEV0", 20_000)) == ["non_us_cap"]   # 26%


def test_region_budget_blocks(make):
    pf_holdings = {**names(("IND", 5, 30_000), ("HC", 5, 30_000)), "DEV0": 30_000,
                   **names(("EM", 3, 30_000))}
    rt, pf = make(pf_holdings, 1_000_000 - 390_000)
    assert blocks(run(rt, pf, "EM3", 15_000)) == ["region_budget"]  # EM 10.5%


# --- asset-class budgets ---------------------------------------------------------

def test_equity_budget_blocks(make):
    rt, pf = make(BASE, 250_000)                               # equity 75%
    assert blocks(run(rt, pf, "BROAD", 160_000)) == ["asset_class_budget"]   # 91%


def test_asset_class_minimum_only_warns(make):
    rt, pf = make(BASE, 250_000, {"asset_class_budgets.equity": [0.80, 0.90]})
    res = run(rt, pf, "STRIP", 10_000)
    assert blocks(res) == []
    assert "asset_class_min" in warns(res)


# --- correlation -----------------------------------------------------------------

def test_buy_that_pushes_correlation_over_limit_is_blocked(make):
    rt, pf = make({"CL0": 30_000, "CL1": 30_000, "IND0": 10_000}, 930_000)
    res = run(rt, pf, "CL2", 30_000)
    before, after = res.details["before"].avg_corr, res.details["after"].avg_corr
    assert before < 0.6 < after
    assert "correlation_cap" in blocks(res)


def test_buy_that_lowers_correlation_is_allowed_even_above_limit(make):
    rt, pf = make({"CL0": 30_000, "CL1": 30_000}, 940_000)
    res = run(rt, pf, "IND0", 30_000)
    assert res.details["before"].avg_corr > 0.6
    assert res.details["after"].avg_corr < res.details["before"].avg_corr
    assert blocks(res) == []


def test_equal_averaging_option(make):
    rt, pf = make({"CL0": 30_000, "CL1": 30_000, "IND0": 10_000}, 930_000,
                  {"correlation.averaging": "equal"})
    a = rt.assess(pf)
    assert a.avg_corr == pytest.approx(sum([0.95, 0, 0]) / 3, abs=0.12)


def test_new_stock_without_60_days_of_prices_is_blocked(make):
    rt, pf = make(BASE, 250_000)
    assert blocks(run(rt, pf, "NEWCO", 10_000)) == ["missing_history"]


def test_missing_history_can_be_set_to_warn(make):
    rt, pf = make(BASE, 250_000, {"correlation.missing_history": "warn"})
    res = run(rt, pf, "NEWCO", 10_000)
    assert blocks(res) == [] and "missing_history" in warns(res)


# --- number of positions ---------------------------------------------------------

FULL = names(("IND", 10, 20_000), ("HC", 10, 20_000), ("FIN", 10, 20_000), ("UTL", 5, 20_000))  # 35


def test_36th_name_is_blocked(make):
    rt, pf = make(FULL, 300_000)
    assert blocks(run(rt, pf, "UTL5", 10_000)) == ["max_positions"]


def test_adding_to_an_existing_name_at_max_is_fine(make):
    rt, pf = make(FULL, 300_000)
    assert blocks(run(rt, pf, "UTL0", 10_000)) == []


def test_bonds_dont_count_as_positions(make):
    rt, pf = make(FULL, 300_000)
    assert blocks(run(rt, pf, "BONDETF", 10_000)) == []


# --- trade mechanics -------------------------------------------------------------

def test_not_enough_cash_blocks(make):
    rt, pf = make(BASE, 5_000)
    assert blocks(run(rt, pf, "IND0", 6_000)) == ["insufficient_cash"]


def test_selling_more_than_held_blocks(make):
    rt, pf = make(BASE, 250_000)
    assert blocks(run(rt, pf, "IND0", 40_000, "sell")) == ["oversell"]


def test_unknown_ticker_blocks(make):
    rt, pf = make(BASE, 250_000)
    assert blocks(run(rt, pf, "ZZZZ", 1_000)) == ["unknown_security"]


# --- limits already broken before the trade ---------------------------------------

def test_existing_breach_only_blocks_trades_that_add_to_it(make):
    # Health Care drifted to 22% on its own; IND0 grew to 6%
    h = {**BASE, **names(("HC", 5, 44_000)), "IND0": 60_000}
    rt, pf = make(h, 1_000_000 - sum(h.values()))
    assert blocks(run(rt, pf, "FIN0", 10_000)) == []            # unrelated: fine
    assert blocks(run(rt, pf, "HC5", 5_000)) == ["sector_cap"]  # adds to breach
    assert blocks(run(rt, pf, "IND0", 1_000)) == ["single_name_cap"]
    assert blocks(run(rt, pf, "IND1", 10_000)) == []


def test_sells_are_never_blocked_by_caps(make):
    h = {**BASE, **names(("HC", 5, 44_000)), "IND0": 60_000}
    rt, pf = make(h, 1_000_000 - sum(h.values()))
    for t in ("HC0", "IND0", "DEV0"):
        assert blocks(run(rt, pf, t, 5_000, "sell")) == []


# --- the Range score ---------------------------------------------------------------

def test_diversifying_buy_scores_higher_than_concentrating_buy(make):
    h = {"CL0": 30_000, "CL1": 30_000, **names(("IND", 4, 30_000))}
    rt, pf = make(h, 1_000_000 - sum(h.values()))
    diversify = run(rt, pf, "HC0", 30_000).score
    concentrate = run(rt, pf, "CL2", 30_000).score
    assert diversify > concentrate


def test_score_parts_follow_config(make):
    rt, pf = make(BASE, 250_000)
    a = rt.assess(pf)
    assert a.positions == 25 and a.subscores["positions"] == 100
    assert a.score == pytest.approx(sum(rt.rt["score_weights"][k] * v for k, v in a.subscores.items()))
    few = rt.assess(Portfolio(names(("IND", 10, 30_000)), 700_000))
    assert few.subscores["positions"] == 0          # 15 below the 25 minimum


def test_all_cash_portfolio_has_full_headroom(make):
    rt, pf = make({}, 1_000_000)
    assert rt.assess(pf).subscores["headroom"] == 100


def test_score_weights_must_add_to_one(make):
    with pytest.raises(ConfigError, match="add up to 1"):
        make(BASE, 0, {"score_weights": {"correlation": 0.5, "bets": 0.5, "headroom": 0.5, "positions": 0}})


def test_every_region_needs_a_budget(make):
    with pytest.raises(ConfigError, match="no budget"):
        make(BASE, 0, {"region_budgets": {"US": [0, 1]}})
