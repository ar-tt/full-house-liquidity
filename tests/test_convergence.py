"""The Convergence Engine on a made-up company whose numbers we control."""
import copy
from datetime import date

import pytest

from fhl.fundamentals.edgar import InMemoryFundamentals
from fhl.pillars.base import BLOCK, WARN, Context
from fhl.pillars.convergence import MISSING, NO_OPINION, SCORED, ConvergenceEngine
from fhl.portfolio import Portfolio, Proposal
from fhl.prices import InMemoryPrices
from conftest import sec
from fin_builder import companyfacts, compounder, market_bars, price_bars, quarterly_dividends

AS_OF = date(2026, 9, 18)
SPLIT_DAY = date(2020, 6, 1)
DIVS = quarterly_dividends({y: 1.0 + 0.1 * (y - 2014) for y in range(2014, 2027)})  # rising every year
PF = Portfolio({}, 1_000_000)


def securities():
    s = {t: sec(t, "Industrials", cik=i + 1) for i, t in enumerate(
        ["GOOD", "PEER1", "PEER2", "FOREIGN", "LOSSCO", "SPLITCO", "YOUNG", "SILENT", "NOPAY"])}
    s["NOCIK"] = sec("NOCIK", "Industrials")
    s["BANK"] = sec("BANK", "Financials", cik=99)
    s["VTI"] = sec("VTI", "Diversified", type="etf")
    return s


def facts():
    loss = compounder()
    for r in loss:
        r["capex"] = r["operating_cash_flow"] * 3          # burns cash: free cash flow negative
    silent = compounder()
    for r in silent:
        r["skip"] = ("dividends_paid",)                    # pays, but doesn't tag the amount
    return {
        "GOOD": companyfacts(compounder()),
        "PEER1": companyfacts(compounder(margin=0.20)),
        "PEER2": companyfacts(compounder(margin=0.30)),
        "FOREIGN": companyfacts(compounder(), currency="EUR"),
        "LOSSCO": companyfacts(loss),
        "SPLITCO": companyfacts(compounder(split_on=SPLIT_DAY)),
        "YOUNG": companyfacts(compounder(years=range(2023, 2026))),
        "SILENT": companyfacts(silent),
        "NOPAY": companyfacts(compounder(payout=0.0)),
        "BANK": companyfacts(compounder()),
    }


DCF_ONLY = {"valuation.margin_of_safety_basis": "dcf"}


def engine(cfg, level=50.0, changes=None, levels=None):
    """Most tests measure the margin of safety on the DCF alone: in these fake
    price histories the price never changes level, so the own-history values
    would move with the price and hide what is being tested. The blend tests
    ask for it explicitly."""
    c = copy.deepcopy(cfg)
    for path, value in {**DCF_ONLY, **(changes or {})}.items():
        node = c["convergence"]
        *head, last = path.split(".")
        for k in head:
            node = node[k]
        node[last] = value
    levels = levels or {}
    bars = {t: price_bars(levels.get(t, level), dividends=None if t == "NOPAY" else DIVS)
            for t in securities() if t != "VTI"}
    bars["SPLITCO"] = price_bars(levels.get("SPLITCO", level), dividends=DIVS, splits={SPLIT_DAY: 2.0})
    bars["VTI"] = market_bars()
    ctx = Context(c, securities(), InMemoryPrices(bars), AS_OF,
                  fundamentals=InMemoryFundamentals(c, facts()), curve=None)
    return ConvergenceEngine({"name": "convergence", "weight": 25}, ctx)


def run(eng, ticker, action="buy"):
    return eng.evaluate(Proposal(ticker, action, 10_000, AS_OF), PF)


def blocks(res):
    return [r.code for r in res.fired if r.severity == BLOCK]


@pytest.fixture
def iv(cfg):
    """GOOD's intrinsic value per share (it barely depends on the price)."""
    return engine(cfg).analyze("GOOD").facts["intrinsic_value_per_share"]


# --- who gets scored ---------------------------------------------------------------

def test_etfs_get_no_opinion(cfg):
    res = run(engine(cfg), "VTI")
    assert res.score is None and blocks(res) == []
    assert res.details["analysis"].status == NO_OPINION


def test_banks_get_no_opinion(cfg):
    a = engine(cfg).analyze("BANK")
    assert a.status == NO_OPINION and "bank" in a.reason


def test_missing_data_blocks_a_buy_but_not_a_sell(cfg):
    eng = engine(cfg)
    assert blocks(run(eng, "NOCIK")) == ["missing_data"]
    assert blocks(run(eng, "NOCIK", "sell")) == []


def test_missing_data_can_be_no_opinion_instead(cfg):
    assert blocks(run(engine(cfg, changes={"missing_data": "no_opinion"}), "NOCIK")) == []


def test_too_few_years_of_filings(cfg):
    a = engine(cfg).analyze("YOUNG")
    assert a.status == MISSING and "only 3 years" in a.reason


# --- the scoring -------------------------------------------------------------------

def test_cheap_compounder_scores_high_and_passes(cfg, iv):
    eng = engine(cfg, level=0.5 * iv)
    res = run(eng, "GOOD")
    a = res.details["analysis"]
    assert a.status == SCORED and blocks(res) == []
    assert set(k for k, v in a.sources.items() if v is not None) == {"quality", "dividend", "value"}
    assert a.facts["margin_of_safety"] >= 0.25
    assert a.agreeing == 3 and a.factor == 1.0
    assert res.score >= 70


def test_same_business_too_expensive_is_blocked(cfg, iv):
    eng = engine(cfg, level=2.0 * iv)
    res = run(eng, "GOOD")
    assert blocks(res) == ["margin_of_safety"]
    assert blocks(run(eng, "GOOD", "sell")) == []                         # sells never blocked
    cheap = engine(cfg, level=0.5 * iv).analyze("GOOD")
    assert res.details["analysis"].sources["value"] < cheap.sources["value"]
    assert res.details["analysis"].sources["quality"] == pytest.approx(cheap.sources["quality"], abs=5)


def test_agreement_factor_is_applied(cfg, iv):
    a = engine(cfg, level=2.0 * iv).analyze("GOOD")
    avail = [x for x in a.sources.values() if x is not None]
    floor = cfg["convergence"]["agreement_floor"]
    assert a.score == pytest.approx(a.base * (floor + (1 - floor) * a.agreeing / len(avail)))
    assert a.agreeing < len(avail)                     # value disagrees, so the score is marked down
    assert a.score < a.base


def test_unmeasurable_margin_of_safety_blocks_by_default(cfg):
    res = run(engine(cfg), "LOSSCO")
    assert "margin_of_safety" in blocks(res)
    assert "free cash flow is not positive" in res.fired[0].message


def test_unmeasurable_margin_of_safety_can_warn_instead(cfg):
    res = run(engine(cfg, changes={"hard_rules.margin_of_safety_unknown": "warn"}), "LOSSCO")
    assert blocks(res) == [] and any(r.severity == WARN for r in res.fired)


def test_foreign_accounts_skip_value_and_say_why(cfg):
    a = engine(cfg).analyze("FOREIGN")
    assert a.sources["value"] is None
    assert any("EUR" in n for n in a.notes)
    assert "margin_of_safety" in blocks(run(engine(cfg), "FOREIGN"))


def test_needs_at_least_two_sources(cfg):
    a = engine(cfg, changes={"min_sources_scored": 3}).analyze("FOREIGN")
    assert a.status == MISSING and "2 of 3" in a.reason


def test_leverage_can_be_made_a_hard_stop(cfg, iv):
    res = run(engine(cfg, level=0.5 * iv, changes={"hard_rules.max_net_debt_to_ebitda": 0.1}), "GOOD")
    assert blocks(res) == ["leverage"]


# --- dividends ---------------------------------------------------------------------

def test_dividend_streak_counts_rising_years_from_prices(cfg):
    m = engine(cfg).analyze("GOOD").metrics["dividend"]["growth_streak"]
    assert m.value == 11                               # 2015..2025, each above the year before


def test_untagged_dividend_is_missing_not_zero(cfg):
    a = engine(cfg).analyze("SILENT")
    assert a.sources["dividend"] is None
    assert any("don't tag the amount" in n for n in a.notes)


def test_non_payer_is_skipped_by_default_or_zero_if_configured(cfg):
    a = engine(cfg).analyze("NOPAY")
    assert a.sources["dividend"] is None and any("pays no dividend" in n for n in a.notes)
    z = engine(cfg, changes={"non_payer_dividend": "zero", "min_metrics_per_source": 1}).analyze("NOPAY")
    assert z.sources["dividend"] == 0


# --- stock splits and point in time ------------------------------------------------

def test_split_does_not_distort_eps_growth_or_history(cfg):
    eng = engine(cfg)
    split, plain = eng.analyze("SPLITCO"), eng.analyze("GOOD")
    assert split.metrics["quality"]["eps_growth"].value == pytest.approx(
        plain.metrics["quality"]["eps_growth"].value, abs=1e-9)
    assert split.metrics["value"]["fcf_yield_vs_own"].value == pytest.approx(
        plain.metrics["value"]["fcf_yield_vs_own"].value, abs=1e-6)


def test_uses_only_reports_public_by_the_as_of_date(cfg):
    eng = engine(cfg)
    eng.ctx.as_of = date(2026, 2, 14)                  # the day before the 2025 report
    assert eng.analyze("GOOD").facts["latest_year"] == "2024-12-31"


# --- DCF settings --------------------------------------------------------------------

def test_fade_raises_value_for_a_business_growing_faster_than_terminal(cfg):
    plain = engine(cfg, changes={"dcf.fade_years": 0}).analyze("GOOD").facts["intrinsic_value_per_share"]
    faded = engine(cfg, changes={"dcf.fade_years": 10}).analyze("GOOD").facts["intrinsic_value_per_share"]
    assert faded > plain


def test_fcf_start_latest_uses_last_years_cash_flow(cfg):
    a = engine(cfg, changes={"dcf.fcf_start": "latest"}).analyze("GOOD")
    last = compounder()[-1]
    assert a.facts["fcf_start"] == pytest.approx(last["operating_cash_flow"] - last["capex"])
    avg = engine(cfg).analyze("GOOD").facts["fcf_start"]
    assert avg < a.facts["fcf_start"]                   # a growing business: average lags


def test_what_if_refuses_unknown_settings(cfg):
    from fhl.config import ConfigError
    from fhl.pillars.convergence import what_if
    for bad in ("dcf.fade_yeers", "valuaton.margin_of_safety_basis", "dcf.fade_years.deeper"):
        with pytest.raises(ConfigError, match="unknown setting"):
            what_if(cfg, {bad: 10})
    before = copy.deepcopy(cfg["convergence"])
    changed = what_if(cfg, {"dcf.fade_years": 3, "valuation.margin_of_safety_basis": "dcf"})
    assert changed["convergence"]["dcf"]["fade_years"] == 3
    assert changed["convergence"]["valuation"]["margin_of_safety_basis"] == "dcf"
    assert cfg["convergence"] == before                           # original untouched


def test_every_configured_variant_is_valid(cfg):
    from fhl.pillars.convergence import what_if
    codes = [v["code"] for v in cfg["convergence"]["valuation_variants"]]
    assert len(codes) == len(set(codes))
    for v in cfg["convergence"]["valuation_variants"]:
        what_if(cfg, v["changes"])


# --- option 3: margin of safety against a blended value ----------------------------

BLEND = {"valuation.margin_of_safety_basis": "blended"}


def test_blended_value_is_the_weighted_mix_of_every_method(cfg):
    a = engine(cfg, changes=BLEND).analyze("GOOD")
    vps = a.facts["value_per_share"]
    w = cfg["convergence"]["valuation"]["blend_weights"]
    methods = ["dcf", "own_ev_ebit", "own_fcf_yield", "sector_ev_ebit"]
    assert set(methods) <= set(vps)
    expected = sum(w[m] * vps[m] for m in methods) / sum(w[m] for m in methods)
    assert vps["blended"] == pytest.approx(expected)
    assert a.facts["margin_of_safety_basis"] == "blended"
    assert a.facts["margin_of_safety"] == pytest.approx(a.facts["mos_blended"])
    assert a.facts["margin_of_safety"] == pytest.approx(1 - a.facts["price"] / vps["blended"], abs=0.02)


def test_default_is_3c_blend_with_fade_10_dcf(cfg):
    assert cfg["convergence"]["valuation"]["margin_of_safety_basis"] == "blended"
    assert cfg["convergence"]["dcf"]["fade_years"] == 10
    three_c = next(v for v in cfg["convergence"]["valuation_variants"] if v["code"] == "3C")
    from fhl.pillars.convergence import what_if
    assert what_if(cfg, three_c["changes"])["convergence"] == cfg["convergence"]


def test_dcf_basis_is_unchanged_by_the_blend_settings(cfg):
    a = engine(cfg).analyze("GOOD")
    assert a.facts["margin_of_safety_basis"] == "dcf"
    assert a.facts["margin_of_safety"] == pytest.approx(a.facts["mos_dcf"])
    assert "mos_blended" in a.facts            # still computed, so it can be compared


def test_blend_can_value_a_company_without_free_cash_flow(cfg):
    a = engine(cfg, changes=BLEND).analyze("LOSSCO")
    vps = a.facts["value_per_share"]
    assert "dcf" not in vps and "own_fcf_yield" not in vps
    assert {"own_ev_ebit", "sector_ev_ebit", "blended"} <= set(vps)
    assert "can't verify" not in " ".join(r.message for r in run(engine(cfg, changes=BLEND), "LOSSCO").rules)


def test_too_few_methods_means_no_blended_value(cfg):
    res = run(engine(cfg, changes={**BLEND, "valuation.blend_min_methods": 5}), "GOOD")
    assert blocks(res) == ["margin_of_safety"]
    assert "no blended value" in res.fired[0].message


def test_unknown_basis_is_refused(cfg):
    from fhl.config import ConfigError
    with pytest.raises(ConfigError, match="margin_of_safety_basis"):
        engine(cfg, changes={"valuation.margin_of_safety_basis": "vibes"}).analyze("GOOD")


# --- robustness found by backtest 4 ---------------------------------------------

def test_stale_filings_are_not_scored(cfg):
    eng = engine(cfg)
    eng.ctx.as_of = date(2029, 6, 1)             # the latest report (2025) is now years old
    a = eng.analyze("GOOD")
    assert a.status == MISSING and "days old" in a.reason


def test_a_stock_without_prices_does_not_crash_the_engine(cfg):
    eng = engine(cfg)
    del eng.ctx.prices.data["GOOD"]               # filings but no price history at all
    a = eng.analyze("GOOD")
    assert a.status in (SCORED, MISSING)
    assert a.facts.get("margin_of_safety") is None
