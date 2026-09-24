"""Reading filings point in time, and the business/valuation math."""
from datetime import date

import pytest

from fhl.fundamentals import metrics as M
from fhl.fundamentals.annual import build_financials
from fin_builder import add_fact, companyfacts, compounder


@pytest.fixture
def fcfg(cfg):
    return cfg["fundamentals"]


# --- reading filings -------------------------------------------------------------

def test_one_row_per_fiscal_year_with_every_concept(fcfg):
    fin = build_financials(companyfacts(compounder()), fcfg, date(2026, 9, 20))
    assert [fy.end.year for fy in fin.years] == list(range(2014, 2026))
    assert fin.currency == "USD"
    assert fin.years[-1].get("revenue") == pytest.approx(1000 * 1.08 ** 11)
    assert fin.years[-1].available == date(2026, 2, 15)


def test_numbers_filed_after_the_as_of_date_are_invisible(fcfg):
    fin = build_financials(companyfacts(compounder()), fcfg, date(2025, 2, 14))
    assert fin.years[-1].end.year == 2023          # the 2024 report came out Feb 15, 2025


def test_restatement_counts_only_once_it_was_filed(fcfg):
    facts = companyfacts(compounder(years=range(2020, 2025)))
    add_fact(facts, "Revenues", "USD", "2023-12-31", 9999.0, "2025-02-15", "2023-01-01")  # restated later
    before = build_financials(facts, fcfg, date(2025, 1, 1))
    after = build_financials(facts, fcfg, date(2025, 3, 1))
    assert before.years[-1].get("revenue") == pytest.approx(1000 * 1.08 ** 3)
    assert after.years[-2].get("revenue") == 9999.0
    assert after.years[-2].available == date(2024, 2, 15)   # first public date doesn't move


def test_tags_fall_back_year_by_year(fcfg):
    rows = compounder(years=range(2016, 2022))
    for r in rows[:3]:
        r["skip"] = ("revenue",)
    facts = companyfacts(rows)
    for r in rows[:3]:  # the early years used an older tag
        add_fact(facts, "SalesRevenueNet", "USD", f"{r['year']}-12-31", 111.0, f"{r['year'] + 1}-02-15",
                 f"{r['year']}-01-01")
    fin = build_financials(facts, fcfg, date(2026, 1, 1))
    assert [fy.get("revenue") for fy in fin.years[:3]] == [111.0] * 3
    assert fin.years[3].get("revenue") == pytest.approx(1000 * 1.08 ** 3)


def test_alternatives_add_their_extra_tags(fcfg):
    facts = companyfacts(compounder(years=range(2020, 2023)))
    add_fact(facts, "CommercialPaper", "USD", "2022-12-31", 70.0, "2023-02-15")
    fin = build_financials(facts, fcfg, date(2024, 1, 1))
    assert fin.years[-1].get("total_debt") == 570.0
    assert fin.years[-2].get("total_debt") == 500.0         # no commercial paper that year


def test_missing_debt_means_zero_but_missing_dividend_stays_unknown(fcfg):
    rows = compounder(years=range(2020, 2023))
    for r in rows:
        r["skip"] = ("total_debt", "dividends_paid")
    fin = build_financials(companyfacts(rows), fcfg, date(2024, 1, 1))
    assert fin.years[-1].get("total_debt") == 0.0
    assert fin.years[-1].get("dividends_paid") is None


def test_quarterly_facts_are_not_mistaken_for_years(fcfg):
    facts = companyfacts(compounder(years=range(2020, 2023)))
    add_fact(facts, "Revenues", "USD", "2022-09-30", 1.0, "2022-11-01", "2022-07-01", form="10-Q")
    add_fact(facts, "Revenues", "USD", "2022-12-31", 2.0, "2023-02-15", "2022-10-01")   # a quarter inside a 10-K
    fin = build_financials(facts, fcfg, date(2024, 1, 1))
    assert fin.years[-1].get("revenue") == pytest.approx(1000 * 1.08 ** 2)


def test_foreign_currency_is_detected(fcfg):
    fin = build_financials(companyfacts(compounder(years=range(2020, 2023)), currency="EUR"), fcfg, date(2024, 1, 1))
    assert fin.currency == "EUR" and fin.years[-1].get("eps_diluted") is not None


def test_newest_share_count_skips_a_tag_the_company_stopped_using(fcfg):
    facts = companyfacts(compounder(years=range(2020, 2023)))
    for r in facts["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"]:
        r["end"], r["filed"] = "2012-12-31", "2013-02-15"          # stale since 2013
    add_fact(facts, "WeightedAverageNumberOfSharesOutstandingBasic", "shares", "2026-06-30", 97.0,
             "2026-08-01", "2026-04-01", form="10-Q")
    fin = build_financials(facts, fcfg, date(2026, 9, 1))
    assert fin.latest_shares == (date(2026, 8, 1), 97.0)


# --- the math --------------------------------------------------------------------

def test_roic_by_hand(fcfg):
    fin = build_financials(companyfacts(compounder(years=range(2020, 2022))), fcfg, date(2023, 1, 1))
    (_, roic), = M.roic_series(fin, 0.21, 0.6)
    op = 1000 * 1.08 * 0.25
    ic = ((2000 + 500 - 200) + (2060 + 500 - 200)) / 2
    assert roic == pytest.approx(op * 0.79 / ic)


def test_negative_invested_capital_counts_as_the_cap(fcfg):
    rows = compounder(years=range(2020, 2022))
    for r in rows:
        r["equity"] = -5000.0
    fin = build_financials(companyfacts(rows), fcfg, date(2023, 1, 1))
    assert M.roic_series(fin, 0.21, 0.6)[0][1] == 0.6


def test_growth_rate_exact_for_steady_growth():
    assert M.growth_rate([100 * 1.07 ** i for i in range(10)]) == pytest.approx(0.07)


def test_growth_rate_ignores_one_freak_year():
    vals = [100 * 1.07 ** i for i in range(10)]
    vals[1] = 20.0                                   # e.g. a one-off tax charge
    assert M.growth_rate(vals) == pytest.approx(0.07, abs=0.005)


def test_growth_rate_with_losses_falls_back_to_averages():
    # first 3 average 25/3, last 3 average 40, three years apart
    assert M.growth_rate([-5, 10, 20, 30, 40, 50]) == pytest.approx((40 / (25 / 3)) ** (1 / 3) - 1)
    assert M.growth_rate([10, 5, -1, -2, -3, -4]) is None
    assert M.growth_rate([1, 2]) is None


def test_accrual_ratio_by_hand(fcfg):
    rows = compounder(years=range(2020, 2022))
    rows[-1]["operating_cash_flow"] = rows[-1]["net_income"] - 310.0
    fin = build_financials(companyfacts(rows), fcfg, date(2023, 1, 1))
    assert M.accrual_ratio(fin) == pytest.approx(310 / 3050)


def test_ebit_falls_back_to_pretax_plus_interest(fcfg):
    rows = compounder(years=range(2020, 2022))
    for r in rows:
        r["skip"] = ("operating_income",)
    fy = build_financials(companyfacts(rows), fcfg, date(2023, 1, 1)).years[-1]
    assert M.ebit(fy) == pytest.approx(fy.get("pretax_income") + fy.get("interest_expense"))


def test_dividend_streak():
    d = lambda y: date(y, 6, 1)
    divs = [(d(2018), 1.0), (d(2019), 1.0), (d(2020), 1.1), (d(2021), 1.2), (d(2022), 1.3), (d(2023), 1.4)]
    assert M.dividend_streak(divs, date(2024, 3, 1)) == 4            # 2020..2023 each higher; 2019 flat
    assert M.dividend_streak(divs + [(d(2024), 9.9)], date(2024, 9, 1)) == 4   # 2024 not complete yet
    assert M.dividend_streak(divs, date(2026, 3, 1)) == 0            # stopped paying after 2023
    assert M.dividend_streak([], date(2024, 1, 1)) == 0


def test_dcf_matches_the_textbook_formula():
    # no growth, terminal 0: a perpetuity worth fcf / r
    assert M.dcf_value(100, 0.0, 0.10, 10, 0.0) == pytest.approx(1000)
    with pytest.raises(ValueError):
        M.dcf_value(100, 0.05, 0.02, 10, 0.03)


def test_reverse_dcf_recovers_the_growth_it_was_built_with():
    ev = M.dcf_value(100, 0.093, 0.085, 10, 0.025)
    assert M.implied_growth(ev, 100, 0.085, 10, 0.025, -0.2, 0.5) == pytest.approx(0.093, abs=1e-6)
    assert M.implied_growth(ev, -5, 0.085, 10, 0.025, -0.2, 0.5) is None
    assert M.implied_growth(1e12, 100, 0.085, 10, 0.025, -0.2, 0.5) == 0.5    # capped at the search range


def test_position_in_range():
    assert M.position_in_range(15, [10, 20]) == 0.5
    assert M.position_in_range(25, [10, 20]) == 1.0
    assert M.position_in_range(5, [10]) is None


def test_beta_with_blume_adjustment():
    m = {(2020, w): x for w, x in enumerate([0.01, -0.02, 0.015, -0.005] * 30)}
    s = {k: 1.5 * v for k, v in m.items()}
    assert M.beta(s, m, 0.33, (0.5, 2.5)) == pytest.approx(0.67 * 1.5 + 0.33)
    assert M.beta(s, m, 0.0, (0.5, 1.2)) == 1.2
    assert M.beta({k: v for k, v in list(s.items())[:50]}, m, 0.33, (0.5, 2.5)) is None


# --- growth fade -------------------------------------------------------------------

def test_growth_path_steps_down_evenly():
    assert M.growth_path(0.10, 2, 0.02, fade_years=3) == pytest.approx([0.10, 0.10, 0.08, 0.06, 0.04])
    assert M.growth_path(0.10, 2, 0.02) == [0.10, 0.10]


def test_fade_by_hand():
    # year 1 grows 10% -> 110; year 2 (fade) grows 5% -> 115.5; then 0% forever at 10%
    expected = 110 / 1.1 + 115.5 / 1.1 ** 2 + (115.5 / 0.10) / 1.1 ** 2
    assert M.dcf_value(100, 0.10, 0.10, 1, 0.0, fade_years=1) == pytest.approx(expected)


def test_fade_changes_nothing_when_growth_already_equals_terminal():
    assert M.dcf_value(100, 0.03, 0.09, 10, 0.03, fade_years=10) == pytest.approx(M.dcf_value(100, 0.03, 0.09, 10, 0.03))


def test_longer_fade_helps_fast_growers_and_hurts_slow_ones():
    fast = [M.dcf_value(100, 0.12, 0.09, 10, 0.025, f) for f in (0, 5, 10)]
    slow = [M.dcf_value(100, 0.01, 0.09, 10, 0.025, f) for f in (0, 5, 10)]
    assert fast == sorted(fast) and slow == sorted(slow, reverse=True)


def test_reverse_dcf_round_trip_with_fade():
    ev = M.dcf_value(100, 0.11, 0.09, 10, 0.025, fade_years=10)
    assert M.implied_growth(ev, 100, 0.09, 10, 0.025, -0.2, 0.5, fade_years=10) == pytest.approx(0.11, abs=1e-6)
