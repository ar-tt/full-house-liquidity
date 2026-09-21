from datetime import date

import pytest

from fhl.cashflows import Schedule
from fhl.config import ConfigError
from fhl.curves import FlatCurve, load_fed_curve
from fhl.reserve import build_ladder, reserve_curve
from conftest import FIXTURES


def ladder(cfg, schedule, carve, rate, comp="annual"):
    return build_ladder(schedule, cfg["reserve"], carve, FlatCurve(rate, comp))


def annuity_due(pmt, rate, n):
    """Textbook formula, written independently of the engine."""
    return pmt * (1 - (1 + rate) ** -n) / rate * (1 + rate)


@pytest.mark.parametrize("rate, anchor", [(0.03, 439_305), (0.04, 421_767), (0.05, 405_391)])
def test_reserve_cost_matches_team_anchors(cfg, schedule, carve, rate, anchor):
    cost = ladder(cfg, schedule, carve, rate).cost
    assert round(cost) == anchor
    assert cost == pytest.approx(annuity_due(50_000, rate, 10))


def test_first_payment_is_not_discounted(cfg, schedule, carve):
    first = ladder(cfg, schedule, carve, 0.04).rungs[0]
    assert first.years_to_maturity == 0 and first.cost == 50_000


def test_zero_rate_means_reserve_equals_total_payments(cfg, schedule, carve):
    assert ladder(cfg, schedule, carve, 0.0).cost == pytest.approx(500_000)


def test_higher_rates_make_the_reserve_cheaper(cfg, schedule, carve):
    costs = [ladder(cfg, schedule, carve, r).cost for r in (0.02, 0.03, 0.04, 0.05, 0.06)]
    assert costs == sorted(costs, reverse=True)


def test_semiannual_quote_is_slightly_cheaper_than_annual(cfg, schedule, carve):
    """4% paid twice a year is really 4.04% a year, so the ladder costs a bit less."""
    annual = ladder(cfg, schedule, carve, 0.04).cost
    semi = ladder(cfg, schedule, carve, 0.04, "semiannual").cost
    assert semi < annual
    assert semi == pytest.approx(ladder(cfg, schedule, carve, 1.02 ** 2 - 1).cost)


def test_rungs_add_up_and_each_locks_the_right_yield(cfg, schedule, carve):
    lad = ladder(cfg, schedule, carve, 0.045)
    assert sum(r.cost for r in lad.rungs) == pytest.approx(lad.cost)
    assert lad.total_face == 500_000
    for r in lad.rungs[1:]:
        assert r.locked_yield == pytest.approx(0.045)


def test_composition_shares_sum_to_one(cfg, schedule, carve):
    comp = ladder(cfg, schedule, carve, 0.04).composition(cfg["reserve"]["maturity_buckets"])
    assert sum(comp.values()) == pytest.approx(1.0)
    assert comp["Paid today (cash)"] == pytest.approx(50_000 / annuity_due(50_000, 0.04, 10))


# --- Run-down: the reserve must pay every payment and end at exactly zero ---

@pytest.mark.parametrize("rate", [0.0, 0.03, 0.04, 0.05])
def test_runoff_pays_everything_and_ends_at_zero(cfg, schedule, carve, rate):
    rows = ladder(cfg, schedule, carve, rate).runoff(cfg["reserve"]["maturity_buckets"])
    assert len(rows) == 10
    assert all(r.payment == 50_000 for r in rows)
    assert rows[-1].value_after == pytest.approx(0, abs=1e-6)
    assert [r.rungs_left for r in rows] == list(range(9, -1, -1))
    # each year's leftover plus its growth is exactly next year's starting value
    for a, b in zip(rows, rows[1:]):
        assert a.value_after + a.interest_next_year == pytest.approx(b.value_before)


def test_runoff_value_equals_cost_of_remaining_payments(cfg, schedule, carve):
    """On any payment date, the reserve holds exactly what the remaining
    payments would cost to buy fresh at the same rate: it never runs short."""
    rows = ladder(cfg, schedule, carve, 0.04).runoff(cfg["reserve"]["maturity_buckets"])
    for k, r in enumerate(rows):
        assert r.value_before == pytest.approx(annuity_due(50_000, 0.04, 10 - k))


def test_runoff_gets_shorter_every_year(cfg, schedule, carve):
    rows = ladder(cfg, schedule, carve, 0.04).runoff(cfg["reserve"]["maturity_buckets"])
    avgs = [r.avg_years_left for r in rows]
    assert avgs == sorted(avgs, reverse=True)


def test_two_payments_on_one_date_are_both_paid(cfg, carve):
    cfg["cash_flows"].append({"date": date(2035, 1, 1), "amount": -10_000, "kind": "residency_payment"})
    lad = ladder(cfg, Schedule.from_config(cfg), carve, 0.04)
    rows = lad.runoff(cfg["reserve"]["maturity_buckets"])
    assert len(rows) == 10
    assert rows[2].payment == 60_000
    assert rows[-1].value_after == pytest.approx(0, abs=1e-6)


# --- Guard rails ---

def test_payment_before_carve_out_is_rejected(cfg, carve):
    cfg["cash_flows"].append({"date": date(2032, 1, 1), "amount": -50_000, "kind": "residency_payment"})
    with pytest.raises(ConfigError, match="before the reserve carve-out"):
        ladder(cfg, Schedule.from_config(cfg), carve, 0.04)


def test_curve_pricing_without_curve_is_rejected(cfg):
    with pytest.raises(ConfigError, match="needs the Fed Treasury curve"):
        reserve_curve(cfg, "curve_forward")


def test_unknown_pricing_method_rejected(cfg):
    with pytest.raises(ConfigError, match="must be one of"):
        reserve_curve(cfg, "vibes")


def test_curve_pricing_lands_in_a_sensible_range(cfg, schedule, carve):
    """With 2026 yields around 4.4-5.3%, the ladder should cost between the 5%
    and 4% flat anchors (roughly) and the forward price should be cheaper,
    because the curve slopes upward."""
    fed = load_fed_curve(FIXTURES / "gsw_sample.csv")
    spot = build_ladder(schedule, cfg["reserve"], carve, reserve_curve(cfg, "curve_spot", fed_curve=fed)).cost
    fwd = build_ladder(schedule, cfg["reserve"], carve, reserve_curve(cfg, "curve_forward", fed_curve=fed)).cost
    assert 395_000 < spot < 421_767
    assert fwd < spot
