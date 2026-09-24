"""The long hand: calendar bands, funded status and the hedge ramp."""
import copy
from datetime import date

import pytest

from fhl.cashflows import Schedule
from fhl.curves import FlatCurve
from fhl.wind_down import (band_on, hedge_share_on, lock_in_cost, projected_reserve_cost,
                           projected_value, pv_future_contributions, status)

FLAT4 = FlatCurve(0.04)
RESERVE_AT_4 = 421_766.58          # the team's anchor
RECEIVED = date(2028, 1, 1)        # both contributions already in the portfolio


@pytest.fixture
def flat_cfg(cfg):
    """Expected returns of zero, so projected value = today's value and the
    funded status is exactly value / reserve cost: easy to steer in tests."""
    c = copy.deepcopy(cfg)
    c["wind_down"]["expected_return_growth"] = 0.0
    c["wind_down"]["expected_return_safe"] = 0.0
    c["wind_down"]["hedge_ramp"] = FULL_RAMP        # ramp mechanics tested on round numbers
    c["wind_down"]["risk_dial"] = 1.0               # the rules themselves, before the dial
    return c


FULL_RAMP = [{"from": date(2030, 1, 1), "share": 0.33}, {"from": date(2031, 1, 1), "share": 0.67},
             {"from": date(2032, 1, 1), "share": 1.00}]


def run(cfg, schedule, value, as_of, growth_share=0.8):
    return status(value, growth_share * value, schedule, cfg, as_of, FLAT4, RECEIVED)


# --- calendar -----------------------------------------------------------------

@pytest.mark.parametrize("d, band", [
    (date(2026, 9, 1), (0.85, 0.90)),
    (date(2029, 12, 31), (0.85, 0.90)),
    (date(2030, 1, 1), (0.70, 0.80)),
    (date(2031, 12, 31), (0.70, 0.80)),
    (date(2032, 1, 1), (0.50, 0.60)),
    (date(2033, 1, 1), (0.0, 0.0)),
    (date(2020, 1, 1), (0.85, 0.90)),           # before the first row: the first row
])
def test_calendar_bands_switch_on_their_dates(cfg, d, band):
    assert band_on(cfg, d) == band


@pytest.mark.parametrize("d, share", [
    (date(2029, 12, 31), 0.0), (date(2030, 1, 1), 0.33), (date(2031, 6, 1), 0.67), (date(2032, 1, 1), 1.0)])
def test_live_hedge_ramp_is_the_full_one(cfg, d, share):
    assert hedge_share_on(cfg, d) == share


# --- money measures -------------------------------------------------------------

def test_lock_in_cost_on_carve_out_day_is_the_reserve_anchor(cfg, schedule):
    assert lock_in_cost(schedule, cfg, date(2033, 1, 1), FLAT4) == pytest.approx(RESERVE_AT_4, abs=1)
    assert lock_in_cost(schedule, cfg, date(2027, 1, 1), FlatCurve(0.0)) == pytest.approx(500_000)


def test_projected_reserve_cost_is_the_forward_price(cfg, schedule):
    # on a flat curve the forward price of the 2033 reserve is the same 4% price
    assert projected_reserve_cost(schedule, cfg, date(2027, 6, 30), FLAT4) == pytest.approx(RESERVE_AT_4, abs=1)


def test_future_contributions_count_until_received(cfg, schedule):
    assert pv_future_contributions(schedule, date(2027, 6, 30), FlatCurve(0.0)) == 150_000
    assert pv_future_contributions(schedule, date(2027, 6, 30), FlatCurve(0.0), RECEIVED) == 0


def test_projected_value_follows_expected_returns_and_contributions(flat_cfg, schedule):
    assert projected_value(300_000, schedule, flat_cfg, date(2027, 1, 1)) == pytest.approx(450_000)
    c = copy.deepcopy(flat_cfg)
    c["wind_down"]["expected_return_growth"] = c["wind_down"]["expected_return_safe"] = 0.05
    got = projected_value(300_000, schedule, c, date(2027, 1, 1))
    assert got == pytest.approx(300_000 * 1.05 ** 6 + 150_000 * 1.05 ** 5)


# --- funded status picks the spot ------------------------------------------------

AS_OF = date(2027, 6, 30)


def test_behind_stays_at_the_ceiling(flat_cfg, schedule):
    st = run(flat_cfg, schedule, 1.2 * RESERVE_AT_4, AS_OF)
    assert st.target == 0.90 and "behind" in st.reasons[0]


def test_on_track_slides_through_the_band(flat_cfg, schedule):
    st = run(flat_cfg, schedule, 1.6 * RESERVE_AT_4, AS_OF)            # halfway from 1.30 to 1.90
    assert st.funded_status == pytest.approx(1.6, abs=1e-3)
    assert st.target == pytest.approx(0.875, abs=1e-3)


def test_ahead_locks_the_whole_reserve(flat_cfg, schedule):
    v = 2.0 * RESERVE_AT_4
    st = run(flat_cfg, schedule, v, AS_OF)
    assert st.target == pytest.approx(1 - lock_in_cost(schedule, flat_cfg, AS_OF, FLAT4) / v)
    assert st.target < 0.85                                            # below the band: allowed
    assert "lock the whole reserve" in st.reasons[0]


@pytest.mark.parametrize("fs", [0.3, 0.9, 1.2, 1.5, 1.8, 2.5, 10.0])
def test_never_above_the_ceiling_never_below_zero(flat_cfg, schedule, fs):
    for d in (date(2027, 3, 1), date(2030, 6, 1), date(2031, 6, 1), date(2032, 6, 1)):
        st = run(flat_cfg, schedule, fs * RESERVE_AT_4, d)
        assert 0 <= st.target <= band_on(flat_cfg, d)[1] + 1e-12


def test_target_falls_when_the_calendar_steps_down(flat_cfg, schedule):
    before = run(flat_cfg, schedule, 1.6 * RESERVE_AT_4, date(2029, 12, 31))
    after = run(flat_cfg, schedule, 1.6 * RESERVE_AT_4, date(2030, 1, 1))
    assert after.target < before.target


def test_nothing_in_growth_after_the_carve_out(flat_cfg, schedule):
    st = run(flat_cfg, schedule, 1_000_000, date(2033, 1, 1))
    assert st.target == 0 and st.ceiling == 0


# --- hedge ramp -----------------------------------------------------------------

def test_hedge_ramp_can_push_below_the_band(flat_cfg, schedule):
    d = date(2032, 6, 30)
    v = 1.6 * RESERVE_AT_4
    st = run(flat_cfg, schedule, v, d)
    assert st.target == pytest.approx(1 - lock_in_cost(schedule, flat_cfg, d, FLAT4) / v)
    assert st.target < 0.50 and "hedge ramp" in st.reasons[-1]


def test_partial_hedge_in_2030(flat_cfg, schedule):
    d = date(2030, 6, 30)
    v = 1.35 * RESERVE_AT_4                     # near "behind": calendar alone says ~79%
    st = run(flat_cfg, schedule, v, d)
    cap = 1 - 0.33 * lock_in_cost(schedule, flat_cfg, d, FLAT4) / v
    assert st.target == pytest.approx(min(cap, st.target))
    assert st.target <= cap + 1e-12


def test_hedge_ramp_waived_when_it_would_lock_in_a_shortfall(flat_cfg, schedule):
    st = run(flat_cfg, schedule, 300_000, date(2032, 6, 30))    # can't afford the reserve even all in bonds
    assert st.lock_ratio < 1
    assert st.target == 0.60 and "waived" in st.reasons[-1]


def test_action_text(flat_cfg, schedule):
    st = run(flat_cfg, schedule, 1.6 * RESERVE_AT_4, AS_OF, growth_share=0.95)
    assert st.action(0.02).startswith("over target: move $")
    st = run(flat_cfg, schedule, 1.6 * RESERVE_AT_4, AS_OF, growth_share=0.87)
    assert st.action(0.02) == "on target"


def test_live_risk_dial_scales_the_target(cfg, flat_cfg, schedule):
    assert cfg["wind_down"]["risk_dial"] == 0.75
    dialled = copy.deepcopy(flat_cfg)
    dialled["wind_down"]["risk_dial"] = 0.75
    for d in (AS_OF, date(2031, 6, 30)):
        plain = run(flat_cfg, schedule, 1.6 * RESERVE_AT_4, d)
        scaled = run(dialled, schedule, 1.6 * RESERVE_AT_4, d)
        assert scaled.target == pytest.approx(0.75 * plain.target)
        assert "risk dial" in scaled.reasons[-1]

