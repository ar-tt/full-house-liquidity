from datetime import date

import pytest

from fhl.cashflows import Schedule
from fhl.config import ConfigError
from fhl.curves import FlatCurve
from fhl.projection import facility_contribution, project
from fhl.reserve import build_ladder


def test_seven_percent_anchor(schedule, carve):
    p = project(schedule, carve, 0.07)
    assert p.final_value == pytest.approx(300_000 * 1.07 ** 6 + 150_000 * 1.07 ** 5)
    assert round(p.final_value) == 660_602


def test_facility_anchor_is_238_835_not_238_837(cfg, schedule, carve):
    v = project(schedule, carve, 0.07).final_value
    reserve = build_ladder(schedule, cfg["reserve"], carve, FlatCurve(0.04)).cost
    assert round(facility_contribution(v, reserve).contribution) == 238_835


def test_zero_growth_leaves_only_the_contributions(schedule, carve):
    assert project(schedule, carve, 0.0).final_value == pytest.approx(450_000)


def test_year_by_year_path_is_applied_in_order(schedule, carve):
    path = {2027: 0.10, 2028: -0.20, 2029: 0.05, 2030: 0.0, 2031: 0.03, 2032: 0.08}
    expected = ((300_000 * 1.10 + 150_000) * 0.80) * 1.05 * 1.0 * 1.03 * 1.08
    assert project(schedule, carve, path).final_value == pytest.approx(expected)


def test_missing_year_in_path_is_an_error(schedule, carve):
    with pytest.raises(ConfigError, match="no return given for 2032"):
        project(schedule, carve, {y: 0.05 for y in range(2027, 2032)})


def test_rows_chain_together(schedule, carve):
    rows = project(schedule, carve, 0.06).rows
    assert [r.year for r in rows] == list(range(2027, 2033))
    for a, b in zip(rows, rows[1:]):
        assert b.value_start == pytest.approx(a.value_end)
        assert b.value_invested == pytest.approx(b.value_start + b.flows)


def test_payments_on_or_after_carve_out_are_not_in_the_projection(schedule, carve):
    assert all(r.flows >= 0 for r in project(schedule, carve, 0.07).rows)


def test_mid_year_cash_flow_is_rejected_not_silently_moved(cfg, carve):
    cfg["cash_flows"].append({"date": date(2029, 7, 1), "amount": 10_000, "kind": "contribution"})
    with pytest.raises(ConfigError, match="Jan 1"):
        project(Schedule.from_config(cfg), carve, 0.07)


def test_end_date_must_be_jan_first(schedule):
    with pytest.raises(ConfigError, match="must be a Jan 1"):
        project(schedule, date(2032, 6, 30), 0.07)


# --- The reserve comes first; the facility gets only what is left ---

def test_facility_gets_the_remainder_when_reserve_is_covered():
    fr = facility_contribution(500_000, 421_767)
    assert fr.fully_funded and fr.contribution == 78_233 and fr.shortfall == 0


def test_shortfall_gives_facility_zero_and_reports_the_gap():
    fr = facility_contribution(400_000, 421_767)
    assert not fr.fully_funded
    assert fr.contribution == 0
    assert fr.shortfall == 21_767


def test_a_bad_path_produces_a_shortfall(cfg, schedule, carve):
    v = project(schedule, carve, -0.03).final_value
    reserve = build_ladder(schedule, cfg["reserve"], carve, FlatCurve(0.04)).cost
    assert facility_contribution(v, reserve).shortfall > 0
