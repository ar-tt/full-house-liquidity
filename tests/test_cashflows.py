from datetime import date

import pytest

from fhl.cashflows import Schedule
from fhl.config import ConfigError


def test_schedule_matches_the_brief(schedule):
    assert schedule.total_contributed() == 450_000
    payments = schedule.of_kind("residency_payment")
    assert len(payments) == 10
    assert all(p.amount == -50_000 for p in payments)
    assert [p.date for p in payments] == [date(y, 1, 1) for y in range(2033, 2043)]


def test_rows_are_sorted_whatever_order_config_uses(cfg):
    cfg["cash_flows"].reverse()
    s = Schedule.from_config(cfg)
    assert [r.date for r in s.rows] == sorted(r.date for r in s.rows)


def test_schedule_is_rows_not_a_formula(cfg):
    """Adding one row changes the totals: nothing assumes ten payments."""
    cfg["cash_flows"].append({"date": date(2043, 1, 1), "amount": -50_000, "kind": "residency_payment"})
    s = Schedule.from_config(cfg)
    assert len(s.of_kind("residency_payment")) == 11
    assert s.total_paid_out() == 550_000


@pytest.mark.parametrize("bad_row, message", [
    ({"amount": 1, "kind": "contribution"}, "missing 'date'"),
    ({"date": date(2027, 1, 1), "kind": "contribution"}, "missing 'amount'"),
    ({"date": "2027", "amount": 1, "kind": "contribution"}, "YYYY-MM-DD"),
    ({"date": date(2027, 1, 1), "amount": 0, "kind": "contribution"}, "zero"),
])
def test_bad_rows_are_rejected_with_a_clear_message(cfg, bad_row, message):
    cfg["cash_flows"].append(bad_row)
    with pytest.raises(ConfigError, match=message):
        Schedule.from_config(cfg)
