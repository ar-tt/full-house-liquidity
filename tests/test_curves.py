from datetime import date

import pytest

from fhl.curves import FlatCurve, ForwardCurve, load_fed_curve, year_fraction
from conftest import FIXTURES

# Zero yields the Fed published for 2026-09-11 (SVENYxx columns, percent)
FED_ZEROS_2026_09_11 = {1: 4.3608, 2: 4.5943, 5: 4.7310, 10: 4.9735, 20: 5.4512, 30: 5.3962}


def test_year_fraction_is_whole_years_between_jan_firsts():
    assert year_fraction(date(2033, 1, 1), date(2042, 1, 1)) == 9
    assert year_fraction(date(2033, 1, 1), date(2033, 1, 1)) == 0
    assert year_fraction(date(2026, 7, 2), date(2027, 1, 1)) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("comp, expected", [
    ("annual", 1 / 1.04 ** 3),
    ("semiannual", 1 / 1.02 ** 6),
    ("continuous", 2.718281828459045 ** (-0.12)),
])
def test_flat_curve_compounding(comp, expected):
    assert FlatCurve(0.04, comp).discount_factor(3) == pytest.approx(expected)


def test_unknown_compounding_rejected():
    with pytest.raises(ValueError):
        FlatCurve(0.04, "monthly")


def test_fed_curve_formula_reproduces_published_zero_yields():
    curve = load_fed_curve(FIXTURES / "gsw_sample.csv")
    assert curve.as_of == date(2026, 9, 11)
    for t, pct in FED_ZEROS_2026_09_11.items():
        assert curve.zero_rate(t) * 100 == pytest.approx(pct, abs=0.0005)


def test_fed_curve_is_point_in_time():
    """Asking 'as of' an earlier date must never return a later curve."""
    curve = load_fed_curve(FIXTURES / "gsw_sample.csv", as_of=date(2026, 9, 10))
    assert curve.as_of == date(2026, 9, 10)
    curve = load_fed_curve(FIXTURES / "gsw_sample.csv", as_of=date(1961, 6, 14))
    assert curve.as_of == date(1961, 6, 14)


def test_forward_of_a_flat_curve_is_the_same_flat_curve():
    flat = FlatCurve(0.04)
    fwd = ForwardCurve(flat, offset=6.3)
    for t in (0.5, 1, 5, 9):
        assert fwd.discount_factor(t) == pytest.approx(flat.discount_factor(t))
