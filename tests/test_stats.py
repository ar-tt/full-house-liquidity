from datetime import date

import numpy as np
import pytest

from fhl.prices import Bar, InMemoryPrices, daily_returns
from fhl.stats import InsufficientHistory, average_pairwise, correlation_matrix, effective_bets
from conftest import bars_from_returns, factor_prices, weekdays


# --- effective number of bets -------------------------------------------------

@pytest.mark.parametrize("method", ["entropy", "participation"])
def test_independent_holdings_are_separate_bets(method):
    assert effective_bets(np.eye(6), method) == pytest.approx(6)


@pytest.mark.parametrize("method", ["entropy", "participation"])
def test_identical_holdings_are_one_bet(method):
    assert effective_bets(np.ones((6, 6)), method) == pytest.approx(1)


@pytest.mark.parametrize("method", ["entropy", "participation"])
def test_two_clusters_are_two_bets(method):
    block = np.ones((3, 3))
    corr = np.block([[block, np.zeros((3, 3))], [np.zeros((3, 3)), block]])
    assert effective_bets(corr, method) == pytest.approx(2)


def test_more_correlation_means_fewer_bets():
    def c(rho):
        m = np.full((10, 10), rho)
        np.fill_diagonal(m, 1)
        return m
    bets = [effective_bets(c(r)) for r in (0.0, 0.2, 0.5, 0.8)]
    assert bets == sorted(bets, reverse=True)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        effective_bets(np.eye(2), "vibes")


# --- average pairwise correlation ---------------------------------------------

def test_average_pairwise_equal_weights():
    corr = np.array([[1, .5, .1], [.5, 1, .3], [.1, .3, 1]])
    assert average_pairwise(corr) == pytest.approx((.5 + .1 + .3) / 3)


def test_average_pairwise_weighted_counts_big_pairs_more():
    corr = np.array([[1, .9, 0], [.9, 1, 0], [0, 0, 1]])
    # the two big holdings are the correlated pair, so the weighted average is high
    big_pair = average_pairwise(corr, np.array([10, 10, 1]))
    assert big_pair == pytest.approx((.9 * 100) / (100 + 10 + 10))
    assert big_pair > average_pairwise(corr)


def test_single_holding_has_no_pairs():
    assert average_pairwise(np.eye(1)) == 0.0


# --- correlation from prices --------------------------------------------------

def test_correlation_matches_how_the_prices_were_built():
    prices = factor_prices({"a": (0.8, ["A1", "A2", "A3"]), "b": (0.0, ["B1"])}, n_days=2000)
    corr = correlation_matrix(prices, ["A1", "A2", "A3", "B1"], date(2040, 1, 1), 1500)
    assert corr[0, 1] == pytest.approx(0.8, abs=0.05)
    assert abs(corr[0, 3]) < 0.08


def test_mirror_image_stocks_are_minus_one():
    r = list(np.random.default_rng(1).normal(0, 0.01, 80))
    prices = InMemoryPrices({"X": bars_from_returns(r), "Y": bars_from_returns([-x for x in r])})
    assert correlation_matrix(prices, ["X", "Y"], date(2040, 1, 1), 60)[0, 1] == pytest.approx(-1)


def test_correlation_is_point_in_time():
    """Prices after the as-of date must be ignored: the first 70 days move
    together, the rest move opposite. As of day 70 the answer is +1."""
    rng = np.random.default_rng(2)
    a = list(rng.normal(0, .01, 140))
    b = a[:70] + [-x for x in a[70:]]
    prices = InMemoryPrices({"A": bars_from_returns(a), "B": bars_from_returns(b)})
    day70 = weekdays(71)[-1]
    assert correlation_matrix(prices, ["A", "B"], day70, 60)[0, 1] == pytest.approx(1)
    assert correlation_matrix(prices, ["A", "B"], date(2040, 1, 1), 60)[0, 1] == pytest.approx(-1)


def test_short_history_is_reported_by_ticker():
    prices = factor_prices({"a": (0.3, ["OLD"])}, n_days=100)
    prices.data["NEW"] = bars_from_returns([0.01, -0.01] * 10)
    with pytest.raises(InsufficientHistory) as e:
        correlation_matrix(prices, ["OLD", "NEW"], date(2040, 1, 1), 60)
    assert e.value.tickers == ["NEW"]


def test_unknown_ticker_counts_as_missing_history():
    class Broken(InMemoryPrices):
        def bars(self, ticker, as_of):
            if ticker == "GONE":
                raise LookupError("no data")
            return super().bars(ticker, as_of)
    prices = Broken(factor_prices({"a": (0.3, ["A"])}).data)
    with pytest.raises(InsufficientHistory) as e:
        correlation_matrix(prices, ["A", "GONE"], date(2040, 1, 1), 60)
    assert e.value.tickers == ["GONE"]


def test_daily_returns_add_back_dividends():
    d = weekdays(2)
    bars = [Bar(d[0], 100, 100, 100, 100, 1), Bar(d[1], 99, 99, 99, 99, 1, dividend=1.0)]
    assert daily_returns(bars)[d[1]] == pytest.approx(0.0)
