"""Correlation and 'effective number of bets', computed from daily returns.

Correlation (from -1 to 1) says how much two holdings move together. Two
stocks at 0.9 are nearly the same bet; at 0.1 they are nearly independent.

Effective number of bets: a portfolio of 30 stocks that all move together is
really ONE bet. We get the real count from the eigenvalues of the correlation
matrix: think of them as how the portfolio's total "movement" splits into
independent directions. If one direction carries almost everything, there is
about one bet; if it spreads evenly over n directions, there are n bets.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from .prices import PriceSource, daily_returns


class InsufficientHistory(Exception):
    def __init__(self, tickers: list[str], needed: int):
        self.tickers, self.needed = tickers, needed
        super().__init__(f"fewer than {needed} days of returns for: {', '.join(tickers)}")


def correlation_matrix(prices: PriceSource, tickers: list[str], as_of: date,
                       window: int) -> np.ndarray:
    """Correlation of the last `window` daily returns the tickers share, up to as_of."""
    rets, short = {}, []
    for t in tickers:
        try:
            rets[t] = daily_returns(prices.bars(t, as_of))
        except LookupError:  # the price source has nothing for this ticker
            rets[t] = {}
        if len(rets[t]) < window:
            short.append(t)
    if short:
        raise InsufficientHistory(short, window)
    if len(tickers) < 2:
        return np.eye(len(tickers))
    common = sorted(set.intersection(*(set(r) for r in rets.values())))[-window:]
    if len(common) < window:  # enough days each, but not enough days in common
        newest = max(tickers, key=lambda t: min(rets[t]))
        raise InsufficientHistory([newest], window)
    m = np.array([[rets[t][d] for d in common] for t in tickers])
    if np.any(m.std(axis=1) == 0):
        flat = [t for t, row in zip(tickers, m) if row.std() == 0]
        raise InsufficientHistory(flat, window)
    return np.corrcoef(m)


def average_pairwise(corr: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Average correlation over every pair of holdings.
    With weights, a pair counts in proportion to (size of one) x (size of other),
    so two tiny positions barely matter and two big ones matter a lot."""
    n = corr.shape[0]
    if n < 2:
        return 0.0
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    pair_w = np.outer(w, w)
    mask = ~np.eye(n, dtype=bool)
    return float((corr * pair_w)[mask].sum() / pair_w[mask].sum())


def effective_bets(corr: np.ndarray, method: str = "entropy") -> float:
    n = corr.shape[0]
    if n == 0:
        return 0.0
    lam = np.clip(np.linalg.eigvalsh(corr), 0, None)
    p = lam / lam.sum()
    if method == "participation":
        return float(1.0 / np.sum(p ** 2))
    if method == "entropy":
        nz = p[p > 0]
        return float(np.exp(-np.sum(nz * np.log(nz))))
    raise ValueError(f"unknown effective-bets method '{method}'")
