"""Price indicators for the Position Clock, computed from raw daily bars.

  200-day moving average  the average closing price over 200 trading days.
                          Price above it = long-term uptrend.
  ADX (14)                how STRONG the current trend is (up or down), 0-100.
                          Above 20 = a real trend; at or below 20 = drifting.
  RSI (14)                how stretched recent moves are, 0-100. Under 40 =
                          sold off (a better moment to buy); over 70 = run up.
  60-day volatility       how much the price has been swinging, per year.

RSI and ADX use J. Welles Wilder's original smoothing (1978), the standard
definition charting sites use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .prices import Bar, daily_returns


def sma(closes: list[float], n: int) -> float | None:
    return float(np.mean(closes[-n:])) if len(closes) >= n else None


def rsi(closes: list[float], n: int = 14) -> float | None:
    if len(closes) < n + 1:
        return None
    diffs = np.diff(closes)
    gains, losses = np.clip(diffs, 0, None), np.clip(-diffs, 0, None)
    avg_g, avg_l = gains[:n].mean(), losses[:n].mean()
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 50.0 if avg_g == 0 else 100.0
    return float(100 - 100 / (1 + avg_g / avg_l))


def adx(bars: list[Bar], n: int = 14) -> float | None:
    if len(bars) < 2 * n + 1:
        return None
    tr, pdm, mdm = [], [], []
    for prev, cur in zip(bars, bars[1:]):
        tr.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
        up, down = cur.high - prev.high, prev.low - cur.low
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
    s_tr, s_p, s_m = sum(tr[:n]), sum(pdm[:n]), sum(mdm[:n])
    dxs = []
    for i in range(n, len(tr) + 1):
        if i > n:
            s_tr = s_tr - s_tr / n + tr[i - 1]
            s_p = s_p - s_p / n + pdm[i - 1]
            s_m = s_m - s_m / n + mdm[i - 1]
        if s_tr == 0:
            dxs.append(0.0)
            continue
        p_di, m_di = 100 * s_p / s_tr, 100 * s_m / s_tr
        dxs.append(0.0 if p_di + m_di == 0 else 100 * abs(p_di - m_di) / (p_di + m_di))
    if len(dxs) < n:
        return None
    a = float(np.mean(dxs[:n]))
    for dx in dxs[n:]:
        a = (a * (n - 1) + dx) / n
    return a


def realized_vol(bars: list[Bar], n: int = 60) -> float | None:
    rets = list(daily_returns(bars).values())[-n:]
    if len(rets) < n:
        return None
    logs = [math.log(1 + r) for r in rets]
    return float(np.std(logs, ddof=1) * math.sqrt(252))


@dataclass
class Snapshot:
    close: float
    ma: float | None
    adx: float | None
    rsi: float | None
    vol: float | None
    regime: str               # uptrend | downtrend | no_trend
    notes: list


def snapshot(bars: list[Bar], icfg: dict) -> Snapshot | None:
    if len(bars) < 2:
        return None
    closes = [b.close for b in bars]
    ma = sma(closes, icfg["trend_ma_days"])
    a = adx(bars, icfg["adx_days"])
    notes = []
    if ma is None or a is None:
        regime = "no_trend"
        notes.append(f"under {icfg['trend_ma_days']} days of prices: trend unknown, treated as no trend")
    elif a <= icfg["adx_trending_above"]:
        regime = "no_trend"
    else:
        regime = "uptrend" if closes[-1] > ma else "downtrend"
    return Snapshot(closes[-1], ma, a, rsi(closes, icfg["rsi_days"]),
                    realized_vol(bars, icfg["vol_days"]), regime, notes)


# ------------------------------------------------------------------------------
# One value per day, in a single pass (for backtests). Each entry i uses only
# bars up to and including day i, and equals the single-day function above.
# ------------------------------------------------------------------------------

def sma_series(closes, n: int) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        cs = np.cumsum(np.insert(c, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def rsi_series(closes, n: int = 14) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) < n + 1:
        return out
    d = np.diff(c)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = g[:n].mean(), l[:n].mean()

    def value(ag, al):
        if al == 0:
            return 50.0 if ag == 0 else 100.0
        return 100 - 100 / (1 + ag / al)
    out[n] = value(ag, al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        out[i + 1] = value(ag, al)
    return out


def adx_series(bars: list, n: int = 14) -> np.ndarray:
    out = np.full(len(bars), np.nan)
    if len(bars) < 2 * n + 1:
        return out
    tr, pdm, mdm = [], [], []
    for prev, cur in zip(bars, bars[1:]):
        tr.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
        up, down = cur.high - prev.high, prev.low - cur.low
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
    s_tr, s_p, s_m = sum(tr[:n]), sum(pdm[:n]), sum(mdm[:n])
    dxs = []
    for i in range(n, len(tr) + 1):          # dx for the window ending at bar i
        if i > n:
            s_tr = s_tr - s_tr / n + tr[i - 1]
            s_p = s_p - s_p / n + pdm[i - 1]
            s_m = s_m - s_m / n + mdm[i - 1]
        if s_tr == 0:
            dx = 0.0
        else:
            p_di, m_di = 100 * s_p / s_tr, 100 * s_m / s_tr
            dx = 0.0 if p_di + m_di == 0 else 100 * abs(p_di - m_di) / (p_di + m_di)
        dxs.append(dx)
        if len(dxs) == n:
            a = float(np.mean(dxs))
            out[i] = a
        elif len(dxs) > n:
            a = (a * (n - 1) + dx) / n
            out[i] = a
    return out
