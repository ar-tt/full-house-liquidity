"""Daily prices (OHLCV + dividends) and returns computed from them.

Free source: Yahoo's chart feed. It is unofficial and can change without
notice, which is why everything else talks to the PriceSource interface
and a paid source can replace it later without touching the pillars.

Limitations of the free source (they matter for backtests in step 5):
  - delisted companies are usually missing, so a universe built from it is
    biased toward survivors;
  - closes arrive already split-adjusted. We store each split event too, so
    code that compares prices with share counts from old filings can undo
    the adjustment (see split_factor_after). Dividends we add back ourselves.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    dividend: float = 0.0
    split: float = 1.0      # e.g. 10.0 on the day of a 10-for-1 split


class PriceSource:
    def bars(self, ticker: str, as_of: date) -> list[Bar]:
        """Daily bars up to and including as_of (never later: point in time)."""
        raise NotImplementedError

    def split_events(self, ticker: str) -> list[tuple[date, float]]:
        """EVERY split in the file, including ones after any as-of date. The
        vendor already adjusted all old prices for these, so undoing that
        adjustment needs the full list. It never feeds a decision directly."""
        return [(b.date, b.split) for b in self.bars(ticker, date.max) if b.split != 1.0]


class InMemoryPrices(PriceSource):
    """For tests and what-ifs: bars supplied directly."""

    def __init__(self, data: dict[str, list[Bar]]):
        self.data = data

    def bars(self, ticker, as_of):
        return [b for b in self.data.get(ticker, []) if b.date <= as_of]


class YahooChartPrices(PriceSource):
    def __init__(self, cfg: dict, offline: bool = False):
        p, d = cfg["prices"], cfg["data"]
        self.url = p["url"]
        self.start = p["history_start"]
        self.cache = Path(cfg["_path"]).parent / d["cache_dir"] / p["cache_subdir"]
        self.max_age = d["cache_max_age_days"] * 86400
        self.timeout = d["request_timeout_seconds"]
        self.offline = offline
        self._mem: dict[str, list[Bar]] = {}

    def bars(self, ticker, as_of):
        if ticker not in self._mem:
            self._mem[ticker] = self._load(ticker)
        return [b for b in self._mem[ticker] if b.date <= as_of]

    def _load(self, ticker: str) -> list[Bar]:
        path = self.cache / f"{ticker}.csv"
        fresh = path.exists() and time.time() - path.stat().st_mtime < self.max_age
        if not fresh and not self.offline:
            try:
                self._download(ticker, path)
            except (urllib.error.URLError, ValueError, KeyError) as e:
                if not path.exists():
                    raise LookupError(f"no price data for {ticker}: {e}") from e
        if not path.exists():
            raise LookupError(f"no cached price data for {ticker} (offline)")
        with open(path, newline="") as f:
            return [Bar(date.fromisoformat(r["date"]), float(r["open"]), float(r["high"]),
                        float(r["low"]), float(r["close"]), float(r["volume"]), float(r["dividend"]),
                        float(r.get("split") or 1.0))
                    for r in csv.DictReader(f)]

    def _download(self, ticker: str, path: Path) -> None:
        start = int(datetime(self.start.year, self.start.month, self.start.day, tzinfo=timezone.utc).timestamp())
        url = self.url.format(ticker=ticker, start=start, end=int(time.time()))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 fhl-quant-engine"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        chart = payload["chart"]
        if chart.get("error") or not chart.get("result"):
            raise ValueError(f"{ticker}: {chart.get('error')}")
        r = chart["result"][0]
        q = r["indicators"]["quote"][0]
        tz_offset = r["meta"].get("gmtoffset", 0)
        day = lambda ts: datetime.fromtimestamp(ts + tz_offset, timezone.utc).date()
        events = r.get("events") or {}
        divs = {day(ev["date"]): ev["amount"] for ev in events.get("dividends", {}).values()}
        splits = {day(ev["date"]): ev["numerator"] / ev["denominator"]
                  for ev in events.get("splits", {}).values() if ev.get("denominator")}
        rows = []
        for i, ts in enumerate(r.get("timestamp") or []):
            if q["close"][i] is None:
                continue  # holiday or halted day
            d = day(ts)
            rows.append([d.isoformat(), q["open"][i], q["high"][i], q["low"][i], q["close"][i],
                         q["volume"][i] or 0, divs.get(d, 0.0), splits.get(d, 1.0)])
        if not rows:
            raise ValueError(f"{ticker}: empty price history")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume", "dividend", "split"])
            w.writerows(rows)
        tmp.replace(path)


def daily_returns(bars: list[Bar]) -> dict[date, float]:
    """Total return each day: (close + dividend paid that day) / previous close - 1."""
    out = {}
    for prev, cur in zip(bars, bars[1:]):
        if prev.close > 0:
            out[cur.date] = (cur.close + cur.dividend) / prev.close - 1
    return out


def split_factor_after(splits: list[tuple[date, float]], d: date) -> float:
    """How many shares (on the vendor's split-adjusted basis) one share on
    date d has become. A share count from a filing on date d x this factor
    can be multiplied by the split-adjusted price."""
    f = 1.0
    for when, ratio in splits:
        if when > d:
            f *= ratio
    return f


def close_on_or_after(bars: list[Bar], d: date) -> Bar | None:
    """The first trading day's bar on or after d (e.g. the day a filing came out)."""
    for b in bars:
        if b.date >= d:
            return b
    return None


def weekly_total_returns(bars: list[Bar]) -> dict[tuple, float]:
    """Total return per calendar week (ISO year, week), dividends included."""
    level, last_by_week = 1.0, {}
    for prev, cur in zip(bars, bars[1:]):
        if prev.close > 0:
            level *= (cur.close + cur.dividend) / prev.close
        last_by_week[tuple(cur.date.isocalendar()[:2])] = level
    weeks = sorted(last_by_week)
    return {w: last_by_week[w] / last_by_week[p] - 1 for p, w in zip(weeks, weeks[1:])}
