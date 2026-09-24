"""SEC EDGAR 'company facts': every number a company has tagged in its filings.

Free and official. What it CAN do for backtests: every fact carries its
filing date (so we lag fundamentals to when they were public), and filings
of companies that were later delisted or acquired stay available.
What it CAN'T: tagged filings only start around 2009-2011, so a full
10-year history exists only from about 2019 onward; foreign companies that
file IFRS reports have tagged data only from about 2018.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from .annual import Financials, build_financials


class FundamentalsSource:
    def financials(self, security, as_of: date) -> Financials | None:
        """Annual financials as known on as_of, or None if unavailable."""
        raise NotImplementedError


class InMemoryFundamentals(FundamentalsSource):
    """For tests: raw company-facts dicts supplied directly, parsed by the real code."""

    def __init__(self, cfg: dict, facts_by_ticker: dict):
        self.fcfg = cfg["fundamentals"]
        self.facts = facts_by_ticker

    def financials(self, security, as_of):
        raw = self.facts.get(security.ticker)
        return build_financials(raw, self.fcfg, as_of) if raw else None


_last_request = [0.0]


def sec_get(cfg: dict, url: str, path: Path, max_age_days: float, offline: bool = False) -> dict | None:
    """GET a JSON document from the SEC, politely (identified, rate-limited,
    compressed), cached on disk gzipped. Returns None if unavailable offline."""
    f = cfg["fundamentals"]
    fresh = path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400
    if not fresh and not offline:
        wait = f["seconds_between_requests"] - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": f["user_agent"], "Accept-Encoding": "gzip"})
        _last_request[0] = time.time()
        try:
            with urllib.request.urlopen(req, timeout=cfg["data"]["request_timeout_seconds"]) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") != "gzip":
                    body = gzip.compress(body)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        json.loads(gzip.decompress(body))   # make sure it is valid before caching
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
    if not path.exists():
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


class EdgarFundamentals(FundamentalsSource):
    def __init__(self, cfg: dict, offline: bool = False, keep_raw: bool = True):
        self.cfg = cfg
        self.fcfg = cfg["fundamentals"]
        d = cfg["data"]
        self.cache = Path(cfg["_path"]).parent / d["cache_dir"] / self.fcfg["cache_subdir"]
        self.offline = offline
        self.keep_raw = keep_raw
        self._raw: dict[int, dict] = {}
        self._built: dict[tuple, Financials] = {}

    def financials(self, security, as_of):
        if not security.cik:
            return None
        key = (security.cik, as_of)
        if key not in self._built:
            raw = self._load(security.cik)
            self._built[key] = build_financials(raw, self.fcfg, as_of) if raw else None
        return self._built[key]

    def _load(self, cik: int) -> dict | None:
        if cik in self._raw:
            return self._raw[cik]
        path = self.cache / f"CIK{cik:010d}.json.gz"
        old = self.cache / f"CIK{cik:010d}.json"          # uncompressed cache from earlier versions
        if old.exists() and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(old.read_bytes()))
            old.unlink()
        try:
            raw = sec_get(self.cfg, self.fcfg["edgar_url"].format(cik=cik), path,
                          self.fcfg["cache_max_age_days"], self.offline)
        except (urllib.error.URLError, ValueError) as e:
            if not path.exists():
                raise LookupError(f"no EDGAR data for CIK {cik}: {e}") from e
            raw = sec_get(self.cfg, "", path, float("inf"), offline=True)
        if self.keep_raw:
            self._raw[cik] = raw
        return raw

    def prebuild(self, security, dates: list) -> None:
        """Build a company's financials for every date now, then let go of its
        raw filings (a few MB each), so hundreds of companies fit in memory."""
        if not security.cik:
            return
        raw = self._load(security.cik)
        for d in dates:
            self._built[(security.cik, d)] = build_financials(raw, self.fcfg, d) if raw else None
