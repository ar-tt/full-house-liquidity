"""SEC EDGAR 'company facts': every number a company has tagged in its filings.

Free and official. What it CAN do for backtests: every fact carries its
filing date (so we lag fundamentals to when they were public), and filings
of companies that were later delisted or acquired stay available.
What it CAN'T: tagged filings only start around 2009-2011, so a full
10-year history exists only from about 2019 onward; foreign companies that
file IFRS reports have tagged data only from about 2018.
"""
from __future__ import annotations

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


class EdgarFundamentals(FundamentalsSource):
    def __init__(self, cfg: dict, offline: bool = False):
        self.fcfg = cfg["fundamentals"]
        d = cfg["data"]
        self.cache = Path(cfg["_path"]).parent / d["cache_dir"] / self.fcfg["cache_subdir"]
        self.timeout = d["request_timeout_seconds"]
        self.offline = offline
        self._raw: dict[int, dict] = {}
        self._built: dict[tuple, Financials] = {}
        self._last_request = 0.0

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
        path = self.cache / f"CIK{cik:010d}.json"
        max_age = self.fcfg["cache_max_age_days"] * 86400
        fresh = path.exists() and time.time() - path.stat().st_mtime < max_age
        if not fresh and not self.offline:
            try:
                self._download(cik, path)
            except (urllib.error.URLError, ValueError) as e:
                if not path.exists():
                    raise LookupError(f"no EDGAR data for CIK {cik}: {e}") from e
        raw = json.loads(path.read_text()) if path.exists() else None
        self._raw[cik] = raw
        return raw

    def _download(self, cik: int, path: Path) -> None:
        wait = self.fcfg["seconds_between_requests"] - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(self.fcfg["edgar_url"].format(cik=cik),
                                     headers={"User-Agent": self.fcfg["user_agent"]})
        self._last_request = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read()
        json.loads(body)  # make sure it is valid before caching
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
