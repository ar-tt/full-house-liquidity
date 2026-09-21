"""THE RANGE TABLE — does this trade keep the portfolio well spread out?

It is a property of the whole portfolio, not of any one stock, so it runs
first and can veto a trade outright. Two jobs:

1. HARD RULES (block the order if broken):
   - one company over 5%, one ETF over its cap, one sector over 20%,
     non-US over 25%, an asset class or region over its budget
   - a buy that pushes average correlation of the growth holdings above 0.6
   - a buy of a NEW name when the portfolio already holds the maximum
   - not enough cash, or not enough price history to judge correlation

2. RANGE SCORE (0-100): how healthy the portfolio looks AFTER the trade,
   built from correlation, effective number of bets, cap headroom and
   position count. Comparing candidate trades by this score is the same as
   asking which one improves the portfolio's structure most.

A rule that is already broken before the trade (for example a stock that
grew past 5% on its own) does not block unrelated trades. It only blocks
trades that make that same breach worse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ConfigError
from ..portfolio import Portfolio, Proposal, exposures
from ..stats import InsufficientHistory, average_pairwise, correlation_matrix, effective_bets
from .base import BLOCK, INFO, WARN, Pillar, PillarResult, Rule

EPS = 1e-9


def lin(x: float, best: float, worst: float) -> float:
    """100 at `best`, 0 at `worst`, straight line between, clamped."""
    if best == worst:
        return 100.0 if x == best else 0.0
    t = (x - worst) / (best - worst)
    return 100.0 * min(1.0, max(0.0, t))


@dataclass
class Assessment:
    """Structure of one portfolio: the numbers behind the Range score."""
    avg_corr: float
    bets: float
    positions: int
    fullest_cap: tuple            # (description, usage as share of its cap)
    corr_tickers: list
    excluded: list                # holdings left out of correlation (no price history)
    subscores: dict = field(default_factory=dict)
    score: float = 0.0


class RangeTable(Pillar):

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.rt = ctx.cfg["range_table"]
        w = self.rt["score_weights"]
        if abs(sum(w.values()) - 1) > 1e-6:
            raise ConfigError(f"range_table.score_weights must add up to 1, got {sum(w.values())}")
        for key, names in (("asset_class_budgets", "asset_classes"), ("region_budgets", "regions")):
            missing = set(self.rt[names]) - set(self.rt[key])
            if missing:
                raise ConfigError(f"range_table.{key} has no budget for {sorted(missing)}")

    # ------------------------------------------------------------------ caps
    def cap_usages(self, pf: Portfolio) -> list[tuple[str, float, float]]:
        """Every capped bucket as (description, exposure, cap). A 100% "cap"
        (e.g. cash) isn't a real limit, so it is left out."""
        rt, secs = self.rt, self.ctx.securities
        out = []
        issuers = exposures(pf, secs, "issuer")
        seen = set()
        for t in pf.holdings:
            s = secs[t]
            if s.type == "etf":
                out.append((f"ETF {t}", pf.weight(t), rt["etf_cap"]))
            elif s.issuer not in rt["name_cap_exempt_issuers"] and s.issuer not in seen:
                seen.add(s.issuer)
                out.append((f"company {s.issuer}", issuers[s.issuer], rt["single_name_cap"]))
        for sector, x in exposures(pf, secs, "sector", rt["sector_cap_asset_classes"]).items():
            if sector not in rt["sector_cap_exempt"]:
                out.append((f"sector {sector}", x, rt["sector_cap"]))
        regions = exposures(pf, secs, "region")
        out.append(("non-US total", sum(v for k, v in regions.items() if k != rt["us_region"]),
                    rt["non_us_cap"]))
        for region, x in regions.items():
            out.append((f"region {region}", x, rt["region_budgets"][region][1]))
        for ac, x in exposures(pf, secs, "asset_class").items():
            out.append((f"asset class {ac}", x, rt["asset_class_budgets"][ac][1]))
        return [u for u in out if u[2] < 1]

    # ------------------------------------------------------- correlation
    def corr_scope(self, pf: Portfolio) -> list[str]:
        c = self.rt["correlation"]
        return [t for t in pf.tickers(self.ctx.securities, c["asset_classes"])
                if self.ctx.securities[t].has_prices]

    def _corr_for(self, tickers: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
        """Correlation matrix, dropping (and reporting) holdings without enough history."""
        window = self.rt["correlation"]["window_days"]
        keep, dropped = list(tickers), []
        while True:
            try:
                return correlation_matrix(self.ctx.prices, keep, self.ctx.as_of, window), keep, dropped
            except InsufficientHistory as e:
                dropped += e.tickers
                keep = [t for t in keep if t not in e.tickers]

    # ------------------------------------------------------- assessment
    def assess(self, pf: Portfolio) -> Assessment:
        rt, c, b = self.rt, self.rt["correlation"], self.rt["bets"]
        tickers = self.corr_scope(pf)
        corr, used, dropped = self._corr_for(tickers)
        w = np.array([pf.holdings[t] for t in used], dtype=float)
        avg = average_pairwise(corr, w if c["averaging"] == "weighted" else None)
        bets = effective_bets(corr, b["method"]) if used else 0.0
        n = len(pf.tickers(self.ctx.securities, rt["positions"]["count_asset_classes"]))
        usages = self.cap_usages(pf)
        desc, x, cap = max(usages, key=lambda u: u[1] / u[2]) if usages else ("none", 0, 1)

        lo, hi = rt["positions"]["target"]
        band = rt["positions"]["soft_band"]
        gap = lo - n if n < lo else (n - hi if n > hi else 0)
        h = rt["headroom"]
        subs = {
            "correlation": lin(avg, c["good_avg_pairwise"], c["max_avg_pairwise"]),
            "bets": lin(bets, b["good"], b["floor"]),
            "headroom": lin(x / cap, h["full_score_below"], h["zero_score_at"]),
            "positions": lin(gap, 0, band),
        }
        score = sum(rt["score_weights"][k] * v for k, v in subs.items())
        return Assessment(avg, bets, n, (desc, x / cap), used, dropped, subs, score)

    # ------------------------------------------------------- evaluate
    def evaluate(self, proposal: Proposal, pf: Portfolio) -> PillarResult:
        rt, secs = self.rt, self.ctx.securities
        t = proposal.ticker
        rules: list[Rule] = []
        res = PillarResult(self.name, None, rules)

        if t not in secs:
            rules.append(Rule("unknown_security", False,
                              f"{t} is not in securities.yaml, so its sector and region are unknown", BLOCK))
            return res
        s = secs[t]
        buy = proposal.action == "buy"
        if buy and proposal.amount > pf.cash + EPS:
            rules.append(Rule("insufficient_cash", False,
                              f"buy of ${proposal.amount:,.0f} needs more than the ${pf.cash:,.0f} cash available",
                              BLOCK, proposal.amount, pf.cash))
            return res
        try:
            after = pf.after(proposal)
        except ConfigError as e:
            rules.append(Rule("oversell", False, str(e), BLOCK))
            return res

        def cap_rule(code, label, before, now, limit):
            breached, worse = now > limit + EPS, now > before + EPS
            if breached and worse and buy:
                rules.append(Rule(code, False, f"{label} would be {now:.1%}, over the {limit:.0%} limit",
                                  BLOCK, now, limit))
            elif breached:
                rules.append(Rule(code, False, f"{label} is {now:.1%}, over the {limit:.0%} limit, "
                                  "but this trade does not add to it", WARN, now, limit))
            else:
                rules.append(Rule(code, True, f"{label} {now:.1%} (limit {limit:.0%})", INFO, now, limit))

        # 1. one company / one ETF
        if s.type == "etf":
            cap_rule("etf_cap", f"ETF {t}", pf.weight(t), after.weight(t), rt["etf_cap"])
        elif s.issuer in rt["name_cap_exempt_issuers"]:
            rules.append(Rule("single_name_cap", True, f"{s.issuer} is exempt from the single-name cap"))
        else:
            iss_b = exposures(pf, secs, "issuer").get(s.issuer, 0.0)
            iss_a = exposures(after, secs, "issuer").get(s.issuer, 0.0)
            cap_rule("single_name_cap", f"company {s.issuer}", iss_b, iss_a, rt["single_name_cap"])

        # 2. sectors (equity only; broad ETFs exempt)
        if s.asset_class in rt["sector_cap_asset_classes"]:
            sb = exposures(pf, secs, "sector", rt["sector_cap_asset_classes"])
            sa = exposures(after, secs, "sector", rt["sector_cap_asset_classes"])
            for sector in s.sector_weights:
                if sector not in rt["sector_cap_exempt"]:
                    cap_rule("sector_cap", f"sector {sector}", sb.get(sector, 0), sa.get(sector, 0),
                             rt["sector_cap"])

        # 3. geography: total non-US, then each region's budget
        rb, ra = exposures(pf, secs, "region"), exposures(after, secs, "region")
        us = rt["us_region"]
        non_us = lambda e: sum(v for k, v in e.items() if k != us)
        cap_rule("non_us_cap", "non-US total", non_us(rb), non_us(ra), rt["non_us_cap"])
        for region in s.region_weights:
            lo, hi = rt["region_budgets"][region]
            cap_rule("region_budget", f"region {region}", rb.get(region, 0), ra.get(region, 0), hi)

        # 4. asset class budget
        ab, aa = exposures(pf, secs, "asset_class"), exposures(after, secs, "asset_class")
        lo, hi = rt["asset_class_budgets"][s.asset_class]
        cap_rule("asset_class_budget", f"asset class {s.asset_class}",
                 ab.get(s.asset_class, 0), aa.get(s.asset_class, 0), hi)
        for ac, (lo, _) in rt["asset_class_budgets"].items():
            if aa.get(ac, 0) < lo - EPS:
                rules.append(Rule("asset_class_min", False, f"asset class {ac} {aa.get(ac, 0):.1%} is "
                                  f"under its {lo:.0%} minimum", WARN, aa.get(ac, 0), lo))

        # 5. number of positions
        p = rt["positions"]
        n_after = len(after.tickers(secs, p["count_asset_classes"]))
        new_name = buy and t not in pf.holdings and s.asset_class in p["count_asset_classes"]
        if new_name and n_after > p["hard_max"]:
            rules.append(Rule("max_positions", False, f"a new name would make {n_after} positions, "
                              f"over the maximum of {p['hard_max']}", BLOCK, n_after, p["hard_max"]))
        else:
            rules.append(Rule("max_positions", True, f"{n_after} positions (target "
                              f"{p['target'][0]}-{p['target'][1]})", INFO, n_after, p["hard_max"]))

        # 6. correlation and effective bets
        c = rt["correlation"]
        in_scope = s.asset_class in c["asset_classes"] and s.has_prices
        before_a = self.assess(pf)
        after_a = self.assess(after)
        if in_scope and buy and t in after_a.excluded:
            sev = BLOCK if c["missing_history"] == "block" else WARN
            rules.append(Rule("missing_history", False, f"{t} has fewer than {c['window_days']} days of "
                              "prices, so its correlation with the portfolio can't be measured", sev))
        for x in sorted(set(after_a.excluded) - {t}):
            rules.append(Rule("missing_history", False, f"holding {x} left out of the correlation check "
                              "(not enough price history)", WARN))
        mx = c["max_avg_pairwise"]
        pushed = after_a.avg_corr > mx + EPS and after_a.avg_corr > before_a.avg_corr + EPS
        if buy and in_scope and pushed:
            rules.append(Rule("correlation_cap", False, f"average correlation would rise from "
                              f"{before_a.avg_corr:.2f} to {after_a.avg_corr:.2f}, above {mx:.2f}",
                              BLOCK, after_a.avg_corr, mx))
        else:
            rules.append(Rule("correlation_cap", after_a.avg_corr <= mx + EPS,
                              f"average correlation {before_a.avg_corr:.2f} -> {after_a.avg_corr:.2f} "
                              f"(limit {mx:.2f})", WARN if after_a.avg_corr > mx + EPS else INFO,
                              after_a.avg_corr, mx))

        res.score = after_a.score
        res.details = {"before": before_a, "after": after_a}
        return res
