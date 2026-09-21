"""THE CONVERGENCE ENGINE — do several sources of return point the same way?

Three sources, each scored 0-100 from raw filings and prices:

  QUALITY   the business compounds: ROIC above WACC for years, real free
            cash flow, growth through a full cycle, modest debt, profits
            backed by cash (low accruals)
  DIVIDEND  the cash it pays is durable: a long streak of increases, payout
            under 60% of profit, free cash flow covering it 1.5x or more
  VALUE     the price leaves room: the growth the price implies is below
            what the business has delivered, a 25% margin of safety to a
            DCF value, and cheap versus its own 10 years and its sector

The final score is the weighted average of the sources, marked down when
they disagree: final = average x (floor + (1 - floor) x share of sources
at or above the agreement threshold). A great business at a silly price
(one source agreeing) scores well below one that is good, fairly priced
and paying a growing dividend (three agreeing).

Hard rule: a buy needs the 25% margin of safety. Sells are never blocked.
ETFs and bonds get "no opinion" (this pillar judges businesses).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..fundamentals import metrics as M
from ..prices import close_on_or_after, split_factor_after, weekly_total_returns
from .base import BLOCK, INFO, WARN, Pillar, PillarResult, Rule
from .range_table import lin

SCORED, NO_OPINION, MISSING = "scored", "no opinion", "missing data"


@dataclass
class Metric:
    value: float | None
    score: float | None
    shown: str               # the value in words, e.g. "+8.1%/yr"
    note: str = ""


@dataclass
class Analysis:
    ticker: str
    status: str
    reason: str = ""
    metrics: dict = field(default_factory=dict)   # source -> {metric -> Metric}
    sources: dict = field(default_factory=dict)   # source -> score or None
    facts: dict = field(default_factory=dict)     # WACC, DCF value, margin of safety, ...
    notes: list = field(default_factory=list)
    base: float | None = None
    agreeing: int = 0
    factor: float | None = None
    score: float | None = None


def pct(x):
    return "n/a" if x is None else f"{x:+.1%}" if abs(x) < 10 else f"{x:+.0%}"


class ConvergenceEngine(Pillar):

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.c = ctx.cfg["convergence"]
        self._cache: dict[str, Analysis] = {}
        self._peer_cache: dict[str, float | None] = {}

    # ------------------------------------------------------------ helpers
    def _score(self, source: str, name: str, value, shown: str, note: str = "") -> Metric:
        m = self.c["metrics"][source][name]
        return Metric(value, None if value is None else lin(value, m["best"], m["worst"]), shown, note)

    def _bars(self, ticker):
        try:
            return self.ctx.prices.bars(ticker, self.ctx.as_of)
        except LookupError:
            return []

    def _market_value(self, sec, fin, bars):
        """(price, shares on today's basis, market cap) or None."""
        if not bars or not fin.latest_shares:
            return None
        filed, shares = fin.latest_shares
        shares *= split_factor_after(self.ctx.prices.split_events(sec.ticker), filed)
        price = bars[-1].close
        return price, shares, price * shares

    def _risk_free(self):
        w = self.c["wacc"]
        if self.ctx.curve is None:
            return w["risk_free_fallback"], "fallback (Fed curve not loaded)"
        r = self.ctx.curve.zero_rate(w["risk_free_maturity_years"])
        return math.exp(r) - 1, f"{w['risk_free_maturity_years']}y Treasury on {self.ctx.curve.as_of}"

    def _wacc(self, sec, fin, bars, mv, facts, notes):
        w = self.c["wacc"]
        rf, rf_src = self._risk_free()
        weeks = w["beta_years"] * 52
        start = self.ctx.as_of - timedelta(weeks=weeks)
        stock = {k: v for k, v in weekly_total_returns([b for b in bars if b.date >= start]).items()}
        mkt_bars = [b for b in self._bars(w["market_proxy"]) if b.date >= start]
        b = M.beta(stock, weekly_total_returns(mkt_bars), w["beta_shrink_to_one"], w["beta_bounds"])
        if b is None:
            b = 1.0
            notes.append("beta set to 1.0 (under 2 years of weekly prices)")
        ke = rf + b * w["equity_risk_premium"]
        last = fin.years[-1]
        debt = last.get("total_debt") or 0.0
        wacc = ke
        if mv and fin.currency == "USD" and debt > 0:
            prev_debt = fin.years[-2].get("total_debt") if len(fin.years) > 1 else None
            avg_debt = (debt + (prev_debt or debt)) / 2
            lo, hi = w["cost_of_debt_spread_bounds"]
            kd = min(max((last.get("interest_expense") or 0.0) / avg_debt, rf + lo), rf + hi)
            kd_after = kd * (1 - M.tax_rate(last, self.c["statutory_tax_rate"]))
            e = mv[2]
            wacc = e / (e + debt) * ke + debt / (e + debt) * kd_after
            facts["cost_of_debt_after_tax"] = kd_after
            facts["debt_weight"] = debt / (e + debt)
        elif fin.currency != "USD":
            notes.append(f"WACC uses cost of equity only ({fin.currency} accounts vs USD price)")
        wacc = max(wacc, w["floor"])
        facts.update(risk_free=rf, risk_free_source=rf_src, beta=b, cost_of_equity=ke, wacc=wacc)
        return wacc

    # ------------------------------------------------------------ analysis
    def analyze(self, ticker: str) -> Analysis:
        if ticker not in self._cache:
            self._cache[ticker] = self._analyze(ticker)
        return self._cache[ticker]

    def _analyze(self, ticker: str) -> Analysis:
        c = self.c
        sec = self.ctx.securities.get(ticker)
        if sec is None:
            return Analysis(ticker, MISSING, f"{ticker} is not in securities.yaml")
        if sec.type not in c["applies_to_types"]:
            return Analysis(ticker, NO_OPINION, f"{sec.type}s are judged by the other pillars, not by business quality")
        if sec.main_sector in c["unsupported_sectors"]:
            return Analysis(ticker, NO_OPINION, f"{sec.main_sector}: bank and insurer accounts don't fit "
                            "ROIC / free cash flow / EBITDA, so this engine can't score them fairly")
        if not sec.cik:
            return Analysis(ticker, MISSING, "no SEC company number (cik) in securities.yaml")
        try:
            fin = self.ctx.fundamentals.financials(sec, self.ctx.as_of) if self.ctx.fundamentals else None
        except LookupError as e:
            return Analysis(ticker, MISSING, str(e))
        if fin is None or len(fin.years) < c["min_years"]:
            n = 0 if fin is None else len(fin.years)
            return Analysis(ticker, MISSING, f"only {n} years of filings (need {c['min_years']})")

        a = Analysis(ticker, SCORED)
        hist = fin.years[-c["history_years"]:]
        last = hist[-1]
        k = c["dcf"]["fcf_average_years"]
        bars = self._bars(ticker)
        splits = self.ctx.prices.split_events(ticker)

        def on_todays_basis(fy, concept):
            """Share counts and per-share figures restated for every split after
            the filing the number came from (older filings predate later splits)."""
            v = fy.get(concept)
            if v is None:
                return None
            f = split_factor_after(splits, fy.filed.get(concept, fy.available))
            return v * f if concept == "diluted_shares" else v / f
        mv = self._market_value(sec, fin, bars)
        wacc = self._wacc(sec, fin, bars, mv, a.facts, a.notes)
        a.facts.update(currency=fin.currency, years_used=len(hist), latest_year=str(last.end),
                       latest_year_public=str(last.available))

        # ---------------- QUALITY
        q = {}
        roics = [r for d, r in M.roic_series(fin, c["statutory_tax_rate"], c["roic_cap"]) if d >= hist[0].end]
        if roics:
            spread = statistics.median(roics) - wacc
            q["roic_spread"] = self._score("quality", "roic_spread", spread,
                                           f"median ROIC {statistics.median(roics):.1%} vs WACC {wacc:.1%}")
            share = sum(r > wacc for r in roics) / len(roics)
            q["roic_consistency"] = self._score("quality", "roic_consistency", share,
                                                f"{sum(r > wacc for r in roics)} of {len(roics)} years above WACC")
        margins = [M.fcf(fy) / fy.get("revenue") for fy in hist[-k:]
                   if M.fcf(fy) is not None and fy.get("revenue")]
        if margins:
            m = sum(margins) / len(margins)
            q["fcf_margin"] = self._score("quality", "fcf_margin", m, f"{m:.1%} ({len(margins)}-yr avg)")
        rev_g = M.growth_rate([fy.get("revenue") for fy in hist])
        if rev_g is not None:
            q["revenue_growth"] = self._score("quality", "revenue_growth", rev_g, f"{pct(rev_g)}/yr over {len(hist)} yrs")
        eps = [on_todays_basis(fy, "eps_diluted") for fy in hist]
        eps_g = M.growth_rate(eps)
        recent_eps = [e for e in eps[-3:] if e is not None]
        if eps_g is not None:
            q["eps_growth"] = self._score("quality", "eps_growth", eps_g, f"{pct(eps_g)}/yr")
        elif recent_eps and sum(recent_eps) <= 0:
            q["eps_growth"] = Metric(None, 0.0, "losing money", "recent EPS at or below zero")
        nd, eb = M.net_debt(last), M.ebitda(last)
        if nd is not None and eb is not None:
            if eb <= 0:
                q["net_debt_to_ebitda"] = Metric(None, 0.0, "EBITDA negative", "no operating profit to cover debt")
            else:
                x = nd / eb
                q["net_debt_to_ebitda"] = self._score("quality", "net_debt_to_ebitda", x,
                                                      "net cash" if x < 0 else f"{x:.1f}x")
                a.facts["net_debt_to_ebitda"] = x
        acc = M.accrual_ratio(fin)
        if acc is not None:
            q["accrual_ratio"] = self._score("quality", "accrual_ratio", acc, f"{acc:+.1%} of assets")
        a.metrics["quality"] = q

        # ---------------- DIVIDEND
        d = {}
        paid = [fy.get("dividends_paid") or 0.0 for fy in hist[-k:]]
        recent_divs = [b for b in bars if b.dividend and (self.ctx.as_of - b.date).days <= 400]
        if hist[-1].get("dividends_paid") is None and recent_divs:
            a.notes.append("pays a dividend, but its filings don't tag the amount: dividend source not scored")
        elif paid[-1] > 0:
            streak = M.dividend_streak([(b.date, b.dividend) for b in bars if b.dividend], self.ctx.as_of)
            d["growth_streak"] = self._score("dividend", "growth_streak", streak, f"{streak} yrs of increases",
                                             "counted from price-feed dividends since " + str(self.ctx.cfg["prices"]["history_start"]))
            ni = sum(fy.get("net_income") or 0.0 for fy in hist[-k:])
            payout = sum(paid) / ni if ni > 0 else None
            d["payout_ratio"] = (self._score("dividend", "payout_ratio", payout, f"{payout:.0%} of profit")
                                 if payout is not None else Metric(None, 0.0, "no profit to pay from"))
            f_ = [M.fcf(fy) for fy in hist[-k:]]
            if None not in f_:
                cover = sum(f_) / sum(paid)
                d["fcf_coverage"] = self._score("dividend", "fcf_coverage", cover, f"{cover:.1f}x")
        elif c["non_payer_dividend"] == "zero":
            d["growth_streak"] = Metric(0, 0.0, "pays no dividend")
        else:
            a.notes.append("pays no dividend: the dividend source is left out, not scored as zero")
        a.metrics["dividend"] = d

        # ---------------- VALUE
        v = {}
        dc = c["dcf"]
        if fin.currency != "USD":
            a.notes.append(f"value not scored: accounts are in {fin.currency}, price is in USD "
                           "(currency conversion not built yet)")
        elif mv is None:
            a.notes.append("value not scored: no price or share count")
        else:
            price, shares, mcap = mv
            ev = mcap + (nd or 0.0)
            fcfs = [M.fcf(fy) for fy in hist[-k:] if M.fcf(fy) is not None]
            fcf0 = sum(fcfs) / len(fcfs) if fcfs else None
            a.facts.update(price=price, market_cap=mcap, enterprise_value=ev, fcf_start=fcf0)
            if fcf0 and fcf0 > 0 and rev_g is not None:
                terminal = min(dc["terminal_growth"], wacc - 0.01)
                g_used = min(max(rev_g, dc["growth_bounds"][0]), dc["growth_bounds"][1])
                iv_equity = M.dcf_value(fcf0, g_used, wacc, dc["years"], terminal) - (nd or 0.0)
                mos = 1 - mcap / iv_equity if iv_equity > 0 else -1.0
                a.facts.update(dcf_growth_used=g_used, intrinsic_value_per_share=iv_equity / shares,
                               margin_of_safety=mos)
                v["margin_of_safety"] = self._score(
                    "value", "margin_of_safety", mos,
                    f"{mos:+.0%} (value ${iv_equity / shares:,.2f} vs price ${price:,.2f})")
                implied = M.implied_growth(ev, fcf0, wacc, dc["years"], terminal, *dc["reverse_search"])
                if implied is not None:
                    gap = rev_g - implied
                    a.facts.update(implied_growth=implied, delivered_growth=rev_g)
                    v["reverse_dcf_gap"] = self._score(
                        "value", "reverse_dcf_gap", gap,
                        f"price implies {pct(implied)}/yr, business delivered {pct(rev_g)}/yr")
            else:
                a.notes.append("no DCF: free cash flow is not positive or growth unknown")
            # vs its own history: each past year valued on the day its annual report came out
            fy_hist, ev_hist = [], []
            for fy in hist:
                bar = close_on_or_after(bars, fy.available)
                sh = on_todays_basis(fy, "diluted_shares")
                if bar is None or not sh or (bar.date - fy.available).days > 10:
                    continue  # no price near the day this year's report came out
                cap = bar.close * sh
                if M.fcf(fy) is not None and cap > 0:
                    fy_hist.append(M.fcf(fy) / cap)
                e = M.ebit(fy)
                if e and e > 0 and M.net_debt(fy) is not None:
                    ev_hist.append((cap + M.net_debt(fy)) / e)
            now_fcf, now_ebit = M.fcf(last), M.ebit(last)
            if now_fcf is not None and fy_hist:
                y_now = now_fcf / mcap
                pos = M.position_in_range(y_now, fy_hist)
                if pos is not None:
                    v["fcf_yield_vs_own"] = self._score("value", "fcf_yield_vs_own", pos,
                                                        f"FCF yield {y_now:.1%}, {pos:.0%} of the way to its 10-yr high")
            if now_ebit and now_ebit > 0 and ev_hist:
                m_now = ev / now_ebit
                a.facts["ev_ebit"] = m_now
                pos = M.position_in_range(m_now, ev_hist)
                if pos is not None:
                    v["ev_ebit_vs_own"] = self._score("value", "ev_ebit_vs_own", pos,
                                                      f"EV/EBIT {m_now:.1f}x, {pos:.0%} of the way to its 10-yr high")
                peers = [p for p in (self._peer_ev_ebit(t) for t, s in self.ctx.securities.items()
                                     if t != ticker and s.type == "stock" and s.main_sector == sec.main_sector)
                         if p is not None]
                if len(peers) >= 2:
                    ratio = m_now / statistics.median(peers)
                    v["ev_ebit_vs_sector"] = self._score("value", "ev_ebit_vs_sector", ratio,
                                                         f"{ratio:.2f}x the sector median of {len(peers)} peers")
        a.metrics["value"] = v

        # ---------------- COMBINE
        for source, ms in a.metrics.items():
            weights = c["metrics"][source]
            got = [(weights[n]["weight"], m.score) for n, m in ms.items() if m.score is not None]
            if len(got) >= c["min_metrics_per_source"]:
                a.sources[source] = sum(w * s for w, s in got) / sum(w for w, _ in got)
            else:
                a.sources[source] = None
        avail = {s: x for s, x in a.sources.items() if x is not None}
        if len(avail) < c["min_sources_scored"]:
            a.status = MISSING
            a.reason = (f"only {len(avail)} of {len(a.sources)} return sources could be scored "
                        f"(need {c['min_sources_scored']}); " + "; ".join(a.notes))
            return a
        sw = {s: c["sources"][s]["weight"] for s in avail}
        a.base = sum(sw[s] * x for s, x in avail.items()) / sum(sw.values())
        a.agreeing = sum(x >= c["agree_threshold"] for x in avail.values())
        a.factor = c["agreement_floor"] + (1 - c["agreement_floor"]) * a.agreeing / len(avail)
        a.score = a.base * a.factor
        return a

    def _peer_ev_ebit(self, ticker: str):
        """Current EV/EBIT of a sector peer (USD reporters only), or None."""
        if ticker in self._peer_cache:
            return self._peer_cache[ticker]
        out = None
        sec = self.ctx.securities[ticker]
        if sec.cik and self.ctx.fundamentals and sec.main_sector not in self.c["unsupported_sectors"]:
            try:
                fin = self.ctx.fundamentals.financials(sec, self.ctx.as_of)
            except LookupError:
                fin = None
            if fin and fin.years and fin.currency == "USD":
                mv = self._market_value(sec, fin, self._bars(ticker))
                last = fin.years[-1]
                e, nd = M.ebit(last), M.net_debt(last)
                if mv and e and e > 0 and nd is not None:
                    out = (mv[2] + nd) / e
        self._peer_cache[ticker] = out
        return out

    # ------------------------------------------------------------ pillar
    def evaluate(self, proposal, portfolio) -> PillarResult:
        a = self.analyze(proposal.ticker)
        buy = proposal.action == "buy"
        rules: list[Rule] = []
        res = PillarResult(self.name, a.score if a.status == SCORED else None, rules, {"analysis": a})
        if a.status == NO_OPINION:
            rules.append(Rule("no_opinion", True, a.reason))
            return res
        if a.status == MISSING:
            block = buy and self.c["missing_data"] == "block"
            rules.append(Rule("missing_data", False, f"can't score {proposal.ticker}: {a.reason}",
                              BLOCK if block else WARN))
            return res
        h = self.c["hard_rules"]
        mos = a.facts.get("margin_of_safety")
        need = h["min_margin_of_safety"]
        if mos is None:
            why = next((n for n in a.notes if n.startswith(("value not", "no DCF"))), "value not computed")
            sev = BLOCK if buy and h["margin_of_safety_unknown"] == "block" else WARN
            rules.append(Rule("margin_of_safety", False, f"can't verify the {need:.0%} margin of safety: {why}", sev))
        elif mos < need:
            rules.append(Rule("margin_of_safety", False, f"margin of safety {mos:.0%} is below the required {need:.0%}",
                              BLOCK if buy else WARN, mos, need))
        else:
            rules.append(Rule("margin_of_safety", True, f"margin of safety {mos:.0%} (need {need:.0%})", INFO, mos, need))
        lev, max_lev = a.facts.get("net_debt_to_ebitda"), h.get("max_net_debt_to_ebitda")
        if max_lev is not None and lev is not None and lev > max_lev:
            rules.append(Rule("leverage", False, f"net debt/EBITDA {lev:.1f}x is over {max_lev:.1f}x",
                              BLOCK if buy else WARN, lev, max_lev))
        return res


def format_analysis(a: Analysis, explain: bool = False) -> list[str]:
    """Plain-text breakdown for the command line and the decision log."""
    if a.status != SCORED:
        return [f"{a.status.upper()}: {a.reason}"]
    out = []
    for source, ms in a.metrics.items():
        s = a.sources.get(source)
        out.append(f"{source.upper():9} {'not scored' if s is None else f'{s:.0f}'}")
        for name, m in ms.items():
            sc = "  -" if m.score is None else f"{m.score:3.0f}"
            out.append(f"   {sc}  {name:20} {m.shown}" + (f"  ({m.note})" if explain and m.note else ""))
    out.append(f"Average {a.base:.0f} x agreement {a.factor:.2f} ({a.agreeing} of "
               f"{sum(x is not None for x in a.sources.values())} sources >= threshold) = {a.score:.0f}")
    f = a.facts
    if explain:
        out.append(f"WACC {f['wacc']:.1%} = risk-free {f['risk_free']:.2%} ({f['risk_free_source']}) "
                   f"+ beta {f['beta']:.2f} x premium, blended with after-tax debt cost")
        out.append(f"Latest fiscal year {f['latest_year']} (public from {f['latest_year_public']}), "
                   f"{f['years_used']} years used, accounts in {f['currency']}")
        for n in a.notes:
            out.append(f"Note: {n}")
    return out
