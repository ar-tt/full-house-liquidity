"""THE POSITION CLOCK — WHEN to buy and in what steps. Never WHAT to buy.

LONG HAND (the wind-down, see fhl/wind_down.py): a buy of a growth asset
waits if it would lift the growth share above today's wind-down target.

SHORT HAND (entry timing), for stocks and ETFs:
  - The regime sets the pace. Price vs its 200-day average plus ADX (trend
    strength) give uptrend / no trend / downtrend, and each regime spaces
    the 3 tranches 2, 3 or 4 weeks apart (in over 4 to 8 weeks).
  - RSI under 40 (sold off) pulls the next tranche forward a week.
    RSI over 70 (run up) holds it back until it cools. After 8 weeks, every
    tranche is due whatever RSI says.
  - Size: a full position is base weight x (reference volatility / the
    stock's 60-day volatility): calmer stocks get bigger positions, jumpy
    ones smaller. Each tranche is a third of it.

"Not yet" is a WAIT, which holds the order without refusing it. Sells are
never held by the clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ..cashflows import Schedule
from ..curves import FlatCurve
from ..indicators import Snapshot, snapshot
from ..portfolio import exposures
from ..wind_down import WindDownStatus, status
from .base import INFO, WAIT, WARN, Pillar, PillarResult, Rule

EPS = 1e-9


@dataclass
class Plan:
    ticker: str
    snap: Snapshot
    tranche_no: int           # which tranche this order would be (1-based)
    tranches: int
    started: date
    due: date
    window_end: date
    rsi_effect: str           # accelerate | delay | neutral
    full_size: float
    tranche_size: float
    already_held: float
    new_entry: bool
    score: float
    notes: list = field(default_factory=list)

    def dates(self, spacing: timedelta) -> list[date]:
        return [self.started + i * spacing for i in range(self.tranches)]


class PositionClock(Pillar):

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.c = ctx.cfg["position_clock"]
        self.w = ctx.cfg["wind_down"]
        self.schedule = Schedule.from_config(ctx.cfg)

    # -------------------------------------------------------------- long hand
    def curve(self):
        if self.ctx.curve is not None:
            return self.ctx.curve
        r = self.ctx.cfg["reserve"]
        return FlatCurve(r["flat_rate"], r["compounding"])

    def wind_down(self, pf) -> WindDownStatus:
        growth = sum(exposures(pf, self.ctx.securities, "asset_class", self.w["growth_asset_classes"]).values())
        return status(pf.total, growth * pf.total, self.schedule, self.ctx.cfg, self.ctx.as_of,
                      self.curve(), pf.contributions_through)

    # -------------------------------------------------------------- short hand
    def full_size(self, sec, snap: Snapshot, total: float) -> tuple[float, str]:
        s = self.c["sizing"][sec.type]
        lo, hi = s["weight_bounds"]
        if snap.vol:
            w = s["base_weight"] * s["reference_vol"] / snap.vol
            why = (f"{s['base_weight']:.1%} x {s['reference_vol']:.0%} / {snap.vol:.0%} volatility")
        else:
            w, why = s["base_weight"], "base weight (volatility unknown)"
        w = min(max(w, lo), hi)
        return w * total, f"{w:.1%} of the portfolio = {why}, kept within {lo:.0%}-{hi:.0%}"

    def plan(self, ticker: str, pf) -> Plan | None:
        sec = self.ctx.securities[ticker]
        try:
            bars = self.ctx.prices.bars(ticker, self.ctx.as_of)
        except LookupError:
            return None
        snap = snapshot(bars, self.c["indicators"])
        if snap is None:
            return None
        n, today = self.c["tranches"], self.ctx.as_of
        reg = self.c["regimes"][snap.regime]
        spacing = timedelta(weeks=reg["spacing_weeks"])
        held = pf.holdings.get(ticker, 0.0)
        notes = list(snap.notes)
        e = pf.entries.get(ticker)
        if e and e.tranches_done < n:
            k, started, full, new = e.tranches_done + 1, e.started, e.full_size, False
            notes.append(f"continuing an entry started {e.started}: {e.tranches_done} of {n} tranches in")
        else:
            k, started, new = 1, today, True
            target, why = self.full_size(sec, snap, pf.total)
            full = max(0.0, target - held)
            notes.append(f"full position {why}" + (f", minus ${held:,.0f} already held" if held else ""))
        ind = self.c["indicators"]
        rsi = snap.rsi
        effect = ("accelerate" if rsi is not None and rsi < ind["rsi_accelerate_below"] else
                  "delay" if rsi is not None and rsi > ind["rsi_delay_above"] else "neutral")
        due = started + (k - 1) * spacing
        if effect == "accelerate":
            due -= timedelta(weeks=self.c["accelerate_weeks"])
        adj = self.c["rsi_score_adjust"]
        score = reg["score"] + (adj["accelerate"] if effect == "accelerate" else
                                adj["delay"] if effect == "delay" else 0)
        remaining = full - (k - 1) * full / n
        tranche = remaining if k == n else full / n
        return Plan(ticker, snap, k, n, started, due, started + timedelta(weeks=self.c["max_window_weeks"]),
                    effect, full, max(tranche, 0.0), held, new, max(0.0, min(100.0, score)), notes)

    # -------------------------------------------------------------- pillar
    def evaluate(self, proposal, pf) -> PillarResult:
        rules: list[Rule] = []
        res = PillarResult(self.name, None, rules)
        t, today = proposal.ticker, self.ctx.as_of
        sec = self.ctx.securities.get(t)
        if sec is None:
            return res
        if proposal.action == "sell":
            rules.append(Rule("exit", True, "exits are not timed by the clock"))
            return res
        if sec.type not in self.c["applies_to_types"] or not sec.has_prices:
            rules.append(Rule("no_opinion", True, f"{sec.type}s follow the reserve ladder, not the clock"))
            return res

        # LONG HAND: the wind-down target
        wd = self.wind_down(pf)
        res.details["wind_down"] = wd
        if sec.asset_class in self.w["growth_asset_classes"]:
            after = pf.after(proposal)
            g_after = sum(exposures(after, self.ctx.securities, "asset_class",
                                    self.w["growth_asset_classes"]).values())
            limit = wd.target + self.w["rebalance_tolerance"]
            if g_after > limit + EPS and g_after > wd.growth_share + EPS:
                rules.append(Rule("wind_down", False, f"growth share would be {g_after:.1%}, above the "
                                  f"wind-down target {wd.target:.1%} (+{self.w['rebalance_tolerance']:.0%} "
                                  f"tolerance). {wd.reasons[-1]}", WAIT, g_after, limit))
            else:
                rules.append(Rule("wind_down", True, f"growth share {g_after:.1%} within the wind-down "
                                  f"target {wd.target:.1%}", INFO, g_after, limit))

        # SHORT HAND: is a tranche due, and how big?
        plan = self.plan(t, pf)
        if plan is None:
            rules.append(Rule("no_prices", False, f"no price history for {t}: can't time the entry", WAIT))
            return res
        res.details["plan"] = plan
        res.score = plan.score
        ind = self.c["indicators"]
        rsi_txt = "n/a" if plan.snap.rsi is None else f"{plan.snap.rsi:.0f}"
        before_window_end = today < plan.window_end
        if plan.full_size <= 0:
            rules.append(Rule("full_size", False, f"{t} is already at its full size "
                              f"(${plan.already_held:,.0f} held)", WAIT))
        elif plan.rsi_effect == "delay" and before_window_end:
            rules.append(Rule("rsi_delay", False, f"RSI {rsi_txt} is above {ind['rsi_delay_above']}: tranche "
                              f"{plan.tranche_no} waits until it cools (or until {plan.window_end})", WAIT))
        elif today < plan.due and before_window_end:
            rules.append(Rule("tranche_due", False, f"tranche {plan.tranche_no} of {plan.tranches} is due "
                              f"{plan.due} ({plan.snap.regime.replace('_', ' ')} pace)", WAIT))
        else:
            why = ("the 8-week window has run out" if not before_window_end and today >= plan.started
                   else f"RSI {rsi_txt} pulled it forward" if plan.rsi_effect == "accelerate" else "on schedule")
            rules.append(Rule("tranche_due", True, f"tranche {plan.tranche_no} of {plan.tranches} is due now ({why})"))
            limit = plan.tranche_size * (1 + self.c["size_tolerance"])
            if proposal.amount > limit + EPS:
                rules.append(Rule("tranche_size", False, f"order ${proposal.amount:,.0f} is bigger than this "
                                  f"tranche (${plan.tranche_size:,.0f} of a ${plan.full_size:,.0f} position); "
                                  "resize it", WAIT, proposal.amount, limit))
        for n in plan.notes:
            if n.startswith("under"):
                rules.append(Rule("short_history", False, n, WARN))
        return res


def format_plan(plan: Plan, cfg: dict) -> list[str]:
    c = cfg["position_clock"]
    s, ind = plan.snap, c["indicators"]
    spacing = timedelta(weeks=c["regimes"][s.regime]["spacing_weeks"])
    fmt = lambda x, f: "n/a" if x is None else format(x, f)
    ma_txt = "n/a" if s.ma is None else f"${s.ma:,.2f} ({'above' if s.close > s.ma else 'below'})"
    out = [
        f"Price ${s.close:,.2f}   {ind['trend_ma_days']}-day average {ma_txt}",
        f"ADX {fmt(s.adx, '.0f')} (trend if > {ind['adx_trending_above']})   RSI {fmt(s.rsi, '.0f')} "
        f"(< {ind['rsi_accelerate_below']} speeds up, > {ind['rsi_delay_above']} delays)   "
        f"60-day volatility {fmt(s.vol, '.0%')}",
        f"Regime: {s.regime.replace('_', ' ')} -> tranches {spacing.days // 7} weeks apart, RSI effect: {plan.rsi_effect}",
        (f"Full position ${plan.full_size:,.0f}, tranche ${plan.tranche_size:,.0f} "
         f"(this is tranche {plan.tranche_no} of {plan.tranches})" if plan.full_size > 0 else
         f"Already at full size: ${plan.already_held:,.0f} held, nothing left to buy"),
        "Schedule: " + ", ".join(str(d) for d in plan.dates(spacing))
        + f"   (all in by {plan.window_end} at the latest)",
        f"Timing score {plan.score:.0f}",
    ]
    return out + [f"Note: {n}" for n in plan.notes]


def format_wind_down(wd: WindDownStatus, cfg: dict) -> list[str]:
    lo, hi = wd.band
    out = [f"Calendar band {lo:.0%}-{hi:.0%} growth (ceiling {hi:.0%})   now {wd.growth_share:.1%} growth "
           f"of ${wd.value:,.0f}"]
    if wd.funded_status is not None:
        out += [
            f"Projected funded status {wd.funded_status:.2f} = ${wd.projected_value:,.0f} projected at carve-out "
            f"/ ${wd.projected_reserve:,.0f} projected reserve cost",
            f"Lock-in: the remaining payments cost ${wd.lock_cost:,.0f} in Treasuries today; assets "
            f"(incl. promised contributions) cover {wd.lock_ratio:.0%} of that",
        ]
    out += [f"Rule: {r}" for r in wd.reasons]
    out.append(f"Target {wd.target:.1%} growth -> {wd.action(cfg['wind_down']['rebalance_tolerance'])}")
    return out
