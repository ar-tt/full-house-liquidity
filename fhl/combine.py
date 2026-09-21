"""Turn pillar results into one decision: BLOCKED, HOLD or ACT.

Structural pillars (the Range Table) run first. If one blocks, the order
stops there and the other pillars are not consulted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import ConfigError
from .pillars.base import Pillar, PillarResult
from .portfolio import Portfolio, Proposal

BLOCKED, HOLD, ACT = "BLOCKED", "HOLD", "ACT"


@dataclass
class Decision:
    proposal: Proposal
    status: str
    score: float | None
    reason: str
    results: list[PillarResult] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)


def combine_scores(scored: list[tuple[Pillar, float]], method: str) -> float | None:
    if method == "geometric":
        # 100 x product of (score/100)^(weight share): a 0 anywhere makes the whole 0
        total_w = sum(p.weight for p, _ in scored)
        if not scored or total_w == 0:
            return None
        out = 1.0
        for p, s in scored:
            out *= (max(s, 0.0) / 100) ** (p.weight / total_w)
        return 100 * out
    if method == "gate_then_weighted":
        # structural pillars only open or close the gate; their scores aren't used
        use = [(p, s) for p, s in scored if not p.structural]
        total_w = sum(p.weight for p, _ in use)
        return sum(p.weight * s for p, s in use) / total_w if total_w else None
    raise ConfigError(f"combine.method must be geometric or gate_then_weighted, got '{method}'")


def decide(proposal: Proposal, portfolio: Portfolio, pillars: list[Pillar], cfg: dict) -> Decision:
    results: list[PillarResult] = []
    for i, pillar in enumerate(pillars):
        res = pillar.evaluate(proposal, portfolio)
        results.append(res)
        if res.blocked:
            first = next(r for r in res.fired if r.severity == "block")
            return Decision(proposal, BLOCKED, None,
                            f"{pillar.name}: {first.message}", results,
                            not_run=[p.name for p in pillars[i + 1:]])
    scored = [(p, r.score) for p, r in zip(pillars, results) if r.score is not None]
    score = combine_scores(scored, cfg["combine"]["method"])
    if proposal.action == "sell":
        return Decision(proposal, ACT, score, "sells are not held back by score", results)
    threshold = cfg["combine"]["min_score_to_act"]
    if score is None:
        return Decision(proposal, ACT, None, "passed every rule; no scoring pillar has an "
                        "opinion on this security, so the rules alone decide", results)
    if score < threshold:
        return Decision(proposal, HOLD, score,
                        f"passed every rule, but combined score "
                        f"{'n/a' if score is None else f'{score:.0f}'} is below {threshold}", results)
    return Decision(proposal, ACT, score, f"passed every rule; combined score {score:.0f}", results)
