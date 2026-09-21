"""Stand-in pillars for testing the plug-in mechanism (not used by the engine)."""
from fhl.pillars.base import BLOCK, Pillar, PillarResult, Rule

CALLS = []


class FixedScore(Pillar):
    """Returns spec['fixed_score']; blocks if spec['block'] is true."""

    def evaluate(self, proposal, portfolio):
        CALLS.append(self.name)
        rules = [Rule("dummy_block", False, "told to block", BLOCK)] if self.spec.get("block") else []
        return PillarResult(self.name, self.spec.get("fixed_score"), rules)


class NotAPillar:
    pass
