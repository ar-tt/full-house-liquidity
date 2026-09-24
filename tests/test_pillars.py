"""The plug-in interface and how pillar scores combine into a decision."""
import copy

import pytest

import dummy_pillars
from fhl.combine import ACT, BLOCKED, HOLD, decide
from fhl.config import ConfigError
from fhl.pillars.base import Context, load_pillars
from fhl.portfolio import Portfolio, Proposal
from test_range_table import AS_OF, BASE, PRICES, universe

DUMMY = "dummy_pillars:FixedScore"


def ctx_for(cfg):
    return Context(cfg, universe(), PRICES, AS_OF)


def with_pillars(cfg, *extra, range_on=True, method=None):
    """Range (optionally) plus the given stand-in pillars; the real scoring
    pillars are switched off so these tests don't need filings data."""
    c = copy.deepcopy(cfg)
    for p in c["pillars"]:
        p["enabled"] = p["name"] == "range" and range_on
    c["pillars"] = c["pillars"] + list(extra)
    if method:
        c["combine"]["method"] = method
    return c


def buy(ticker="FIN0", amount=5_000):
    return Proposal(ticker, "buy", amount, AS_OF)


PF = Portfolio(dict(BASE), 250_000)


def test_default_config_loads_all_three_pillars(cfg):
    pillars = load_pillars(cfg, ctx_for(cfg))
    assert [p.name for p in pillars] == ["range", "convergence", "position"]
    assert pillars[0].structural and pillars[0].weight == 50
    assert cfg["combine"]["method"] == "gate_then_weighted"


def test_a_new_pillar_is_one_config_line(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 80})
    c["pillars"][0]["enabled"] = True
    assert [p.name for p in load_pillars(c, ctx_for(c))] == ["range", "edge"]


def test_structural_pillars_run_first_whatever_the_config_order(cfg):
    c = copy.deepcopy(cfg)
    c["pillars"] = [{"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 80}] + c["pillars"]
    assert [p.name for p in load_pillars(c, ctx_for(c))] == ["range", "edge", "convergence", "position"]


def test_disabled_pillar_file_need_not_exist(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": "fhl.pillars.edge:Edge", "weight": 25, "enabled": False})
    assert [p.name for p in load_pillars(c, ctx_for(c))] == ["range"]   # edge.py doesn't exist: no error


@pytest.mark.parametrize("module, message", [
    ("fhl.pillars.nope:Nope", "can't load"),
    ("dummy_pillars:NotAPillar", "not a Pillar"),
])
def test_bad_pillar_entries_are_rejected(cfg, module, message):
    c = with_pillars(cfg, {"name": "bad", "module": module, "weight": 1})
    with pytest.raises(ConfigError, match=message):
        load_pillars(c, ctx_for(c))


def test_range_block_stops_before_other_pillars_run(cfg):
    dummy_pillars.CALLS.clear()
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 100})
    d = decide(buy("IND0", 21_000), PF, load_pillars(c, ctx_for(c)), c)   # IND0 -> 5.1%
    assert d.status == BLOCKED
    assert d.not_run == ["edge"] and dummy_pillars.CALLS == []
    assert d.reason.startswith("range:") and "5% limit" in d.reason


def test_geometric_combination(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 90},
                     {"name": "clock", "module": DUMMY, "weight": 25, "fixed_score": 40}, method="geometric")
    d = decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
    r = d.results[0].score
    assert d.score == pytest.approx(100 * (r / 100) ** 0.5 * 0.9 ** 0.25 * 0.4 ** 0.25)


def test_a_zero_anywhere_makes_geometric_zero(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 0}, method="geometric")
    d = decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
    assert d.score == 0 and d.status == HOLD


def test_pillar_with_no_opinion_is_left_out(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": None}, method="geometric")
    d = decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
    assert d.score == pytest.approx(d.results[0].score)


def test_gate_with_no_scoring_opinion_lets_the_rules_decide(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": None})
    d = decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
    assert d.status == ACT and d.score is None and "rules alone" in d.reason


def test_gate_never_uses_the_range_score(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 61})
    d = decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
    assert d.score == pytest.approx(61)


def test_gate_then_weighted_ignores_the_range_score(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 90},
                     {"name": "clock", "module": DUMMY, "weight": 25, "fixed_score": 50},
                     method="gate_then_weighted")
    d = decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
    assert d.score == pytest.approx(70) and d.status == ACT


def test_non_structural_pillar_can_block_too(cfg):
    c = with_pillars(cfg, {"name": "clock", "module": DUMMY, "weight": 25, "fixed_score": 90, "block": True})
    assert decide(buy(), PF, load_pillars(c, ctx_for(c)), c).status == BLOCKED


def test_below_threshold_is_hold_not_act(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 90})
    c["combine"]["min_score_to_act"] = 101
    assert decide(buy(), PF, load_pillars(c, ctx_for(c)), c).status == HOLD


def test_sells_act_even_with_low_score(cfg):
    c = with_pillars(cfg, {"name": "edge", "module": DUMMY, "weight": 25, "fixed_score": 1})
    d = decide(Proposal("FIN0", "sell", 5_000, AS_OF), PF, load_pillars(c, ctx_for(c)), c)
    assert d.status == ACT


def test_unknown_combine_method_rejected(cfg):
    c = with_pillars(cfg, method="average")
    with pytest.raises(ConfigError, match="combine.method"):
        decide(buy(), PF, load_pillars(c, ctx_for(c)), c)
