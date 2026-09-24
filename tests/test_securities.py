import copy

import pytest
import yaml

from fhl.config import ConfigError
from fhl.portfolio import Portfolio
from fhl.securities import load_securities


def write(tmp_path, body):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"securities": body}))
    return p


def test_starter_universe_loads_and_example_portfolio_is_covered(cfg):
    secs = load_securities(cfg)
    pf = Portfolio.from_file("portfolio.yaml")
    assert pf.tickers(secs)  # raises if any holding is missing from securities.yaml
    assert secs["GOOGL"].issuer == secs["GOOG"].issuer == "Alphabet"
    assert secs["AAPL"].issuer == "Apple"          # issuer defaults to the name
    assert not secs["STRIP-2034"].has_prices


def test_weights_must_add_to_one(cfg, tmp_path):
    p = write(tmp_path, {"X": {"type": "etf", "asset_class": "equity", "region": "US",
                               "sector_weights": {"Energy": 0.5, "Utilities": 0.4}}})
    with pytest.raises(ConfigError, match="add up to 0.9"):
        load_securities(cfg, p)


@pytest.mark.parametrize("field, value, message", [
    ("sector", "Crypto", "unknown sector"),
    ("region", "Mars", "unknown region"),
    ("type", "option", "type must be"),
    ("asset_class", "art", "asset_class must be"),
])
def test_bad_labels_rejected(cfg, tmp_path, field, value, message):
    body = {"type": "stock", "asset_class": "equity", "sector": "Energy", "region": "US"}
    body[field] = value
    with pytest.raises(ConfigError, match=message):
        load_securities(cfg, write(tmp_path, {"X": body}))


def test_held_but_unlisted_ticker_is_an_error(cfg):
    with pytest.raises(ConfigError, match="missing from securities.yaml"):
        Portfolio({"ZZZZ": 1.0}).tickers(load_securities(cfg))


def test_local_config_overrides_without_touching_the_rest(tmp_path):
    from fhl.config import DEFAULT_PATH, load_config
    main = tmp_path / "config.yaml"
    main.write_text(DEFAULT_PATH.read_text())
    (tmp_path / "config.local.yaml").write_text('fundamentals:\n  user_agent: "private"\n')
    cfg = load_config(main)
    assert cfg["fundamentals"]["user_agent"] == "private"
    assert cfg["fundamentals"]["edgar_url"].startswith("https://data.sec.gov")   # siblings kept
    (tmp_path / "config.local.yaml").unlink()
    assert "[add contact email" in load_config(main)["fundamentals"]["user_agent"]
