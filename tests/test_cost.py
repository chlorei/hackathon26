"""dreamkit.cost: the one place a dollar is computed, and what the core records with it."""

import json

from dreamkit import cost
from dreamkit.testing import replay


def test_price_by_model_and_nothing_for_a_mock():
    assert cost.price_per_hour("dreamlabs/dream-1-beta") == 3.0
    assert cost.price_per_hour("dreamlabs/dream-1-flash") == 2.0
    assert cost.price_per_hour("dreamlabs/dream-1-beta", mock=True) == 0.0
    assert cost.price_per_hour("dream-1-flash", prices={"flash": 1.5}) == 1.5  # dreamkit.yaml prices win
    assert cost.usd(2250, 3.0) == 3.0 and cost.usd(0, 3.0) == 0.0
    assert cost.price_per_hour("scripted") == 0.0 and cost.price_per_hour("") == 0.0  # not a Dream model: free


def test_each_attempt_and_the_run_carry_what_the_model_cost(monkeypatch):
    from dreamkit import kit

    monkeypatch.setattr(kit.Kit, "_price", lambda self: 3.0)  # a replay names no model: billed as Dream-1 here
    r = replay("success_apple_bowl", plugins=["success-check"], keep=True)
    calls = r.result["requests"]
    assert calls > 0 and r.result["cost_usd"] == round(cost.usd(calls, 3.0), 4)
    run = json.loads((r.run_dir / "run.json").read_text())
    assert run["cost_usd"] == r.result["cost_usd"]


def test_a_mock_run_costs_nothing():
    r = replay("success_apple_bowl", plugins=["success-check"])
    assert r.result["cost_usd"] == 0.0
