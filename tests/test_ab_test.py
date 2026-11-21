"""examples/ab_test.py with a stand-in for sim_loop: variants run on the same seeds and are ranked against the first."""

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ab_test", ROOT / "examples" / "ab_test.py")
ab_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab_test)


class FakeSimLoop:
    """Writes a run like sim_loop would: one attempt per seed; 'place' prompts succeed on every seed."""

    def __init__(self):
        self.calls = []

    def main(self, argv):
        p = argparse.ArgumentParser()
        for flag in ("--config", "--episodes", "--seconds", "--task", "--model", "--seed", "--behaviors"):
            p.add_argument(flag)
        p.add_argument("--mock", action="store_true")
        p.add_argument("--fast", action="store_true")
        a = p.parse_args(argv)
        self.calls.append(a)
        cfg = yaml.safe_load(Path(a.config).read_text())
        a.plugins = cfg["plugins"]
        run = Path(cfg["runs_dir"]) / f"run{len(self.calls)}"
        run.mkdir(parents=True)
        good = a.task.startswith("place")
        eps = [{"episode": i + 1, "scene": i, "attempt": 1, "seed": int(a.seed) + i, "success": good or i == 0,
                "reason": None if good or i == 0 else "dropped", "requests": 10, "duration_s": 20.0}
               for i in range(int(a.episodes))]
        (run / "run.json").write_text(json.dumps({"id": run.name, "episodes": eps}))
        return 0


def test_prompt_tuner_ranks_variants_on_the_same_seeds(tmp_path, capsys):
    config = tmp_path / "dreamkit.yaml"
    config.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"), "plugins": []}))
    fake = FakeSimLoop()
    code = ab_test.main(["--config", str(config), "--mock", "--fast", "--episodes", "8", "--tune",
                         "--max-prompts", "3"], sim_loop=fake)
    assert code == 0
    assert len(fake.calls) == 3 and {c.seed for c in fake.calls} == {"42"}  # same scenes for every variant
    assert fake.calls[0].task == "put apple into the bowl"  # the task itself is the baseline
    first_rel, second_rel = (c.plugins[-1]["reliability"] for c in fake.calls[:2])
    assert "baseline_run" not in first_rel and second_rel["baseline_run"].endswith("run1")  # B vs A in the panel too
    out = sorted((tmp_path / "runs").glob("ab-*.json"))[-1]
    rows = json.loads(out.read_text())
    place = next(r for r in rows if r["prompt"].startswith("place"))
    assert place["summary"]["successes"] == 8
    assert place["vs_baseline"]["only_this_won"] == 7 and place["vs_baseline"]["p_value"] < 0.05
    assert "(real)" in capsys.readouterr().out


def test_paraphrases_keep_object_and_target():
    for p in ab_test.paraphrases("put banana on the plate", 7):
        assert ab_test.parse(p) == ("banana", "plate")
