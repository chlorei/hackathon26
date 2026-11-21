"""dreamkit on top of Dream Labs' own loop (dream_control.py, unchanged): `dreamkit run --mock ...`. No credits."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT.parent / "dreamlabs" / "dream_library" / "examples" / "yam_mujoco"


@pytest.mark.skipif(not (SIM / "dream_control.py").is_file(), reason="Dream Labs' simulator not checked out")
def test_overlay_on_dream_labs_loop_recovers_and_reports(tmp_path):
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs"),
                                   "plugins": ["success-check", "failure-monitor", "recovery"]}))
    out = subprocess.run([sys.executable, "-m", "dreamkit.cli", "-c", str(cfg), "run", "--no-panel", "--mock",
                          "grasped_air", "--", "python", str(SIM / "dream_control.py"), "--headless", "--seconds",
                          "45", "--quiet"], capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert "dreamkit: attempt 1 ✓ success" in out.stdout, out.stdout + out.stderr
    run = next((tmp_path / "runs").iterdir())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    said = [(e.get("plugin"), e.get("message", "")) for e in events]
    assert any(p == "failure-monitor" and "grasped air" in m for p, m in said)
    assert any(p == "recovery" and "grasped air" in m for p, m in said)
    assert json.loads((run / "run.json").read_text())["episodes"][0]["success"] is True
