"""The Create tab's backend with a fake model: no network, no credits."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from dreamkit import create, registry
from dreamkit.manager import discover

ROOT = Path(__file__).resolve().parent.parent
FILES = ["pyproject.toml", "gripper_check/__init__.py", "gripper_check/plugin.py", "tests/test_gripper_check.py",
         "README.md"]


def package(name="grip-watch", **swap):
    """A known-good package: gripper-check under another name. swap={file: (old, new)} breaks it on purpose."""
    module, cls = name.replace("-", "_"), "".join(w.title() for w in name.split("-"))
    files = {}
    for f in FILES:
        text = (ROOT / "plugins-external" / "gripper-check" / f).read_text()
        text = create.strip_ui(text).replace("gripper-check", name).replace("gripper_check", module).replace("GripperCheck", cls)
        files[f.replace("gripper_check", module)] = text
    for f, (old, new) in swap.items():
        files[f] = files[f].replace(old, new)
    return json.dumps({"name": name, "summary": "watches the grippers", "files": files})


@pytest.fixture
def model(monkeypatch):
    calls, script = [], []

    def ask(prompt):
        calls.append(prompt)
        return script.pop(0), "fake-model"

    monkeypatch.setattr(create, "ask", ask)
    return calls, script


def test_a_described_plugin_is_generated_checked_and_catalogued(model, tmp_path):
    calls, script = model
    script.append(package())
    job = create.run(create.Job("j1", [{"role": "user", "text": "watch the grippers"}]), None, set())
    assert job.status == "ready", job.error
    assert [s["text"] for s in job.steps] == ["generating", "checking", "trying it on recorded attempts", "ready"]
    # what it would say on recorded attempts, before anyone reads the code
    assert [r["fixture"] for r in job.tryout] == create.TRYOUT_FIXTURES
    assert job.tryout[0]["success"] and not job.tryout[1]["success"]
    assert all(isinstance(e["message"], str) for r in job.tryout for e in r["said"])
    assert all(job.checks[s]["ok"] for s in ("check", "test", "bench"))
    assert job.info["name"] == "grip-watch" and "on_step" in job.info["hooks"] and "closed_cmd" in job.info["defaults"]
    assert "watch the grippers" in calls[0] and "## 6. Reading the robot" in calls[0]  # the standard is in the prompt

    names = zipfile.ZipFile(io.BytesIO(create.zip_bytes(job))).namelist()
    assert sorted(names) == sorted(f"grip-watch/{f.replace('gripper_check', 'grip_watch')}" for f in FILES)

    create.add_to_catalog(job, tmp_path)
    found, problems = discover([tmp_path / "plugins"])
    assert found["grip-watch"]["source"] == "generated" and not [p for p in problems if "grip" in p]
    assert registry.uninstall("grip-watch", found).returncode == 0
    assert not (tmp_path / "plugins-generated" / "grip-watch").exists()


def test_bad_json_is_asked_again_and_refused_code_goes_back_to_the_model(model):
    calls, script = model
    script += ["not json at all",
               package(**{"grip_watch/plugin.py": ("from __future__ import annotations", "import subprocess")}),
               package()]
    job = create.run(create.Job("j2", [{"role": "user", "text": "watch the grippers"}]), None, set())
    assert job.status == "ready" and job.tries == 2
    assert "not valid" in calls[1] and "import subprocess is not allowed" in calls[2]
    assert any(s["text"].startswith("fixing: ") for s in job.steps)


def test_three_refused_tries_and_it_gives_up(model):
    calls, script = model
    script += [package(**{"grip_watch/plugin.py": ('capability="observe"', 'capability="control"')})] * 3
    job = create.run(create.Job("j3", [{"role": "user", "text": "x"}]), None, set())
    assert job.status == "failed" and job.tries == 3 and len(calls) == 3 and "capability" in job.error


def test_the_panel_route_from_chat_to_catalog_zip_and_back(model, tmp_path, monkeypatch):
    """What the Create tab and the Plugins page do, over HTTP."""
    import time

    import yaml
    from fastapi.testclient import TestClient

    from dreamkit import server

    calls, script = model
    script.append(package())
    monkeypatch.chdir(tmp_path)  # LOCAL_DIRS has a relative plugins/; keep it in this test's folder
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": "runs", "plugins": ["success-check"]}))
    c = TestClient(server.create_app(cfg))

    assert c.post("/api/create", json={"messages": []}).status_code == 400
    job = c.post("/api/create", json={"messages": [{"role": "user", "text": "watch the grippers"}]}).json()
    for _ in range(300):
        job = c.get(f"/api/create/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "ready", job["error"]

    z = c.get(f"/api/create/{job['id']}/zip")
    assert z.headers["content-type"] == "application/zip" and "grip-watch.zip" in z.headers["content-disposition"]
    plugins = c.post(f"/api/create/{job['id']}/catalog", json={"enable": True}).json()["plugins"]
    mine = next(p for p in plugins if p["name"] == "grip-watch")
    assert mine["source"] == "generated" and mine["enabled"]
    assert "grip-watch" in cfg.read_text()

    assert c.delete("/api/plugins/grip-watch").status_code == 200
    assert "grip-watch" not in cfg.read_text() and not (tmp_path / "plugins-generated" / "grip-watch").exists()


@pytest.mark.parametrize("code, bad", [
    ("import os", "import os is not allowed"),
    ("from pathlib import Path", "from pathlib import is not allowed"),
    ("open('x')", "open() is not allowed"),
    ("import time\ntime.sleep(1)", "import time is not allowed"),
    ("x = ().__class__.__bases__", ".__class__ is not allowed"),
    ("p = ctx.files.path('a.json').parent", ".parent is not allowed"),
    ("u = 'https://example.com'", "no URLs"),
])
def test_the_scan_refuses_what_reaches_outside(code, bad):
    assert any(bad in p for p in create._scan("m/plugin.py", code, "m"))


def test_names_and_files_are_checked_before_anything_runs():
    data = json.loads(package())
    assert "taken" in create.validate(data, {"grip-watch"})[0]
    assert "kebab" in create.validate({**data, "name": "Bad Name"}, set())[0]
    extra = {**data, "files": {**data["files"], "../evil.py": "x = 1"}}
    assert any("unexpected file ../evil.py" in p for p in create.validate(extra, set()))


def test_a_failure_reads_as_a_sentence_in_the_chat(monkeypatch):
    from dreamkit import llm

    def busy(prompt):
        raise llm.Unavailable("no Gemini model available (gemini-3.6-flash: 429)", quota=True)

    monkeypatch.setattr(create, "ask", busy)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    job = create.run(create.Job("q", [{"role": "user", "text": "x"}]), None, set())
    assert job.status == "failed" and job.error.startswith("Gemini quota exhausted for today")
    hung = llm.Unavailable("no Gemini model available (gemini-3.8-flash: 504; out of time)")
    assert create.friendly(hung).startswith("Gemini did not answer in time")


def test_checks_that_keep_failing_name_the_check(model):
    calls, script = model
    script += [package(**{"tests/test_grip_watch.py": ("def test", "def test_broken():\n    assert False\n\n\ndef test")})] * 3
    job = create.run(create.Job("c", [{"role": "user", "text": "x"}]), None, set())
    assert job.status == "failed" and job.error.startswith("generated plugin failed test")


def test_model_calls_are_counted_by_day_and_model(tmp_path, monkeypatch):
    from dreamkit import llm

    monkeypatch.setattr(llm, "USAGE", tmp_path / "usage.json")
    for outcome in ("ok", "ok", "429"):
        llm.record("gemini-3.6-flash", outcome)
    llm.record("gemini-3.8-flash", "504")
    assert llm.usage_today() == {"gemini-3.6-flash": {"requests": 3, "ok": 2, "429": 1},
                                 "gemini-3.8-flash": {"requests": 1, "504": 1}}
