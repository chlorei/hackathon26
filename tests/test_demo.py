"""Live's demo button: the server starts examples/sim_loop.py, one run at a time, and Stop is the loop's Ctrl-C."""

import signal

import yaml
from fastapi.testclient import TestClient

from dreamkit import server


class FakeProc:
    def __init__(self, cmd, **kwargs):
        self.cmd, self.signals, self.code = cmd, [], None

    def poll(self):
        return self.code

    def send_signal(self, sig):
        self.signals.append(sig)
        self.code = 130

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        self.code = -15


def test_the_demo_starts_once_and_stops_like_ctrl_c(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kw: started.append(FakeProc(cmd, **kw)) or started[-1])
    monkeypatch.setenv("HOME", str(tmp_path))  # demo.log goes to a fresh ~/.dreamkit
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": "runs", "plugins": ["success-check"]}))
    c = TestClient(server.create_app(cfg))

    state = c.get("/api/demo").json()
    assert state["running"] is False and state["available"] and state["plugins"] == ["success-check"]
    assert c.post("/api/demo", json={"policy": "nonsense"}).status_code == 400
    assert c.post("/api/demo", json={"policy": "dream"}).status_code == 400  # no Dream API key here

    assert c.post("/api/demo", json={"policy": "mock"}).json()["running"] is True
    cmd = started[0].cmd
    assert cmd[1].endswith("sim_loop.py") and "--mock" in cmd and cmd[cmd.index("--config") + 1] == str(cfg)
    assert c.post("/api/demo", json={"policy": "mock"}).status_code == 409  # one run at a time

    stopped = c.delete("/api/demo").json()
    assert stopped["running"] is False and stopped["failed"] is None  # stopped by the user: not a failure
    assert started[0].signals == [signal.SIGINT]

    # a demo that dies at once (no simulator here): the panel gets its last lines to show
    assert c.post("/api/demo", json={"policy": "mock"}).json()["failed"] is None
    (tmp_path / ".dreamkit" / "demo.log").write_text("Traceback ...\nFileNotFoundError: Dream Labs simulator not found\n")
    started[-1].code = 1
    assert "simulator not found" in c.get("/api/demo").json()["failed"]
