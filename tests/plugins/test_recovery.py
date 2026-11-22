from dreamkit.testing import replay


def test_drives_home_and_asks_again_on_a_drop():
    r = replay("dropped_mid_carry", plugins=["failure-monitor", "recovery"])
    acts = [e["data"].get("action") for e in r.events.of("recovery") if e["data"].get("action")]
    assert acts and acts[0] == "home"
    assert r.result["recovery"]["count"] >= 1
    assert r.result["ui"]["recovery"]["marks"][0]["label"] == "recovery 1/3: home after dropped"  # History's strip
    assert r.budget_ms("recovery") < 1.0


def test_stops_the_robot_on_a_camera_failure():
    r = replay("camera_black", plugins=["camera-check", "failure-monitor", "recovery"])
    assert r.result["reason"] == "camera_failure" and r.result["stopped_by"] == "recovery"


def test_does_nothing_on_success():
    r = replay("success_apple_bowl", plugins=["failure-monitor", "recovery"])
    assert "recovery" not in r.result and not r.errors("recovery")


class _Ctx:
    """Just enough of an attempt's end for on_episode_end."""

    def __init__(self, reason):
        self.result = {"success": False, "reason": reason}
        self.attempt, self.scene, self.said, self.stopped = 1, 0, [], None

    def emit(self, level, message, **data):
        self.said.append((level, message))

    def stop_run(self, reason, message=""):
        self.stopped = (reason, message)

    def request_retry(self):
        pass


def test_ends_the_run_after_model_errors_in_a_row():
    from dreamkit.plugins.recovery import Recovery

    plugin = Recovery(stop_after_model_errors=3)
    ends = [_Ctx(r) for r in ("model_error", "model_error", "dropped", "model_error", "model_error", "model_error")]
    for ctx in ends:
        plugin.on_episode_start(ctx)
        plugin.on_episode_end(ctx)
    assert [c.stopped is not None for c in ends] == [False, False, False, False, False, True]  # a drop breaks the streak
    assert ends[-1].stopped[0] == "model_error" and "3 attempts in a row" in ends[-1].stopped[1]
