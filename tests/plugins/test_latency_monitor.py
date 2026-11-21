from dreamkit.testing import replay


def test_warns_on_slow_round_trips():
    r = replay("slow_model", plugins=["latency-monitor"])
    assert r.events.any(plugin="latency-monitor", level="warn")
    assert r.result["latency-monitor"]["mean_ms"] > 700


def test_mock_reports_no_model_calls():
    r = replay("success_apple_bowl", plugins=["latency-monitor"])
    assert r.result["latency-monitor"] == {"mock": True}
