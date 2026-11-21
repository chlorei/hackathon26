from dreamkit.testing import replay


def test_writes_a_self_contained_report():
    r = replay("dropped_mid_carry", plugins=["failure-monitor", "report"], keep=True)
    page = (r.run_dir / "report" / "report.html").read_text()
    assert "0%" in page and "dropped" in page


def test_report_of_a_success():
    r = replay("success_apple_bowl", plugins=["success-check", "report"], keep=True)
    assert "100%" in (r.run_dir / "report" / "report.html").read_text()
