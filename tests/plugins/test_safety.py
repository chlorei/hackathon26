from dreamkit.testing import replay


def test_slows_a_fast_chunk_and_the_manager_accepts_it():
    r = replay("success_apple_bowl", plugins=[{"safety": {"max_speed_rad_s": 0.3}}])
    assert r.result["safety"]["rows_slowed"] > 0
    assert not r.errors("safety")  # stricter only: never rejected


def test_untouched_under_a_generous_cap():
    r = replay("success_apple_bowl", plugins=[{"safety": {"max_speed_rad_s": 20.0}}])
    assert "safety" not in r.result and not r.errors("safety")
