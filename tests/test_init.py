"""dreamkit init: a commented dreamkit.yaml with the default plugins, never over an existing one."""

from dreamkit import config
from dreamkit.cli import main
from dreamkit.manager import discover


def test_init_writes_a_config_that_loads_and_never_overwrites(tmp_path, capsys):
    path = tmp_path / "dreamkit.yaml"
    assert main(["init", str(path)]) == 0
    cfg, _ = config.load(path)
    names = [p.name for p in cfg.plugins]
    assert names[:1] == ["camera-check"] and {"failure-monitor", "recovery", "budget-guard"} <= set(names)
    found, _ = discover()
    for addon in ("vision-judge", "gripper-check"):  # listed when installed, else commented with how to install it
        assert (addon in names) == (addon in found)
        assert (addon in names) or f"# - {addon}" in path.read_text() and f"dreamkit install {addon}" in path.read_text()
    assert "# was the task done" in path.read_text()  # commented
    path.write_text("plugins: [recorder]\n")
    assert main(["init", str(path)]) == 1 and path.read_text() == "plugins: [recorder]\n"  # left as it is
