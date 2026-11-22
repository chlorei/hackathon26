"""dreamkit init: a commented dreamkit.yaml with the default plugins, never over an existing one."""

from dreamkit import config
from dreamkit.cli import main


def test_init_writes_a_config_that_loads_and_never_overwrites(tmp_path, capsys):
    path = tmp_path / "dreamkit.yaml"
    assert main(["init", str(path)]) == 0
    cfg, _ = config.load(path)
    assert [p.name for p in cfg.plugins] == ["camera-check", "success-check", "failure-monitor", "recovery",
                                            "recorder", "harvester", "gripper-check", "budget-guard"]
    assert "# was the task done" in path.read_text()  # commented
    path.write_text("plugins: [recorder]\n")
    assert main(["init", str(path)]) == 1 and path.read_text() == "plugins: [recorder]\n"  # left as it is
