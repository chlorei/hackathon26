"""App plugins (PLUGIN_STANDARD.md §13): discovery, the standard, the panel's mounts, `dreamkit <app>`, the PDK."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from dreamkit import apps, pdk
from dreamkit.cli import main as cli_main
from dreamkit.server import create_app
from dreamkit.standard import Manifest

PAGE = '<!doctype html><link rel="stylesheet" href="/theme/dreamkit-theme.css"><p>hello app</p>'


def make_app(root: Path, name: str = "toolbox", *, needs=("robot",), page: str = PAGE, title: str = "Toolbox") -> Path:
    """A source folder with one app plugin: a page, a router with /ping, a command that returns 7."""
    src = root / name
    pkg = src / name
    (pkg / "web").mkdir(parents=True)
    (pkg / "web" / "index.html").write_text(page)
    (pkg / "__init__.py").write_text(textwrap.dedent(f'''
        from pathlib import Path
        from fastapi import APIRouter
        from dreamkit.sdk import App, Manifest

        def router(ctx):
            r = APIRouter()
            r.get("/ping")(lambda: {{"pong": True, "world": ctx.world}})
            return r

        app = App(name="{name}", manifest=Manifest(kind="app", title="{title}", icon="Wrench", needs={list(needs)!r}),
                  router=router, page=Path(__file__).parent / "web", commands=lambda argv: 7 if argv == ["hi"] else 1,
                  description="A test app")
    '''))
    (src / "pyproject.toml").write_text(textwrap.dedent(f'''
        [project]
        name = "dreamkit-{name}"
        version = "0.1.0"
        [project.entry-points."dreamkit.apps"]
        {name} = "{name}:app"
        [tool.dreamkit]
        kind = "app"
        title = "{title}"
    '''))
    (src / "README.md").write_text(f"# {name}\n")
    (src / "tests").mkdir()
    (src / "tests" / "test_it.py").write_text("def test_ok():\n    assert True\n")
    return src


@pytest.fixture()
def on_path(monkeypatch):
    def add(src: Path) -> None:
        monkeypatch.setenv("DREAMKIT_PLUGIN_PATH", str(src))
    return add


def test_discovered_checked_and_aliased(tmp_path, on_path):
    on_path(make_app(tmp_path))
    found, problems = apps.discover()
    assert "toolbox" in found and not [p for p in problems if "toolbox" in p]
    app = found["toolbox"]
    assert app.aliases == ["toolbox"]  # the title is the name: no second alias
    assert apps.check_app(app) == []


def test_panel_mounts_api_page_theme_and_marks_sim_only(tmp_path, on_path):
    on_path(make_app(tmp_path, "worldly", needs=("robot", "world"), title="Worldly"))
    (tmp_path / "nosim").mkdir()
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"sim_path": str(tmp_path / "nosim"), "runs_dir": str(tmp_path / "runs")}))
    client = TestClient(create_app(cfg))
    listed = {a["name"]: a for a in client.get("/api/apps").json()["apps"]}
    if listed["worldly"]["inactive"] is None:  # a dreamlabs wheel installed here brings the simulator along
        pytest.skip("dream_examples is installed: the world is available")
    assert "simulator" in listed["worldly"]["inactive"]
    assert client.get("/apps/worldly/api/ping").json() == {"pong": True, "world": False}
    assert "hello app" in client.get("/apps/worldly/").text
    theme = client.get("/theme/dreamkit-theme.css")
    assert theme.status_code == 200 and "--dk-base" in theme.text


def test_cli_runs_the_apps_commands(tmp_path, on_path):
    on_path(make_app(tmp_path, "kitbag", title="Bag"))
    assert cli_main(["kitbag", "hi"]) == 7
    assert cli_main(["bag", "hi"]) == 7  # the one-word title is an alias


def test_refused_without_the_theme(tmp_path, on_path):
    on_path(make_app(tmp_path, "plain", page="<p>no theme</p>"))
    found, problems = apps.discover()
    assert "plain" not in found
    assert any("theme" in p and "§13" in p for p in problems)


def test_reserved_name_and_loop_kind_refused():
    bad = apps.App(name="history", manifest=Manifest(kind="app", title="H"), commands=lambda a: 0)
    assert any("reserved" in str(p) for p in apps.check_app(bad))
    loop = apps.App(name="thing", manifest=Manifest(title="T"), commands=lambda a: 0)
    assert any('kind is not "app"' in str(p) for p in apps.check_app(loop))


def test_pdk_check_test_bench_for_an_app(tmp_path):
    src = make_app(tmp_path, "toolbox")
    ok, lines = pdk.check(str(src))
    assert ok, lines
    assert any("kind=app" in line for line in lines)
    ok, out = pdk.test(str(src))
    assert ok, out
    ok, lines = pdk.bench(str(src))
    assert ok and "no per-step bench" in lines[0]


def test_an_app_installed_while_the_panel_runs_gets_its_tab(tmp_path, monkeypatch):
    """No restart: /api/apps looks again, and /apps/<name>/ serves the new app."""
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs")}))
    monkeypatch.delenv("DREAMKIT_PLUGIN_PATH", raising=False)
    client = TestClient(create_app(cfg))
    assert "latecomer" not in {a["name"] for a in client.get("/api/apps").json()["apps"]}
    assert client.get("/apps/latecomer/api/ping").status_code == 404
    monkeypatch.setenv("DREAMKIT_PLUGIN_PATH", str(make_app(tmp_path, "latecomer", title="Late")))
    assert "latecomer" in {a["name"] for a in client.get("/api/apps").json()["apps"]}
    assert client.get("/apps/latecomer/api/ping").json()["pong"] is True
    assert "hello app" in client.get("/apps/latecomer/").text


def test_forget_drops_a_deleted_clone(tmp_path):
    import shutil
    import sys

    src = make_app(tmp_path, "ephemeral")
    found, _ = apps.discover(extra=[src])
    assert "ephemeral" in sys.modules and str(src) in sys.path
    apps.forget(src)
    shutil.rmtree(src)
    assert "ephemeral" not in sys.modules and str(src) not in sys.path


def test_an_app_can_be_uninstalled_from_its_tab(tmp_path, monkeypatch):
    """DELETE /api/apps/<name>: pip uninstall of the app's distribution, then the tab is gone; an app loaded from a
    source folder is not pip's to remove and says so."""
    import subprocess

    from dreamkit import registry

    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs")}))
    monkeypatch.setenv("DREAMKIT_PLUGIN_PATH", str(make_app(tmp_path, "shelf", title="Shelf")))
    client = TestClient(create_app(cfg))
    res = client.delete("/api/apps/shelf")
    assert res.status_code == 400 and "DREAMKIT_PLUGIN_PATH" in res.json()["detail"]
    assert client.delete("/api/apps/nothing").status_code == 404

    real_discover, removed, calls = apps.discover, [], []

    def discover(extra=None):  # the same app, as if pip had installed it; gone once uninstalled
        found, problems = real_discover(extra)
        if removed:
            return {}, problems
        for a in found.values():
            a.origin = "pip dreamkit-shelf"
        return found, problems

    def uninstall_dist(dist):
        calls.append(dist)
        removed.append(True)
        return subprocess.CompletedProcess([dist], 0, "Successfully uninstalled\n", "")

    monkeypatch.setattr(apps, "discover", discover)
    monkeypatch.setattr(registry, "uninstall_dist", uninstall_dist)
    res = client.delete("/api/apps/shelf")
    assert res.status_code == 200 and res.json()["title"] == "Shelf" and calls == ["dreamkit-shelf"]
    assert "shelf" not in {a["name"] for a in client.get("/api/apps").json()["apps"]}
    assert client.get("/apps/shelf/api/ping").status_code == 404
