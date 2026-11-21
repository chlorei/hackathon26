"""§7a on the server: a plugin's page gets the theme, stays in its folder, and reads only its own run files."""

import importlib.util
import sys

import yaml
from fastapi.testclient import TestClient

from dreamkit import server


def test_plugin_page_is_served_with_the_theme_and_kept_to_its_own_files(tmp_path, monkeypatch):
    pkg = tmp_path / "fakepage"
    (pkg / "ui").mkdir(parents=True)
    (pkg / "ui" / "index.html").write_text("<html><head></head><body>hi</body></html>")
    (pkg / "ui" / "app.js").write_text("1")
    (pkg / "secret.txt").write_text("no")
    (pkg / "__init__.py").write_text(
        "class Manifest: ui = 'ui/index.html'\nclass Fake:\n    manifest = Manifest\n")
    spec = importlib.util.spec_from_file_location("fakepage", pkg / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fakepage", mod)  # inspect.getfile finds a class's file through sys.modules
    spec.loader.exec_module(mod)
    monkeypatch.setattr(server, "discover", lambda dirs: ({"fake": {"cls": mod.Fake}}, []))

    run = tmp_path / "runs" / "r1"
    (run / "fake").mkdir(parents=True)
    (run / "fake" / "a.json").write_text("{}")
    (run / "recorder").mkdir()
    (run / "recorder" / "b.json").write_text("{}")
    cfg = tmp_path / "dreamkit.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": "runs", "plugins": []}))
    c = TestClient(server.create_app(cfg))

    page = c.get("/plugin-ui/fake/")
    assert page.status_code == 200 and server.THEME_LINK in page.text and '<base href="/plugin-ui/fake/">' in page.text
    assert c.get("/plugin-ui/fake/app.js").text == "1"
    assert c.get("/plugin-ui/fake/..%2Fsecret.txt").status_code == 404
    assert c.get("/plugin-ui/nope/").status_code == 404
    assert c.get("/theme/dreamkit-theme.css").status_code == 200

    null = {"origin": "null"}
    assert c.get("/media/r1/fake/a.json", headers=null).headers.get("access-control-allow-origin") == "null"
    assert "access-control-allow-origin" not in c.get("/media/r1/recorder/b.json", headers=null).headers
