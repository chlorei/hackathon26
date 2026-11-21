"""Writes dreamkit/theme/icons.json: the lucide icons plugin pages draw in their header (no React in a page).

Takes every `icon="..."` / `icon = "..."` named by a built-in or external plugin, plus a few spare ones, from the
panel's lucide-react package. Run after adding a plugin with a new icon: python tools/make_icons.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LUCIDE = ROOT / "ui" / "node_modules" / "lucide-react" / "dist" / "esm" / "icons"
OUT = ROOT / "dreamkit" / "theme" / "icons.json"
SPARE = ["Plug", "Hand", "ShieldCheck", "ShieldAlert", "Gauge", "Sparkles", "Activity", "AlertTriangle", "Repeat",
         "Timer", "Eye", "Wallet", "Zap", "Crosshair", "Move", "Bell", "BarChart3", "Brain", "Rocket"]


def kebab(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z0-9])", r"-\1", name).lower()


def node(name: str) -> list | None:
    path = LUCIDE / f"{kebab(name)}.mjs"
    if not path.is_file():
        return None
    source = path.read_text()
    alias = re.search(r"from '\./([\w-]+\.mjs)'", source)  # an old name re-exports the current icon
    if "node:" not in source and alias:
        source = (LUCIDE / alias.group(1)).read_text()
    start = source.find("[", source.find("node:"))
    if source.find("node:") < 0 or start < 0:
        return None
    depth, end = 0, start
    for end, ch in enumerate(source[start:], start):  # the node array, by its matching bracket
        depth += {"[": 1, "]": -1}.get(ch, 0)
        if depth == 0:
            break
    text = re.sub(r",\s*key:\s*\"[^\"]*\"", "", source[start:end + 1])  # React keys are not attributes
    text = re.sub(r"(\{|,)\s*([a-zA-Z][\w-]*)\s*:", r'\1"\2":', text)  # {cx: "1"} -> {"cx": "1"}
    return json.loads(text)


def main() -> None:
    names = set(SPARE)
    for f in [*ROOT.glob("dreamkit/plugins/*/plugin.py"), *ROOT.glob("plugins-external/*/*/*.py")]:
        names |= set(re.findall(r"icon\s*=\s*\"([A-Z]\w+)\"", f.read_text()))
    icons = {n: v for n in sorted(names) if (v := node(n)) is not None}
    OUT.write_text(json.dumps(icons, separators=(",", ":")))
    print(f"{len(icons)} icons -> {OUT.relative_to(ROOT)}; missing: {sorted(names - set(icons)) or 'none'}")


if __name__ == "__main__":
    main()
