"""The rules of PLUGIN_STANDARD.md that can be checked on a plugin class, and the error that names them.

Used three ways: Plugin.__init_subclass__ raises on the first broken rule (a bad plugin does not import),
`dreamkit check` lists every broken rule, and the manager refuses what only shows at load time (conflicts)."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

API_VERSION = "1"
GROUPS = ("core", "diagnosis", "safety", "output")
CAPABILITIES = ("observe", "modify_actions", "control")
HOOKS = ("on_run_start", "on_episode_start", "before_model", "after_model", "on_step", "on_episode_end",
         "on_run_end")
## unit -> the name suffix it requires (§4). Dimensionless numbers declare unit="" and carry no suffix.
UNIT_SUFFIX = {"m": "_m", "cm": "_cm", "s": "_s", "ms": "_ms", "Hz": "_hz", "deg": "_deg", "rad": "_rad",
               "rad/s": "_rad_s", "N": "_n", "fps": "_fps"}
NAME_RE = re.compile(r"^[a-z]+(-[a-z]+){0,2}$")  # 1-3 words, kebab-case (§1; one word allowed, see changelog)


class PluginStandardError(Exception):
    """A plugin breaks PLUGIN_STANDARD.md. The message names the rule, the place and the fix."""

    def __init__(self, problems: list["Problem"]):
        self.problems = problems
        super().__init__("\n".join(str(p) for p in problems))


class Problem:
    def __init__(self, section: str, where: str, what: str, fix: str):
        self.section, self.where, self.what, self.fix = section, where, what, fix

    def __str__(self) -> str:
        return f"{self.where} — {self.what} ({self.section}). {self.fix}"

    def as_dict(self) -> dict[str, str]:
        return {"section": self.section, "where": self.where, "what": self.what, "fix": self.fix}


class Manifest(BaseModel):
    """§2: what a plugin is and what it may do. Built-ins set it as a class attribute; packages may also put it in
    pyproject.toml [tool.dreamkit] (same fields)."""

    api: str = API_VERSION
    ## "loop": hooks inside your control loop (§2). "app": a page, an API and commands of its own (§13).
    kind: Literal["loop", "app"] = "loop"
    ## The tab's label of an app plugin ("Gym"); loop plugins show their name.
    title: str | None = None
    group: Literal["core", "diagnosis", "safety", "output"] = "diagnosis"
    icon: str = "Plug"  # a lucide-react icon name
    capability: Literal["observe", "modify_actions", "control"] = "observe"
    hooks: list[str] = Field(default_factory=list)
    step_rate_hz: float = Field(20, ge=1, le=20)
    needs_frames: bool = False
    requires: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    ## What it reads: "robot" (state, gripper, command, cameras — every robot) and/or "world" (positions,
    ## contacts — only the simulator). A plugin that needs the world is off on a real robot ("sim only").
    needs: list[Literal["robot", "world"]] = Field(default_factory=lambda: ["robot"])
    ## §7a: an optional page shown in the panel, relative to the plugin's module folder ("ui/index.html").
    ui: str | None = None

    @field_validator("ui")
    @classmethod
    def _ui(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parts = Path(value).parts
        if not value.endswith(".html") or Path(value).is_absolute() or ".." in parts:
            raise ValueError(f"ui {value!r} must be a relative .html path inside the module folder, e.g. ui/index.html")
        return value

    @field_validator("hooks")
    @classmethod
    def _hooks(cls, value: list[str]) -> list[str]:
        unknown = [h for h in value if h not in HOOKS]
        if unknown:
            raise ValueError(f"unknown hooks {unknown}; hooks are {', '.join(HOOKS)}")
        return value


class NoParams(BaseModel):
    """A plugin without settings."""


def _where(cls: type, attr: str | None = None) -> str:
    origin = cls.__dict__.get("_dk_origin")
    try:
        if origin is not None:  # recorded while the class was being defined (works for any loader)
            file = Path(origin[0])
            text = file.read_text().splitlines()
            start = next((i + 1 for i in range(origin[1] - 1, -1, -1)
                          if re.match(rf"\s*class {re.escape(cls.__name__)}\b", text[i])), origin[1])
            lines = text[start - 1:]
        else:
            file = Path(inspect.getsourcefile(cls) or "?")
            lines, start = inspect.getsourcelines(cls)
        line = start
        if attr:
            for i, text in enumerate(lines):
                if re.match(rf"\s+{re.escape(attr)}\b", text):
                    line = start + i
                    break
        try:
            file = file.relative_to(Path.cwd())
        except ValueError:
            pass
        return f"{file}:{line}"
    except (TypeError, OSError):
        return cls.__qualname__


def _where_field(cls: type, params: type, field: str) -> str:
    """file:line of a Params field, searched in the plugin's own file."""
    origin = cls.__dict__.get("_dk_origin")
    try:
        file = Path(origin[0]) if origin else Path(inspect.getsourcefile(params) or "?")
        for i, text in enumerate(file.read_text().splitlines(), 1):
            if re.match(rf"\s+{re.escape(field)}\s*:", text):
                try:
                    file = file.relative_to(Path.cwd())
                except ValueError:
                    pass
                return f"{file}:{i}"
    except (OSError, TypeError):
        pass
    return f"{cls.__name__}.Params.{field}"


def check_class(cls: type) -> list[Problem]:
    """Every rule a plugin class can break on its own (§1, §2, §4, §5)."""
    from .plugin import Plugin  # noqa: PLC0415 - circular at import time

    out: list[Problem] = []
    name = getattr(cls, "name", None)
    m0 = getattr(cls, "manifest", None)
    if isinstance(m0, Manifest) and m0.kind != "loop":
        out.append(Problem("§13", _where(cls, "manifest"), 'a Plugin class with kind="app"',
                           "App plugins are dreamkit.sdk.App objects under the dreamkit.apps entry point."))
    if not isinstance(name, str) or not NAME_RE.match(name):
        out.append(Problem("§1", _where(cls, "name"), f"name {name!r} is not 1-3 kebab-case words",
                           'Set name = "gripper-check" (lowercase words joined by "-").'))
    # docstring = the UI description (§4)
    doc = (cls.__doc__ or "").strip()
    first = doc.split("\n\n")[0].replace("\n", " ").strip()
    if not first:
        out.append(Problem("§4", _where(cls), "the class has no docstring",
                           'Add a one-line docstring, e.g. """Catches a gripper closed on nothing."""'))
    else:
        word = first.split()[0]
        if len(first) > 90:
            out.append(Problem("§4", _where(cls), f"the docstring's first sentence is {len(first)} characters (max 90)",
                               "Shorten it; details go in the module docstring or README."))
        if not first.endswith("."):
            out.append(Problem("§4", _where(cls), "the docstring's first sentence does not end with a period",
                               "End it with a period."))
        if first.rstrip(".").count(". ") > 0:
            out.append(Problem("§4", _where(cls), "the docstring's first paragraph is more than one sentence",
                               "Keep one sentence; put the rest after a blank line."))
        if not (word[:1].isupper() and word.lower().endswith("s")):
            out.append(Problem("§4", _where(cls), f"the docstring starts with {word!r}, not a verb",
                               'Start with what it does: "Catches …", "Records …", "Stops …".'))
    # manifest (§2)
    manifest = cls.__dict__.get("manifest")
    if not isinstance(manifest, Manifest):
        out.append(Problem("§2", _where(cls, "manifest"), "no manifest",
                           'Add manifest = Manifest(group=..., icon=..., capability=..., hooks=[...]).'))
    else:
        if manifest.api != API_VERSION:
            out.append(Problem("§2", _where(cls, "manifest"), f"api {manifest.api!r}, this dreamkit runs api "
                               f"{API_VERSION!r}", f'Set api="{API_VERSION}" and check the changelog.'))
        implemented = [h for h in HOOKS if getattr(cls, h, None) is not getattr(Plugin, h, None)]
        missing = [h for h in implemented if h not in manifest.hooks]
        extra = [h for h in manifest.hooks if h not in implemented]
        if missing:
            out.append(Problem("§2", _where(cls, missing[0]), f"hooks {missing} are implemented but not declared",
                               "Add them to manifest hooks (undeclared hooks are never called)."))
        if extra:
            out.append(Problem("§2", _where(cls, "manifest"), f"hooks {extra} are declared but not implemented",
                               "Remove them from manifest hooks."))
        if name in manifest.conflicts or name in manifest.requires:
            out.append(Problem("§2", _where(cls, "manifest"), "the plugin requires or conflicts with itself",
                               "Remove its own name."))
    # Params (§4)
    params = getattr(cls, "Params", None)
    if not (isinstance(params, type) and issubclass(params, BaseModel)):
        out.append(Problem("§4", _where(cls, "Params"), "Params is not a pydantic model",
                           "Define class Params(BaseModel) with Field(..., description=...) for each setting."))
    else:
        for field_name, info in params.model_fields.items():
            where = _where_field(cls, params, field_name)
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
            if not info.description:
                out.append(Problem("§4", where, f"Params.{field_name} has no description",
                                   f'Add description="..." to Field() of {field_name}.'))
            if not re.match(r"^[a-z][a-z0-9_]*$", field_name):
                out.append(Problem("§4", where, f"Params.{field_name} is not snake_case", "Rename it in snake_case."))
            annotation = info.annotation
            numeric = annotation in (int, float) and annotation is not bool
            unit = extra.get("unit")
            if numeric and unit is None:
                out.append(Problem("§4", where, f"Params.{field_name} is a number without a unit",
                                   'Add json_schema_extra={"unit": "s"} (or "" for a count or a fraction).'))
            if unit:
                suffix = UNIT_SUFFIX.get(unit)
                if suffix is None:
                    out.append(Problem("§4", where, f"Params.{field_name} has unit {unit!r}, which has no suffix",
                                       f"Use one of {', '.join(UNIT_SUFFIX)}."))
                elif not field_name.endswith(suffix):
                    out.append(Problem("§4", where, f"Params.{field_name} has no unit suffix",
                                       f'Rename to {field_name}{suffix} and keep unit="{unit}".'))
            elif numeric and any(field_name.endswith(s) for s in UNIT_SUFFIX.values()):
                out.append(Problem("§4", where, f"Params.{field_name} ends with a unit suffix but has no unit",
                                   'Add the matching unit to json_schema_extra.'))
    return out


def check_pyproject(path: Path, cls: type | None = None) -> list[Problem]:
    """An external plugin package: pyproject [tool.dreamkit] (if any) agrees with the class, entry point, README,
    tests (§1, §2, §9)."""
    import tomllib  # noqa: PLC0415

    out: list[Problem] = []
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return [Problem("§1", str(path), "no pyproject.toml", "Run dreamkit new <name> for the layout.")]
    data = tomllib.loads(pyproject.read_text())
    eps = data.get("project", {}).get("entry-points", {}).get("dreamkit.plugins", {})
    if not eps:
        out.append(Problem("§2", str(pyproject), 'no [project.entry-points."dreamkit.plugins"]',
                           'Add gripper-check = "gripper_check:GripperCheck".'))
    tool = data.get("tool", {}).get("dreamkit")
    if tool is not None and cls is not None and isinstance(getattr(cls, "manifest", None), Manifest):
        declared = cls.manifest.model_dump()
        for key, value in tool.items():
            if key in declared and declared[key] != value:
                out.append(Problem("§2", str(pyproject), f"[tool.dreamkit] {key}={value!r} disagrees with the "
                                   f"class manifest ({declared[key]!r})", "Make them the same."))
    if not (path / "README.md").is_file():
        out.append(Problem("§12", str(path), "no README.md", "Add one; dreamkit check --readme writes the params table."))
    tests = list((path / "tests").glob("test_*.py")) if (path / "tests").is_dir() else []
    if not tests:
        out.append(Problem("§9", str(path), "no tests/test_*.py", "Add replay tests: one passing, one failing case."))
    return out


# -- §7a: a plugin's own page -----------------------------------------------------------------------------------------
_UI_FILES = (".html", ".htm", ".css", ".js", ".mjs")
_EXTERNAL = re.compile(r"""(?:https?:)?//[a-z0-9.-]+\.[a-z]{2,}|https?://""", re.I)
_CSS_COLOR = re.compile(r"#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(", re.I)
_NAMED_COLOR = re.compile(r"\b(?:white|black|red|green|blue|gray|grey|yellow|orange|purple|pink|brown|navy|teal|"
                          r"silver|gold|maroon|olive|lime|aqua|cyan|magenta|fuchsia)\b", re.I)
_COLOR_PROPS = re.compile(r"(?:^|[;{\s\"'])(?:color|background(?:-color)?|border(?:-[a-z]+)*|outline(?:-color)?|fill|"
                          r"stroke|box-shadow|text-shadow|caret-color|accent-color)\s*:\s*([^;{}\"'<>]*)", re.I)
_DECL = re.compile(r"(?:^|[;{])\s*[-a-z]+\s*:\s*([^;{}]*)", re.I)
_FONT_FAMILY = re.compile(r"font-family\s*:\s*([^;{}\"]*)", re.I)
_STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.I | re.S)
_STYLE_ATTR = re.compile(r"""style\s*=\s*(["'])(.*?)\1""", re.I | re.S)
_SCRIPT_BLOCK = re.compile(r"<script[^>]*>(.*?)</script>", re.I | re.S)
_JS_COLOR = re.compile(r"""["'`]\s*(#[0-9a-f]{3,8}|(?:rgba?|hsla?)\([^"'`]*\))\s*["'`]""", re.I)


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _css_problems(where: str, css: str, offset_text: str, offset: int) -> list[Problem]:
    out: list[Problem] = []
    for m in _DECL.finditer(css):
        value = re.sub(r"var\([^)]*\)", "", m.group(1))
        hit = _CSS_COLOR.search(value)
        if hit:
            out.append(Problem("§7a", f"{where}:{_line(offset_text, offset + m.start())}",
                               f"colour {hit.group(0)!r} outside the theme's variables",
                               "Use a theme variable, e.g. var(--dk-ok) or var(--dk-line)."))
    for m in _COLOR_PROPS.finditer(css):
        value = re.sub(r"var\([^)]*\)", "", m.group(1))
        hit = _NAMED_COLOR.search(value)
        if hit:
            out.append(Problem("§7a", f"{where}:{_line(offset_text, offset + m.start())}",
                               f"named colour {hit.group(0)!r} outside the theme's variables",
                               "Use a theme variable, e.g. var(--dk-ink)."))
    for m in _FONT_FAMILY.finditer(css):
        value = m.group(1).strip()
        if value and not re.fullmatch(r"(?:var\([^)]*\)\s*,?\s*)+|inherit|initial|unset", value):
            out.append(Problem("§7a", f"{where}:{_line(offset_text, offset + m.start())}",
                               f"own font {value!r}", "Use var(--dk-font-sans) or var(--dk-font-mono)."))
    if "@font-face" in css:
        out.append(Problem("§7a", where, "@font-face: a font of its own", "Remove it; the theme sets the fonts."))
    return out


def check_ui(module_dir: Path, ui: str) -> list[Problem]:
    """§7a: the page exists, links nothing external, uses only the theme's colours and fonts."""
    page = module_dir / ui
    if not page.is_file():
        return [Problem("§7a", str(page), "manifest ui points to a missing file", "Add the page or fix the path.")]
    out: list[Problem] = []
    for f in sorted(p for p in page.parent.rglob("*") if p.is_file() and p.suffix.lower() in _UI_FILES):
        text = f.read_text(errors="replace")
        where = str(f)
        for m in _EXTERNAL.finditer(text):
            out.append(Problem("§7a", f"{where}:{_line(text, m.start())}", f"external URL {m.group(0)!r}",
                               "Ship the file next to index.html; read data only from /media/<run_id>/<plugin>/."))
        if f.suffix.lower() == ".css":
            out += _css_problems(where, text, text, 0)
            continue
        if f.suffix.lower() in (".html", ".htm"):
            for m in _STYLE_BLOCK.finditer(text):
                out += _css_problems(where, m.group(1), text, m.start(1))
            for m in _STYLE_ATTR.finditer(text):
                out += _css_problems(where, ";" + m.group(2), text, m.start(2))
            scripts = [(m.group(1), m.start(1)) for m in _SCRIPT_BLOCK.finditer(text)]
        else:
            scripts = [(text, 0)]
        for code, start in scripts:
            for m in _JS_COLOR.finditer(code):
                out.append(Problem("§7a", f"{where}:{_line(text, start + m.start())}",
                                   f"colour {m.group(1)!r} in a script",
                                   "Use a theme variable: 'var(--dk-bad)' or a .dk-* class."))
    return out


def params_schema(params: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema the panel builds the settings form from (descriptions, units, choices, bounds)."""
    return params.model_json_schema()
