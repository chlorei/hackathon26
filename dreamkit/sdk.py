"""dreamkit.sdk — everything a plugin author imports (PLUGIN_STANDARD.md §10.2).

    from dreamkit.sdk import Plugin, Manifest, Params, Field, PluginContext, reasons
    from dreamkit.sdk import App, AppContext                     # app plugins (§13)
    from dreamkit.sdk.testing import replay, fake_ctx, fixtures
"""

from pydantic import BaseModel as Params
from pydantic import Field

from . import reasons, testing
from .apps import App, AppContext
from .plugin import Plugin, PluginContext
from .standard import Manifest, PluginStandardError

__all__ = ["App", "AppContext", "Plugin", "Manifest", "Params", "Field", "PluginContext", "PluginStandardError", "reasons", "testing"]
