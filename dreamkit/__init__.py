"""dreamkit: plugins and a monitor for Dream-1, inside your own control loop.

    import dream, dreamkit
    client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")
"""

from .kit import Kit, wrap
from .plugin import Context, EndEpisode, Plugin, PluginContext, StopEpisode
from .standard import Manifest, PluginStandardError

__all__ = ["wrap", "Kit", "Plugin", "Manifest", "Context", "PluginContext", "StopEpisode", "EndEpisode",
           "PluginStandardError"]
__version__ = "0.3.0"
