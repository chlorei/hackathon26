"""dreamkit: plugins and a monitor for Dream-1, inside your own control loop.

    import dream, dreamkit
    client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")
"""

from .kit import Kit, wrap
from .plugin import Context, EndEpisode, Plugin, StopEpisode

__all__ = ["wrap", "Kit", "Plugin", "Context", "StopEpisode", "EndEpisode"]
__version__ = "0.2.0"
