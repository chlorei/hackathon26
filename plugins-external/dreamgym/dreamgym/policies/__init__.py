"""Policy plugins and the spec strings the CLI takes.

- `dream-1` / `beta`: Dream Labs' base model, `dreamlabs/dream-1-beta`.
- `dream:<model id>` or a bare model id: any model your account may run (a fine-tune `dream-1-ft:...`, flash).
  `dreamgym models` lists them.
- `scripted` / `scripted:<noise_m>`: the IK expert (free, offline).
"""

from __future__ import annotations

from .dream_policy import DreamPolicy
from .scripted import ScriptedPickPlace

BETA = "dreamlabs/dream-1-beta"
ALIASES = {"dream-1": BETA, "beta": BETA}


def make_policy(spec: str, *, action_limit: int = 48):
    if spec == "scripted" or spec.startswith("scripted:"):
        noise = float(spec.split(":", 1)[1]) if ":" in spec else 0.0
        return ScriptedPickPlace(noise_m=noise)
    checkpoint = ALIASES.get(spec, spec.removeprefix("dream:"))
    price = 2.0 if "flash" in checkpoint else 3.0  # $/robot-hour at launch pricing
    return DreamPolicy(checkpoint, action_limit=action_limit, price_per_hour=price, label=spec)


__all__ = ["DreamPolicy", "ScriptedPickPlace", "make_policy", "BETA"]
