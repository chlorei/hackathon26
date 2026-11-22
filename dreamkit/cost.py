"""What the model's time costs: the one place every dollar in dreamkit comes from (Live, History, reliability,
budget-guard). Billing is by robot-hour of the model: CALLS_PER_HOUR model calls (1.6 s of actions each) are one
hour, at the model's price per hour. A mock run costs nothing.

Prices are a setting: `prices: {dream-1: 3.0, flash: 2.0}` in dreamkit.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping

PRICES: dict[str, float] = {"dream-1": 3.0, "flash": 2.0}  # dollars per robot-hour of the model
CALLS_PER_HOUR = 2250  # model calls billed as one robot-hour (1.6 s of actions each)


def price_per_hour(model: str | None, mock: bool = False, prices: Mapping[str, float] | None = None) -> float:
    """Dollars per robot-hour for this model; 0 for a mock run or a policy that is not Dream Labs' (scripted,
    an expert plan). A flash model by name, else Dream-1's price."""
    name = (model or "").lower()
    if mock or "dream" not in name:  # only Dream Labs' models bill: a scripted or expert policy calls nothing
        return 0.0
    table = {**PRICES, **(prices or {})}
    for key in sorted(table, key=len, reverse=True):  # the most specific name first
        if key != "dream-1" and key in name:
            return float(table[key])
    return float(table["dream-1"])


def usd(calls: int, price: float, calls_per_hour: int = CALLS_PER_HOUR) -> float:
    """What `calls` model calls cost at `price` dollars per robot-hour."""
    return calls * price / calls_per_hour if calls_per_hour else 0.0
