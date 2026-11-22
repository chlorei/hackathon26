"""Kept for older imports: the dataset checks now live in dreamkit.datasets (datasets.validate / datasets.check).

Successful attempts are no longer exported: the harvester appends them to <data_dir>/datasets/harvest-<task>/ as
they end, so the dataset is always ready. `dreamkit datasets check <name>` runs validate_dataset on it."""

from __future__ import annotations

from .datasets import validate  # noqa: F401  (older code imports harvest.validate)
