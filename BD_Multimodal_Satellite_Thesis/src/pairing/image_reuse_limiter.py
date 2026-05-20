"""Utilities for limiting image reuse in selected temporal pairs."""

from collections import Counter
from typing import Any


class ImageReuseTracker:
    """Track how many times each image appears across selected pairs."""

    def __init__(self, max_use: int) -> None:
        """Initialize a reuse tracker with a maximum image usage count."""
        if max_use < 1:
            raise ValueError("max_use must be at least 1.")
        self.max_use = int(max_use)
        self._usage: Counter[str] = Counter()

    def can_add(self, image_id_t1: str, image_id_t2: str) -> bool:
        """Return True when both images can be added without exceeding max_use."""
        return (
            self._usage[str(image_id_t1)] < self.max_use
            and self._usage[str(image_id_t2)] < self.max_use
        )

    def add_pair(self, image_id_t1: str, image_id_t2: str) -> None:
        """Add a pair to the tracker after validating reuse limits."""
        if not self.can_add(image_id_t1, image_id_t2):
            raise ValueError(
                "Cannot add pair because one or both images would exceed max_use."
            )
        self._usage[str(image_id_t1)] += 1
        self._usage[str(image_id_t2)] += 1

    def get_usage(self, image_id: str) -> int:
        """Return the current usage count for one image."""
        return int(self._usage[str(image_id)])

    def get_usage_summary(self) -> dict[str, Any]:
        """Return aggregate image reuse statistics."""
        if not self._usage:
            return {
                "unique_images": 0,
                "max_usage": 0,
                "mean_usage": 0.0,
                "usage_distribution": {},
            }

        usages = list(self._usage.values())
        distribution = Counter(usages)
        return {
            "unique_images": len(self._usage),
            "max_usage": int(max(usages)),
            "mean_usage": float(sum(usages) / len(usages)),
            "usage_distribution": {
                int(usage): int(count)
                for usage, count in sorted(distribution.items())
            },
        }
