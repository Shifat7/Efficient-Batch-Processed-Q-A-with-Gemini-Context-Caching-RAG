"""Utility helpers: token estimation, metrics logging, timing."""
from __future__ import annotations
import json
import os
import time
from typing import Any


def calculate_token_usage(text: str) -> int:
    """Estimate token count for a text string.

    Uses a simple 4-chars-per-token heuristic. For exact counts use
    ``model.count_tokens(text)`` from the google-generativeai SDK.

    Args:
        text: Input text.

    Returns:
        Non-negative estimated token count.
    """
    if not text:
        return 0
    return max(0, len(text) // 4)


def log_metrics(metrics: dict[str, Any]) -> None:
    """Print and persist a metrics dict as a timestamped JSON file under logs/.

    Args:
        metrics: Arbitrary key/value metrics to record.
    """
    print("\n=== Performance Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/metrics_{int(time.time())}.json"
    with open(log_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {log_file}")


class Timer:
    """Context manager for timing a code block.

    Usage::

        with Timer("embed") as t:
            build_index(chunks)
        print(t.interval)  # seconds elapsed
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.interval: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.interval = time.perf_counter() - self._start
        if self.name:
            print(f"  {self.name} took {self.interval:.4f}s")
