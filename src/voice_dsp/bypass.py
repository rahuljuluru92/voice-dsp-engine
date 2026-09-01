"""Stage 5: bypass guard — emits silence instead of raw microphone input
if the processing path raises, produces non-finite output, or falls
behind a configured safety time budget.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


class BypassGuard:
    def __init__(
        self,
        process_fn: Callable[[np.ndarray], np.ndarray],
        max_process_seconds: float | None = None,
    ):
        self.process_fn = process_fn
        self.max_process_seconds = max_process_seconds
        self.bypass_count = 0

    def safe_process(self, x: np.ndarray) -> np.ndarray:
        try:
            start = time.perf_counter() if self.max_process_seconds is not None else None
            y = self.process_fn(x)

            if start is not None:
                elapsed = time.perf_counter() - start
                if elapsed > self.max_process_seconds:
                    self.bypass_count += 1
                    return np.zeros_like(x)

            if not np.all(np.isfinite(y)):
                self.bypass_count += 1
                return np.zeros_like(x)

            return y
        except Exception:
            self.bypass_count += 1
            return np.zeros_like(x)
