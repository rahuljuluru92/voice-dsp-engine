"""Stage 5: bounded, drop-oldest queue so latency can't grow without
bound if the processing thread falls behind the audio callback thread.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class BoundedDropOldestQueue:
    """Thread-safe queue with a fixed capacity. When full, `put()` drops
    the oldest queued item to make room for the new one rather than
    blocking or growing.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._deque: deque[Any] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._dropped_count = 0

    def put(self, item: Any) -> None:
        with self._lock:
            if len(self._deque) == self._capacity:
                self._dropped_count += 1
            # deque(maxlen=...) drops the oldest (leftmost) item itself
            # once at capacity, which is exactly drop-oldest semantics.
            self._deque.append(item)

    def get(self) -> Any | None:
        with self._lock:
            if not self._deque:
                return None
            return self._deque.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()
