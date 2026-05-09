"""Small in-memory cache for expensive explainability computations."""

import hashlib
import json
from collections import OrderedDict
from threading import Lock
from typing import Callable


class CacheService:
    """Simple bounded cache."""

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self._lock = Lock()
        self._entries: OrderedDict[str, object] = OrderedDict()

    def make_key(self, namespace: str, components: object) -> str:
        payload = json.dumps(components, default=str, sort_keys=True)
        digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
        return f"{namespace}:{digest}"

    def get_or_compute(
        self,
        namespace: str,
        components: object,
        factory: Callable[[], object],
    ) -> object:
        key = self.make_key(namespace, components)
        with self._lock:
            if key in self._entries:
                value = self._entries.pop(key)
                self._entries[key] = value
                return value
        value = factory()
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            self._entries[key] = value
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
