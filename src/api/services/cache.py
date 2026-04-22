from collections import OrderedDict
from threading import Lock
from typing import Callable
from typing import Generic
from typing import TypeVar

T = TypeVar("T")


class ApiModelCache(Generic[T]):
    """Bounded in-memory cache for loaded model runtimes."""

    def __init__(self, max_entries: int = 1) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, T] = OrderedDict()
        self._lock = Lock()

    @property
    def entries(self) -> dict[str, T]:
        with self._lock:
            return dict(self._entries)

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        with self._lock:
            if key in self._entries:
                value = self._entries.pop(key)
                self._entries[key] = value
                return value

        value = loader()

        with self._lock:
            if key in self._entries:
                return self._entries[key]
            self._entries[key] = value
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return value
