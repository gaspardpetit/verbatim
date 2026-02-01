"""Artifact cache helpers for in-memory and file-backed storage."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Protocol


class ArtifactCache(Protocol):
    def get_text(self, key: str) -> str: ...

    def set_text(self, key: str, value: str) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def set_bytes(self, key: str, value: bytes) -> None: ...

    def read_text(self, key: str) -> str: ...

    def read_bytes(self, key: str) -> bytes: ...

    def bytes_io(self, key: str) -> io.BytesIO: ...

    def delete(self, key: str) -> None: ...


class BaseArtifactCache(ArtifactCache):
    def read_text(self, key: str) -> str:
        return self.get_text(key)

    def read_bytes(self, key: str) -> bytes:
        return self.get_bytes(key)

    def bytes_io(self, key: str) -> io.BytesIO:
        return io.BytesIO(self.read_bytes(key))

    def delete(self, key: str) -> None:
        raise NotImplementedError


@dataclass
class InMemoryArtifactCache(BaseArtifactCache):
    _text: Dict[str, str] = field(default_factory=dict)
    _bytes: Dict[str, bytes] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def get_text(self, key: str) -> str:
        with self._lock:
            return self._text.get(key, "")

    def set_text(self, key: str, value: str) -> None:
        with self._lock:
            self._text[key] = value

    def get_bytes(self, key: str) -> bytes:
        with self._lock:
            return self._bytes.get(key, b"")

    def set_bytes(self, key: str, value: bytes) -> None:
        with self._lock:
            self._bytes[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._text.pop(key, None)
            self._bytes.pop(key, None)


@dataclass
class FileBackedArtifactCache(InMemoryArtifactCache):
    base_dir: Optional[str] = "."

    def _resolve_path(self, key: str) -> Optional[str]:
        if os.path.isabs(key):
            return os.path.normpath(key)
        if not self.base_dir:
            return None
        normalized = os.path.normpath(key)
        if os.path.exists(normalized):
            return normalized
        base_norm = os.path.normpath(self.base_dir)
        if not os.path.isabs(base_norm):
            if normalized == base_norm or normalized.startswith(base_norm + os.sep):
                return normalized
        return os.path.join(self.base_dir, normalized)

    def get_text(self, key: str) -> str:
        cached = super().get_text(key)
        if cached:
            return cached
        path = self._resolve_path(key)
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.read()
        super().set_text(key, data)
        return data

    def set_text(self, key: str, value: str) -> None:
        super().set_text(key, value)
        path = self._resolve_path(key)
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(value)

    def get_bytes(self, key: str) -> bytes:
        cached = super().get_bytes(key)
        if cached:
            return cached
        path = self._resolve_path(key)
        if not path or not os.path.exists(path):
            return b""
        with open(path, "rb") as fh:
            data = fh.read()
        super().set_bytes(key, data)
        return data

    def set_bytes(self, key: str, value: bytes) -> None:
        super().set_bytes(key, value)
        path = self._resolve_path(key)
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(value)

    def delete(self, key: str) -> None:
        super().delete(key)
        path = self._resolve_path(key)
        if not path or not os.path.exists(path):
            return
        try:
            os.remove(path)
        except OSError:
            pass
