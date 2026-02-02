from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pyannote.audio.pipelines.utils.hook import ProgressHook

from verbatim.status_types import StatusHook, StatusProgress, StatusUpdate


class StatusProgressHook:
    def __init__(self, *, status_hook: Optional[StatusHook], state_prefix: str, enable_progress: bool):
        self._status_hook = status_hook
        self._state_prefix = state_prefix
        self._inner = ProgressHook() if enable_progress else None
        self._last_progress: Dict[str, float] = {}

    def __enter__(self) -> "StatusProgressHook":
        if self._inner is not None:
            self._inner.__enter__()
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback) -> bool:
        if self._inner is not None:
            handled = self._inner.__exit__(ex_type, ex_value, ex_traceback)
            return bool(handled)
        return False

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if self._inner is not None:
            self._inner(*args, **kwargs)
        if self._status_hook is None:
            return
        name = _extract_name(args, kwargs) or "progress"
        current, total = _extract_progress(args, kwargs)
        progress = None
        if current is not None:
            last = self._last_progress.get(name)
            if last is not None and current <= last:
                return
            self._last_progress[name] = float(current)
            progress = StatusProgress(
                current=float(current),
                finish=float(total) if total is not None else None,
                units="count",
            )
        self._status_hook(StatusUpdate(state=f"{self._state_prefix}:{name}", progress=progress))


def _extract_name(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[str]:
    for key in ("task", "step", "name", "stage", "pipeline", "activity"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return value
    if args and isinstance(args[0], str):
        return args[0]
    return None


def _extract_progress(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    current = None
    total = None
    for key in ("current", "completed", "value", "done", "progress"):
        value = kwargs.get(key)
        if isinstance(value, (int, float)):
            current = float(value)
            break
    for key in ("total", "maximum", "max", "length"):
        value = kwargs.get(key)
        if isinstance(value, (int, float)):
            total = float(value)
            break
    if current is None and len(args) >= 2 and isinstance(args[1], (int, float)):
        current = float(args[1])
    if total is None and len(args) >= 3 and isinstance(args[2], (int, float)):
        total = float(args[2])
    return current, total
