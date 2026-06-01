"""OpenTelemetry hooks with a graceful no-op fallback.

If ``opentelemetry-api`` is installed, real spans are emitted. Otherwise every
call becomes a cheap no-op, so the SDK has zero hard dependency on OTel.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

try:  # pragma: no cover - exercised only when otel is installed
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("agentraft")
    _OTEL = True
except Exception:  # pragma: no cover
    _TRACER = None
    _OTEL = False


class _NoopSpan:
    def set_attribute(self, *_args, **_kwargs) -> None: ...
    def set_status(self, *_args, **_kwargs) -> None: ...
    def record_exception(self, *_args, **_kwargs) -> None: ...
    def add_event(self, *_args, **_kwargs) -> None: ...


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span (real if OTel present, no-op otherwise)."""
    if not _OTEL or _TRACER is None:
        yield _NoopSpan()
        return
    with _TRACER.start_as_current_span(name) as s:  # pragma: no cover
        for k, v in attributes.items():
            try:
                s.set_attribute(k, v)
            except Exception:
                pass
        yield s


def otel_available() -> bool:
    return _OTEL
