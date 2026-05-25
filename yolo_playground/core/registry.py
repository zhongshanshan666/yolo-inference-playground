from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

BACKENDS: dict[str, type] = {}
MODELS: dict[str, type] = {}
SOURCES: dict[str, type] = {}
SINKS: dict[str, type] = {}


def register_backend(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        BACKENDS[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        MODELS[name] = cls
        return cls

    return decorator


def register_source(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        SOURCES[name] = cls
        return cls

    return decorator


def register_sink(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        SINKS[name] = cls
        return cls

    return decorator
