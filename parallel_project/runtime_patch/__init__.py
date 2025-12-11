"""
runtime_patch - Runtime patches for LLM inference optimization.

This package contains experimental patches and utilities for optimizing
MLC-LLM inference performance through compute/communication overlap.

15-418 Parallel Computer Architecture Project
"""

from .stream_overlap import (
    StreamManager,
    OverlapStrategy,
    create_overlap_context,
    AsyncMemoryManager,
)

__all__ = [
    "StreamManager",
    "OverlapStrategy",
    "create_overlap_context",
    "AsyncMemoryManager",
]

