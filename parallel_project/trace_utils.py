#!/usr/bin/env python3
"""
trace_utils.py - Performance tracing and profiling utilities.

This module provides utilities for parsing GPU profile logs, computing
statistics, and displaying performance summaries.

Key functions:
- parse_gpu_profile(log_file): Parse gpu_profile.log and compute averages
- print_summary(stats): Display parsed values in a formatted summary

Supports both real nvidia-smi logs and simulated mock logs.

15-418 Parallel Computer Architecture Project
"""

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GPUMetrics:
    """Metrics for a single GPU at a point in time."""

    timestamp: str = ""
    gpu_index: int = 0
    gpu_util_pct: float = 0.0
    mem_util_pct: float = 0.0
    power_w: float = 0.0
    temperature_c: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0


@dataclass
class GPUProfileStats:
    """Aggregated statistics from GPU profiling."""

    # Per-GPU statistics
    gpu_count: int = 0
    samples_per_gpu: Dict[int, int] = field(default_factory=dict)

    # Utilization
    avg_gpu_util_pct: Dict[int, float] = field(default_factory=dict)
    max_gpu_util_pct: Dict[int, float] = field(default_factory=dict)
    min_gpu_util_pct: Dict[int, float] = field(default_factory=dict)

    # Memory utilization
    avg_mem_util_pct: Dict[int, float] = field(default_factory=dict)
    max_mem_util_pct: Dict[int, float] = field(default_factory=dict)
    avg_mem_used_mb: Dict[int, float] = field(default_factory=dict)

    # Power
    avg_power_w: Dict[int, float] = field(default_factory=dict)
    max_power_w: Dict[int, float] = field(default_factory=dict)

    # Temperature
    avg_temp_c: Dict[int, float] = field(default_factory=dict)
    max_temp_c: Dict[int, float] = field(default_factory=dict)

    # Overall statistics
    total_samples: int = 0
    duration_s: float = 0.0
    is_mock: bool = False
    source_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_count": self.gpu_count,
            "total_samples": self.total_samples,
            "duration_s": self.duration_s,
            "is_mock": self.is_mock,
            "source_file": self.source_file,
            "per_gpu": {
                gpu_idx: {
                    "samples": self.samples_per_gpu.get(gpu_idx, 0),
                    "avg_gpu_util_pct": self.avg_gpu_util_pct.get(gpu_idx, 0),
                    "max_gpu_util_pct": self.max_gpu_util_pct.get(gpu_idx, 0),
                    "min_gpu_util_pct": self.min_gpu_util_pct.get(gpu_idx, 0),
                    "avg_mem_util_pct": self.avg_mem_util_pct.get(gpu_idx, 0),
                    "avg_mem_used_mb": self.avg_mem_used_mb.get(gpu_idx, 0),
                    "avg_power_w": self.avg_power_w.get(gpu_idx, 0),
                    "max_power_w": self.max_power_w.get(gpu_idx, 0),
                    "avg_temp_c": self.avg_temp_c.get(gpu_idx, 0),
                    "max_temp_c": self.max_temp_c.get(gpu_idx, 0),
                }
                for gpu_idx in sorted(self.samples_per_gpu.keys())
            },
        }


# =============================================================================
# Parsing Functions
# =============================================================================


def parse_gpu_profile(log_file: Union[str, Path]) -> GPUProfileStats:
    """Parse a GPU profile log file and compute statistics.

    Supports two formats:
    1. Mock format: timestamp, gpu_idx, gpu%, mem%, power, temp, mem_used, mem_total
    2. nvidia-smi dmon format: gpu pwr gtemp mtemp sm mem enc dec mclk pclk

    Args:
        log_file: Path to the gpu_profile.log file

    Returns:
        GPUProfileStats with computed averages and statistics
    """
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Profile log not found: {log_file}")

    stats = GPUProfileStats(source_file=str(log_file))

    # Collect metrics per GPU
    metrics_by_gpu: Dict[int, List[GPUMetrics]] = {}
    is_mock = False
    is_dmon = False

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for mock indicator in comments
            if line.startswith("#"):
                if "SIMULATED" in line.upper():
                    is_mock = True
                continue

            # Try to parse as mock format (comma-separated)
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    try:
                        metrics = GPUMetrics(
                            timestamp=parts[0],
                            gpu_index=int(parts[1]),
                            gpu_util_pct=float(parts[2]),
                            mem_util_pct=float(parts[3]),
                            power_w=float(parts[4]),
                            temperature_c=float(parts[5]),
                            memory_used_mb=float(parts[6]),
                            memory_total_mb=float(parts[7]),
                        )

                        gpu_idx = metrics.gpu_index
                        if gpu_idx not in metrics_by_gpu:
                            metrics_by_gpu[gpu_idx] = []
                        metrics_by_gpu[gpu_idx].append(metrics)

                    except (ValueError, IndexError):
                        continue

            # Try to parse as nvidia-smi dmon format (space-separated)
            else:
                parts = line.split()
                # dmon format: gpu pwr gtemp mtemp sm mem enc dec mclk pclk
                # We care about: gpu(0) pwr(1) gtemp(2) sm(4) mem(5)
                if len(parts) >= 6 and parts[0].isdigit():
                    is_dmon = True
                    try:
                        gpu_idx = int(parts[0])
                        metrics = GPUMetrics(
                            gpu_index=gpu_idx,
                            power_w=float(parts[1]) if parts[1] != "-" else 0,
                            temperature_c=float(parts[2]) if parts[2] != "-" else 0,
                            gpu_util_pct=float(parts[4]) if parts[4] != "-" else 0,
                            mem_util_pct=float(parts[5]) if parts[5] != "-" else 0,
                        )

                        if gpu_idx not in metrics_by_gpu:
                            metrics_by_gpu[gpu_idx] = []
                        metrics_by_gpu[gpu_idx].append(metrics)

                    except (ValueError, IndexError):
                        continue

    # Compute statistics
    stats.is_mock = is_mock
    stats.gpu_count = len(metrics_by_gpu)

    for gpu_idx, metrics_list in metrics_by_gpu.items():
        if not metrics_list:
            continue

        stats.samples_per_gpu[gpu_idx] = len(metrics_list)
        stats.total_samples += len(metrics_list)

        # Extract lists for statistics
        gpu_utils = [m.gpu_util_pct for m in metrics_list]
        mem_utils = [m.mem_util_pct for m in metrics_list]
        powers = [m.power_w for m in metrics_list if m.power_w > 0]
        temps = [m.temperature_c for m in metrics_list if m.temperature_c > 0]
        mem_used = [m.memory_used_mb for m in metrics_list if m.memory_used_mb > 0]

        # GPU Utilization
        stats.avg_gpu_util_pct[gpu_idx] = statistics.mean(gpu_utils) if gpu_utils else 0
        stats.max_gpu_util_pct[gpu_idx] = max(gpu_utils) if gpu_utils else 0
        stats.min_gpu_util_pct[gpu_idx] = min(gpu_utils) if gpu_utils else 0

        # Memory Utilization
        stats.avg_mem_util_pct[gpu_idx] = statistics.mean(mem_utils) if mem_utils else 0
        stats.max_mem_util_pct[gpu_idx] = max(mem_utils) if mem_utils else 0
        stats.avg_mem_used_mb[gpu_idx] = statistics.mean(mem_used) if mem_used else 0

        # Power
        stats.avg_power_w[gpu_idx] = statistics.mean(powers) if powers else 0
        stats.max_power_w[gpu_idx] = max(powers) if powers else 0

        # Temperature
        stats.avg_temp_c[gpu_idx] = statistics.mean(temps) if temps else 0
        stats.max_temp_c[gpu_idx] = max(temps) if temps else 0

    return stats


def print_summary(stats: GPUProfileStats) -> None:
    """Print a formatted summary of GPU profile statistics.

    Args:
        stats: GPUProfileStats object with computed statistics
    """
    print()
    print("=" * 60)
    print("  GPU PROFILE SUMMARY")
    print("=" * 60)
    print()

    # File info
    print(f"  Source file: {stats.source_file}")
    print(f"  Data type:   {'SIMULATED' if stats.is_mock else 'REAL (nvidia-smi)'}")
    print(f"  GPUs found:  {stats.gpu_count}")
    print(f"  Total samples: {stats.total_samples}")
    print()

    # Per-GPU statistics
    for gpu_idx in sorted(stats.samples_per_gpu.keys()):
        print(f"  GPU {gpu_idx}:")
        print(f"    Samples: {stats.samples_per_gpu[gpu_idx]}")
        print()

        # Utilization
        print(f"    GPU Utilization:")
        print(f"      Average: {stats.avg_gpu_util_pct.get(gpu_idx, 0):6.1f} %")
        print(f"      Max:     {stats.max_gpu_util_pct.get(gpu_idx, 0):6.1f} %")
        print(f"      Min:     {stats.min_gpu_util_pct.get(gpu_idx, 0):6.1f} %")
        print()

        # Memory
        print(f"    Memory Utilization:")
        print(f"      Average: {stats.avg_mem_util_pct.get(gpu_idx, 0):6.1f} %")
        if stats.avg_mem_used_mb.get(gpu_idx, 0) > 0:
            print(f"      Used:    {stats.avg_mem_used_mb.get(gpu_idx, 0):6.0f} MB")
        print()

        # Power
        if stats.avg_power_w.get(gpu_idx, 0) > 0:
            print(f"    Power:")
            print(f"      Average: {stats.avg_power_w.get(gpu_idx, 0):6.1f} W")
            print(f"      Max:     {stats.max_power_w.get(gpu_idx, 0):6.1f} W")
            print()

        # Temperature
        if stats.avg_temp_c.get(gpu_idx, 0) > 0:
            print(f"    Temperature:")
            print(f"      Average: {stats.avg_temp_c.get(gpu_idx, 0):6.1f} °C")
            print(f"      Max:     {stats.max_temp_c.get(gpu_idx, 0):6.1f} °C")
            print()

    # Overall summary across all GPUs
    if stats.gpu_count > 1:
        print("-" * 60)
        print("  AGGREGATE (all GPUs):")

        all_gpu_utils = list(stats.avg_gpu_util_pct.values())
        all_mem_utils = list(stats.avg_mem_util_pct.values())
        all_powers = [p for p in stats.avg_power_w.values() if p > 0]

        if all_gpu_utils:
            print(f"    Avg GPU Utilization: {statistics.mean(all_gpu_utils):.1f} %")
        if all_mem_utils:
            print(f"    Avg Memory Util:     {statistics.mean(all_mem_utils):.1f} %")
        if all_powers:
            print(f"    Total Avg Power:     {sum(all_powers):.1f} W")
        print()

    if stats.is_mock:
        print("  [Note: This data is SIMULATED for testing purposes]")

    print("=" * 60)


# =============================================================================
# Timeline and Tracing Utilities
# =============================================================================


@dataclass
class TimelineEvent:
    """Represents a single event in an execution timeline."""

    name: str
    timestamp: float
    duration: Optional[float] = None
    category: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chrome_trace(self, pid: int = 0, tid: int = 0) -> Dict[str, Any]:
        """Convert to Chrome trace format for visualization."""
        event = {
            "name": self.name,
            "cat": self.category,
            "ts": self.timestamp * 1e6,  # Convert to microseconds
            "pid": pid,
            "tid": tid,
        }

        if self.duration is not None:
            event["ph"] = "X"  # Complete event
            event["dur"] = self.duration * 1e6
        else:
            event["ph"] = "i"  # Instant event
            event["s"] = "g"  # Global scope

        if self.metadata:
            event["args"] = self.metadata

        return event


class TraceRecorder:
    """Records timeline events for performance analysis."""

    def __init__(self, name: str = "trace"):
        self.name = name
        self.events: List[TimelineEvent] = []
        self.start_time = time.perf_counter()

    def record(self, event: TimelineEvent) -> None:
        """Record an event."""
        self.events.append(event)

    def instant(self, name: str, category: str = "default", **metadata) -> None:
        """Record an instant event."""
        self.events.append(TimelineEvent(
            name=name,
            timestamp=time.perf_counter() - self.start_time,
            category=category,
            metadata=metadata,
        ))

    def begin(self, name: str, category: str = "default") -> float:
        """Begin a duration event. Returns the start timestamp."""
        return time.perf_counter()

    def end(self, name: str, start_time: float, category: str = "default", **metadata) -> float:
        """End a duration event. Returns the duration."""
        duration = time.perf_counter() - start_time
        self.events.append(TimelineEvent(
            name=name,
            timestamp=start_time - self.start_time,
            duration=duration,
            category=category,
            metadata=metadata,
        ))
        return duration

    def save_chrome_trace(self, output_file: Union[str, Path]) -> None:
        """Save events as a Chrome trace JSON file.

        View with chrome://tracing or https://ui.perfetto.dev/
        """
        trace_events = [e.to_chrome_trace() for e in self.events]
        trace_data = {
            "traceEvents": trace_events,
            "metadata": {"name": self.name},
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(trace_data, f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_overlap_efficiency(
    compute_time: float,
    comm_time: float,
    overlapped_time: float,
) -> Dict[str, float]:
    """Compute efficiency metrics for overlapped execution.

    Args:
        compute_time: Time for compute-only execution
        comm_time: Time for communication-only execution
        overlapped_time: Time for overlapped execution

    Returns:
        Dictionary with efficiency metrics
    """
    sequential_time = compute_time + comm_time
    time_saved = sequential_time - overlapped_time
    overlap_efficiency = time_saved / comm_time if comm_time > 0 else 0
    speedup = sequential_time / overlapped_time if overlapped_time > 0 else 1

    return {
        "compute_time": compute_time,
        "comm_time": comm_time,
        "sequential_time": sequential_time,
        "overlapped_time": overlapped_time,
        "time_saved": time_saved,
        "overlap_efficiency": overlap_efficiency,  # 1.0 = perfect overlap
        "speedup": speedup,
    }


def format_duration(seconds: float) -> str:
    """Format a duration in human-readable form."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# =============================================================================
# Main (for testing)
# =============================================================================


def main():
    """Test the trace utilities."""
    import sys

    if len(sys.argv) > 1:
        # Parse provided log file
        log_file = sys.argv[1]
        try:
            stats = parse_gpu_profile(log_file)
            print_summary(stats)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python trace_utils.py <gpu_profile.log>")
        print()
        print("Example with mock data:")
        print("  1. Run: ./profile_gpu.sh gpu_profile.log 10 1")
        print("  2. Run: python trace_utils.py gpu_profile.log")


if __name__ == "__main__":
    main()
