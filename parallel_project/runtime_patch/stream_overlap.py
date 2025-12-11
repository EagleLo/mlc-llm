#!/usr/bin/env python3
"""
stream_overlap.py - Simulated CUDA Stream Overlap Engine

This module provides a FakeStreamOverlapEngine class that simulates
the behavior of overlapping compute and communication operations
using CUDA streams.

=============================================================================
EXPERIMENTAL MODULE - For 15-418 Project Report
=============================================================================

This module is part of the experimental analysis for the 15-418 Parallel
Computer Architecture project. It demonstrates the CONCEPT of stream-based
overlap without requiring actual CUDA hardware.

In a real implementation:
  - CUDA streams would be created via cudaStreamCreate()
  - Compute kernels would be launched on compute streams
  - NCCL operations would use dedicated communication streams
  - Synchronization would use CUDA events (cudaEventRecord/cudaStreamWaitEvent)

This simulation helps us:
  1. Understand the programming model for overlap
  2. Estimate potential performance benefits
  3. Identify implementation challenges
  4. Prepare for real CUDA implementation

For the 15-418 report, this module provides:
  - Code examples of the overlap pattern
  - Simulated timing measurements
  - Analysis of overlap efficiency

=============================================================================

15-418 Parallel Computer Architecture Project
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StreamEvent:
    """
    Simulates a CUDA event for stream synchronization.

    In real CUDA:
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event, stream);
        cudaStreamWaitEvent(other_stream, event);
    """
    name: str
    timestamp: float = 0.0
    recorded: bool = False

    def record(self) -> None:
        """Record the event (mark current time)."""
        self.timestamp = time.perf_counter()
        self.recorded = True

    def elapsed_time(self, other: "StreamEvent") -> float:
        """Get elapsed time between two events in milliseconds."""
        if not (self.recorded and other.recorded):
            return 0.0
        return abs(self.timestamp - other.timestamp) * 1000


@dataclass
class OverlapMetrics:
    """Metrics from overlapped execution."""
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    total_time_ms: float = 0.0
    overlap_time_ms: float = 0.0
    efficiency_pct: float = 0.0


# =============================================================================
# FakeStreamOverlapEngine
# =============================================================================


class FakeStreamOverlapEngine:
    """
    Simulated CUDA stream overlap engine for experimental analysis.

    This class demonstrates the pattern of overlapping compute operations
    with communication operations using separate "streams" (simulated with
    Python threads).

    =========================================================================
    LINK TO 15-418 PROJECT REPORT
    =========================================================================

    This module supports Section 4 of the 15-418 project report:
    "Compute/Communication Overlap Analysis"

    Key concepts demonstrated:
    1. Stream-based concurrency model
    2. Async collective communication (NCCL pattern)
    3. Overlap efficiency measurement
    4. Synchronization patterns

    The experimental results from this module are used to:
    - Validate overlap potential in tensor parallel inference
    - Estimate performance improvements from overlap
    - Identify bottlenecks and optimization opportunities

    =========================================================================

    Usage:
        engine = FakeStreamOverlapEngine(num_streams=2)

        # Launch overlapped execution
        metrics = engine.launch_with_overlap(
            compute_fn=lambda: time.sleep(0.01),  # Simulated compute
            comm_fn=lambda: time.sleep(0.005),    # Simulated communication
        )

        print(f"Overlap efficiency: {metrics.efficiency_pct:.1f}%")

    Attributes:
        num_streams: Number of simulated CUDA streams
        metrics_history: History of execution metrics
    """

    def __init__(self, num_streams: int = 2):
        """
        Initialize the fake stream overlap engine.

        Args:
            num_streams: Number of simulated streams (default: 2 for compute + comm)
        """
        self.num_streams = num_streams
        self.metrics_history: List[OverlapMetrics] = []
        self._executor = ThreadPoolExecutor(max_workers=num_streams)
        self._lock = threading.Lock()

        # Simulated stream state
        self._compute_stream_busy = False
        self._comm_stream_busy = False

        print(f"[FakeStreamOverlapEngine] Initialized with {num_streams} simulated streams")
        print("[FakeStreamOverlapEngine] Note: This is a simulation for 15-418 analysis")

    def async_all_gather(
        self,
        data_size_mb: float = 1.0,
        bandwidth_gbps: float = 300.0,
    ) -> Future:
        """
        Simulated async NCCL all-gather operation.

        In real NCCL:
            ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);

        This would:
        1. Launch the all-gather on a communication stream
        2. Return immediately (non-blocking)
        3. The operation completes asynchronously

        Args:
            data_size_mb: Size of data to gather in MB
            bandwidth_gbps: Simulated interconnect bandwidth

        Returns:
            Future that completes when the operation is done
        """
        print(f"[async_all_gather] Simulated async NCCL all-gather ({data_size_mb:.1f} MB)")

        # Calculate simulated transfer time
        bandwidth_mbps = bandwidth_gbps * 1000 / 8
        transfer_time_s = data_size_mb / bandwidth_mbps

        def _do_gather():
            self._comm_stream_busy = True
            time.sleep(transfer_time_s)
            self._comm_stream_busy = False
            return {"data_size_mb": data_size_mb, "time_ms": transfer_time_s * 1000}

        return self._executor.submit(_do_gather)

    def async_all_reduce(
        self,
        data_size_mb: float = 1.0,
        bandwidth_gbps: float = 300.0,
        num_gpus: int = 2,
    ) -> Future:
        """
        Simulated async NCCL all-reduce operation.

        In real NCCL:
            ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);

        Args:
            data_size_mb: Size of data in MB
            bandwidth_gbps: Simulated bandwidth
            num_gpus: Number of GPUs participating

        Returns:
            Future for the async operation
        """
        print(f"[async_all_reduce] Simulated async NCCL all-reduce ({data_size_mb:.1f} MB)")

        # Ring all-reduce: 2*(n-1)/n * data_size
        effective_size = 2 * (num_gpus - 1) / num_gpus * data_size_mb
        bandwidth_mbps = bandwidth_gbps * 1000 / 8
        transfer_time_s = effective_size / bandwidth_mbps

        def _do_reduce():
            self._comm_stream_busy = True
            time.sleep(transfer_time_s)
            self._comm_stream_busy = False
            return {"data_size_mb": data_size_mb, "time_ms": transfer_time_s * 1000}

        return self._executor.submit(_do_reduce)

    def launch_with_overlap(
        self,
        compute_fn: Callable[[], Any],
        comm_fn: Callable[[], Any],
    ) -> OverlapMetrics:
        """
        Execute compute and communication functions with simulated overlap.

        This demonstrates the key pattern for hiding communication latency:
        1. Launch communication on comm stream (async)
        2. Launch compute on compute stream (async)
        3. Both execute concurrently
        4. Wait for both to complete

        Timeline visualization:
            compute_stream: |---COMPUTE---|
            comm_stream:    |--COMM--|
                           ^---------^
                           Total time = max(compute, comm) + overhead

        Args:
            compute_fn: Function representing compute work
            comm_fn: Function representing communication work

        Returns:
            OverlapMetrics with timing information
        """
        start_event = StreamEvent("start")
        compute_end_event = StreamEvent("compute_end")
        comm_end_event = StreamEvent("comm_end")
        end_event = StreamEvent("end")

        # Record start
        start_event.record()

        # Launch both operations concurrently
        compute_future = self._executor.submit(self._timed_execute, compute_fn)
        comm_future = self._executor.submit(self._timed_execute, comm_fn)

        # Wait for both
        compute_time = compute_future.result()
        compute_end_event.record()

        comm_time = comm_future.result()
        comm_end_event.record()

        # Record end
        end_event.record()

        # Calculate metrics
        total_time = start_event.elapsed_time(end_event)
        sequential_time = compute_time + comm_time

        # Overlap is how much time we saved vs sequential
        overlap_time = sequential_time - total_time
        efficiency = (overlap_time / comm_time * 100) if comm_time > 0 else 0

        metrics = OverlapMetrics(
            compute_time_ms=compute_time,
            comm_time_ms=comm_time,
            total_time_ms=total_time,
            overlap_time_ms=overlap_time,
            efficiency_pct=efficiency,
        )

        with self._lock:
            self.metrics_history.append(metrics)

        return metrics

    def _timed_execute(self, fn: Callable[[], Any]) -> float:
        """Execute a function and return its duration in milliseconds."""
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000

    def synchronize(self) -> None:
        """
        Synchronize all streams (wait for all operations to complete).

        In real CUDA:
            cudaDeviceSynchronize();
        """
        # Wait for executor to be idle
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=self.num_streams)
        print("[synchronize] All simulated streams synchronized")

    def get_average_metrics(self) -> Optional[OverlapMetrics]:
        """Get average metrics from all recorded executions."""
        if not self.metrics_history:
            return None

        n = len(self.metrics_history)
        return OverlapMetrics(
            compute_time_ms=sum(m.compute_time_ms for m in self.metrics_history) / n,
            comm_time_ms=sum(m.comm_time_ms for m in self.metrics_history) / n,
            total_time_ms=sum(m.total_time_ms for m in self.metrics_history) / n,
            overlap_time_ms=sum(m.overlap_time_ms for m in self.metrics_history) / n,
            efficiency_pct=sum(m.efficiency_pct for m in self.metrics_history) / n,
        )

    def print_summary(self) -> None:
        """Print a summary of recorded metrics."""
        avg = self.get_average_metrics()
        if avg is None:
            print("[FakeStreamOverlapEngine] No metrics recorded")
            return

        print("\n" + "=" * 60)
        print("  FakeStreamOverlapEngine Metrics Summary")
        print("  (Simulated for 15-418 Project Analysis)")
        print("=" * 60)
        print(f"  Executions recorded: {len(self.metrics_history)}")
        print(f"\n  Average Compute Time:    {avg.compute_time_ms:.2f} ms")
        print(f"  Average Comm Time:       {avg.comm_time_ms:.2f} ms")
        print(f"  Average Total Time:      {avg.total_time_ms:.2f} ms")
        print(f"  Average Overlap Time:    {avg.overlap_time_ms:.2f} ms")
        print(f"  Average Overlap Efficiency: {avg.efficiency_pct:.1f}%")
        print("=" * 60)

    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        self._executor.shutdown(wait=True)
        print("[FakeStreamOverlapEngine] Shutdown complete")


# =============================================================================
# Convenience Functions
# =============================================================================


def demo_overlap_experiment():
    """
    Demonstrate the overlap engine with a simple experiment.

    This function can be used to quickly test the overlap simulation
    and generate sample metrics for the 15-418 report.
    """
    print("=" * 60)
    print("  FakeStreamOverlapEngine Demo")
    print("  15-418 Parallel Computing Project")
    print("=" * 60)

    engine = FakeStreamOverlapEngine(num_streams=2)

    # Simulate different compute/comm ratios
    test_cases = [
        ("Compute-heavy", 10.0, 3.0),   # 10ms compute, 3ms comm
        ("Comm-heavy", 3.0, 10.0),       # 3ms compute, 10ms comm
        ("Balanced", 5.0, 5.0),          # 5ms each
    ]

    for name, compute_ms, comm_ms in test_cases:
        print(f"\n[Test: {name}]")
        print(f"  Compute: {compute_ms} ms, Comm: {comm_ms} ms")

        metrics = engine.launch_with_overlap(
            compute_fn=lambda c=compute_ms: time.sleep(c / 1000),
            comm_fn=lambda c=comm_ms: time.sleep(c / 1000),
        )

        print(f"  Total time: {metrics.total_time_ms:.2f} ms")
        print(f"  Time saved: {metrics.overlap_time_ms:.2f} ms")
        print(f"  Efficiency: {metrics.efficiency_pct:.1f}%")

    engine.print_summary()
    engine.shutdown()


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    demo_overlap_experiment()
