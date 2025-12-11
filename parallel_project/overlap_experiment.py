#!/usr/bin/env python3
"""
overlap_experiment.py - Compute/Communication Overlap Simulation

This is a CONCEPTUAL EXPERIMENT that simulates the benefits of overlapping
compute (GEMM operations) with communication (NCCL all-gather) using CUDA streams.

=============================================================================
IMPORTANT: CONCEPTUAL EXPERIMENT ONLY
=============================================================================

This script uses Python's time.sleep() to SIMULATE the timing behavior of:
  - GPU compute operations (matrix multiplications)
  - NCCL collective communication (all-gather, all-reduce)
  - CUDA stream-based overlap

In a real implementation, this would require:
  1. Direct CUDA/cuBLAS API calls for compute kernels
  2. NCCL library calls for collective operations
  3. CUDA stream management for concurrent execution
  4. Proper synchronization with CUDA events

The purpose of this simulation is to:
  - Demonstrate the POTENTIAL speedup from overlapping
  - Provide approximate timing estimates for analysis
  - Illustrate the concept for the 15-418 project report

For real overlap implementation, see:
  - runtime_patch/stream_overlap.py (FakeStreamOverlapEngine)
  - MLC-LLM's actual CUDA stream management in cpp/serve/

=============================================================================

15-418 Parallel Computer Architecture Project
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Simulation Parameters
# =============================================================================

# Realistic timing estimates based on typical LLM inference on A100 GPUs
# These are approximations for Llama-3-8B with batch_size=1

DEFAULT_PARAMS = {
    # Compute times (in milliseconds)
    "gemm_time_ms": 2.5,        # Single GEMM operation
    "attention_time_ms": 1.8,   # Attention computation
    "ffn_time_ms": 3.2,         # Feed-forward network

    # Communication times (in milliseconds)
    "all_gather_time_ms": 0.8,  # NCCL all-gather for TP
    "all_reduce_time_ms": 1.2,  # NCCL all-reduce
    "p2p_send_time_ms": 0.3,    # Point-to-point send

    # Number of layers
    "num_layers": 32,

    # Number of GPUs
    "num_gpus": 2,
}


# =============================================================================
# Simulated Stream Operations
# =============================================================================


@dataclass
class StreamOperation:
    """Represents an operation on a CUDA stream."""
    name: str
    stream: str  # "compute" or "comm"
    duration_ms: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class SimulatedStream:
    """
    Simulates a CUDA stream for conceptual demonstration.

    In real CUDA:
    - Streams allow concurrent kernel execution
    - Operations within a stream execute in order
    - Operations across streams can overlap

    This simulation uses Python threads to approximate this behavior.
    """

    def __init__(self, name: str):
        self.name = name
        self.operations: List[StreamOperation] = []
        self._lock = threading.Lock()

    def record_operation(self, op: StreamOperation) -> None:
        """Record an operation that was executed on this stream."""
        with self._lock:
            self.operations.append(op)

    def get_total_time(self) -> float:
        """Get total time spent on this stream."""
        return sum(op.duration_ms for op in self.operations)


def simulate_compute(duration_ms: float, name: str = "compute") -> StreamOperation:
    """
    Simulate a GPU compute operation (e.g., GEMM).

    In reality, this would be a cuBLAS or custom CUDA kernel call:
        cublasSgemm(handle, ...)  // Matrix multiplication

    We use time.sleep() to simulate the execution time.
    """
    op = StreamOperation(name=name, stream="compute", duration_ms=duration_ms)
    op.start_time = time.perf_counter()

    # Simulate compute time
    time.sleep(duration_ms / 1000.0)

    op.end_time = time.perf_counter()
    return op


def simulate_communication(duration_ms: float, name: str = "comm") -> StreamOperation:
    """
    Simulate an NCCL collective communication operation.

    In reality, this would be an NCCL call:
        ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream)

    We use time.sleep() to simulate the communication time.
    """
    op = StreamOperation(name=name, stream="comm", duration_ms=duration_ms)
    op.start_time = time.perf_counter()

    # Simulate communication time
    time.sleep(duration_ms / 1000.0)

    op.end_time = time.perf_counter()
    return op


# =============================================================================
# Execution Modes
# =============================================================================


def run_sequential(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run compute and communication SEQUENTIALLY (no overlap).

    This represents the naive approach where:
    1. All compute operations complete
    2. Then all communication operations complete

    Timeline:
        |----COMPUTE----|----COMM----|
    """
    print("\n[Sequential Mode] Running compute, then communication...")

    total_start = time.perf_counter()

    # Simulate compute phase
    compute_ops = []
    for layer in range(params["num_layers"]):
        # Each layer has: attention + FFN
        op1 = simulate_compute(params["attention_time_ms"], f"layer{layer}_attn")
        op2 = simulate_compute(params["ffn_time_ms"], f"layer{layer}_ffn")
        compute_ops.extend([op1, op2])

    compute_end = time.perf_counter()
    compute_time_ms = (compute_end - total_start) * 1000

    # Simulate communication phase
    comm_ops = []
    for layer in range(params["num_layers"]):
        # Each layer needs: all-gather for attention, all-reduce for output
        op1 = simulate_communication(params["all_gather_time_ms"], f"layer{layer}_gather")
        op2 = simulate_communication(params["all_reduce_time_ms"], f"layer{layer}_reduce")
        comm_ops.extend([op1, op2])

    total_end = time.perf_counter()
    total_time_ms = (total_end - total_start) * 1000
    comm_time_ms = total_time_ms - compute_time_ms

    return {
        "mode": "sequential",
        "compute_time_ms": compute_time_ms,
        "comm_time_ms": comm_time_ms,
        "total_time_ms": total_time_ms,
        "num_compute_ops": len(compute_ops),
        "num_comm_ops": len(comm_ops),
    }


def run_overlapped(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run compute and communication with OVERLAP using simulated streams.

    This represents the optimized approach where:
    - Compute for layer N runs on compute_stream
    - Communication for layer N-1 runs on comm_stream concurrently

    Timeline (overlapped):
        compute_stream: |--L0_compute--|--L1_compute--|--L2_compute--|...
        comm_stream:             |--L0_comm--|--L1_comm--|--L2_comm--|...

    The communication for layer N overlaps with compute for layer N+1.
    """
    print("\n[Overlapped Mode] Running with stream-based overlap...")

    compute_stream = SimulatedStream("compute")
    comm_stream = SimulatedStream("comm")

    total_start = time.perf_counter()

    # Use thread pool to simulate concurrent streams
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_comm = None

        for layer in range(params["num_layers"]):
            # Start compute for current layer
            compute_duration = params["attention_time_ms"] + params["ffn_time_ms"]
            compute_future = executor.submit(
                simulate_compute, compute_duration, f"layer{layer}_compute"
            )

            # If there's pending communication from previous layer, it runs in parallel
            if pending_comm is not None:
                # Wait for previous communication to complete
                comm_op = pending_comm.result()
                comm_stream.record_operation(comm_op)

            # Get compute result
            compute_op = compute_future.result()
            compute_stream.record_operation(compute_op)

            # Schedule communication for this layer (will overlap with next layer's compute)
            comm_duration = params["all_gather_time_ms"] + params["all_reduce_time_ms"]
            pending_comm = executor.submit(
                simulate_communication, comm_duration, f"layer{layer}_comm"
            )

        # Wait for final communication
        if pending_comm is not None:
            comm_op = pending_comm.result()
            comm_stream.record_operation(comm_op)

    total_end = time.perf_counter()
    total_time_ms = (total_end - total_start) * 1000

    compute_time_ms = compute_stream.get_total_time()
    comm_time_ms = comm_stream.get_total_time()

    return {
        "mode": "overlapped",
        "compute_time_ms": compute_time_ms,
        "comm_time_ms": comm_time_ms,
        "total_time_ms": total_time_ms,
        "num_compute_ops": len(compute_stream.operations),
        "num_comm_ops": len(comm_stream.operations),
    }


def run_fully_overlapped(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run with MAXIMUM overlap (theoretical best case).

    This represents ideal overlap where ALL communication is hidden:
    - Communication time is completely overlapped with compute
    - Total time ≈ max(compute_time, comm_time)

    This is the theoretical upper bound and may not be achievable
    in practice due to:
    - Data dependencies
    - Memory bandwidth contention
    - Synchronization requirements
    """
    print("\n[Fully Overlapped Mode] Running with maximum theoretical overlap...")

    total_start = time.perf_counter()

    # Calculate ideal times
    compute_per_layer = params["attention_time_ms"] + params["ffn_time_ms"]
    comm_per_layer = params["all_gather_time_ms"] + params["all_reduce_time_ms"]

    total_compute = compute_per_layer * params["num_layers"]
    total_comm = comm_per_layer * params["num_layers"]

    # In ideal overlap, total time is max of compute and comm
    # Plus one layer of non-overlapped comm at the end
    ideal_time = max(total_compute, total_comm) + comm_per_layer

    # Simulate the ideal time
    time.sleep(ideal_time / 1000.0)

    total_end = time.perf_counter()
    total_time_ms = (total_end - total_start) * 1000

    return {
        "mode": "fully_overlapped",
        "compute_time_ms": total_compute,
        "comm_time_ms": total_comm,
        "total_time_ms": total_time_ms,
        "theoretical_min_ms": ideal_time,
        "note": "Theoretical best case - may not be achievable in practice",
    }


# =============================================================================
# Analysis and Reporting
# =============================================================================


def analyze_results(
    sequential: Dict[str, Any],
    overlapped: Dict[str, Any],
    fully_overlapped: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze and compare the results from different execution modes."""

    seq_time = sequential["total_time_ms"]
    ovl_time = overlapped["total_time_ms"]
    full_time = fully_overlapped["total_time_ms"]

    return {
        # Time saved
        "time_saved_overlapped_ms": seq_time - ovl_time,
        "time_saved_full_ms": seq_time - full_time,

        # Speedup factors
        "speedup_overlapped": seq_time / ovl_time if ovl_time > 0 else 1.0,
        "speedup_full": seq_time / full_time if full_time > 0 else 1.0,

        # Overlap efficiency
        # How much of the communication was hidden?
        "overlap_efficiency_pct": (
            (seq_time - ovl_time) / sequential["comm_time_ms"] * 100
            if sequential["comm_time_ms"] > 0 else 0
        ),

        # Theoretical maximum overlap
        "theoretical_max_efficiency_pct": (
            (seq_time - full_time) / sequential["comm_time_ms"] * 100
            if sequential["comm_time_ms"] > 0 else 0
        ),
    }


def print_results(
    params: Dict[str, Any],
    sequential: Dict[str, Any],
    overlapped: Dict[str, Any],
    fully_overlapped: Dict[str, Any],
    analysis: Dict[str, Any],
) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("  COMPUTE/COMMUNICATION OVERLAP EXPERIMENT RESULTS")
    print("  (Simulated - Conceptual Demonstration for 15-418 Project)")
    print("=" * 70)

    # Configuration
    print("\n[Configuration]")
    print(f"  Number of layers:     {params['num_layers']}")
    print(f"  Number of GPUs:       {params['num_gpus']}")
    print(f"  Attention time:       {params['attention_time_ms']:.1f} ms/layer")
    print(f"  FFN time:             {params['ffn_time_ms']:.1f} ms/layer")
    print(f"  All-gather time:      {params['all_gather_time_ms']:.1f} ms/layer")
    print(f"  All-reduce time:      {params['all_reduce_time_ms']:.1f} ms/layer")

    # Sequential results
    print("\n[Sequential Execution] (Naive - No Overlap)")
    print(f"  Compute time:         {sequential['compute_time_ms']:.1f} ms")
    print(f"  Communication time:   {sequential['comm_time_ms']:.1f} ms")
    print(f"  Total time:           {sequential['total_time_ms']:.1f} ms")

    # Overlapped results
    print("\n[Overlapped Execution] (Stream-based Overlap)")
    print(f"  Compute time:         {overlapped['compute_time_ms']:.1f} ms")
    print(f"  Communication time:   {overlapped['comm_time_ms']:.1f} ms")
    print(f"  Total time:           {overlapped['total_time_ms']:.1f} ms")
    print(f"  Time saved:           {analysis['time_saved_overlapped_ms']:.1f} ms")
    print(f"  Speedup:              {analysis['speedup_overlapped']:.2f}x")

    # Fully overlapped results
    print("\n[Fully Overlapped] (Theoretical Maximum)")
    print(f"  Total time:           {fully_overlapped['total_time_ms']:.1f} ms")
    print(f"  Time saved:           {analysis['time_saved_full_ms']:.1f} ms")
    print(f"  Speedup:              {analysis['speedup_full']:.2f}x")

    # Summary
    print("\n" + "-" * 70)
    print("[SUMMARY]")
    print(f"  Overlap efficiency achieved:    {analysis['overlap_efficiency_pct']:.1f}%")
    print(f"  Theoretical max efficiency:     {analysis['theoretical_max_efficiency_pct']:.1f}%")
    print(f"  Communication hidden:           {analysis['time_saved_overlapped_ms']:.1f} ms")
    print()
    print("  Interpretation:")
    print(f"    - Overlapping saves {analysis['time_saved_overlapped_ms']:.1f} ms per forward pass")
    print(f"    - This is {analysis['overlap_efficiency_pct']:.0f}% of the communication overhead")
    print(f"    - Achieves {analysis['speedup_overlapped']:.2f}x speedup vs sequential")

    print("\n" + "=" * 70)
    print("  NOTE: These are SIMULATED values for conceptual demonstration.")
    print("  Real measurements require actual CUDA/NCCL implementation.")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate compute/communication overlap in LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a CONCEPTUAL experiment that simulates overlap benefits.
See the module docstring for details on the simulation approach.

Examples:
  python overlap_experiment.py
  python overlap_experiment.py --num-layers 64 --num-gpus 4
  python overlap_experiment.py --output results/overlap.json
        """,
    )
    parser.add_argument(
        "--num-layers", type=int, default=32,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2,
        help="Number of GPUs (affects communication time)",
    )
    parser.add_argument(
        "--gemm-time", type=float, default=2.5,
        help="GEMM time per operation (ms)",
    )
    parser.add_argument(
        "--attention-time", type=float, default=1.8,
        help="Attention compute time (ms)",
    )
    parser.add_argument(
        "--ffn-time", type=float, default=3.2,
        help="FFN compute time (ms)",
    )
    parser.add_argument(
        "--all-gather-time", type=float, default=0.8,
        help="All-gather communication time (ms)",
    )
    parser.add_argument(
        "--all-reduce-time", type=float, default=1.2,
        help="All-reduce communication time (ms)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build parameters
    params = {
        "num_layers": args.num_layers,
        "num_gpus": args.num_gpus,
        "gemm_time_ms": args.gemm_time,
        "attention_time_ms": args.attention_time,
        "ffn_time_ms": args.ffn_time,
        "all_gather_time_ms": args.all_gather_time,
        "all_reduce_time_ms": args.all_reduce_time,
    }

    print("=" * 70)
    print("  COMPUTE/COMMUNICATION OVERLAP EXPERIMENT")
    print("  15-418 Parallel Computer Architecture Project")
    print("=" * 70)
    print()
    print("  This experiment SIMULATES the benefits of overlapping GPU compute")
    print("  operations with NCCL communication using CUDA streams.")
    print()
    print("  See overlap_experiment.py docstring for conceptual details.")

    # Run experiments
    sequential_results = run_sequential(params)
    overlapped_results = run_overlapped(params)
    fully_overlapped_results = run_fully_overlapped(params)

    # Analyze
    analysis = analyze_results(
        sequential_results,
        overlapped_results,
        fully_overlapped_results,
    )

    # Print results
    print_results(
        params,
        sequential_results,
        overlapped_results,
        fully_overlapped_results,
        analysis,
    )

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "params": params,
            "sequential": sequential_results,
            "overlapped": overlapped_results,
            "fully_overlapped": fully_overlapped_results,
            "analysis": analysis,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Results saved to: {args.output}]")


if __name__ == "__main__":
    main()
