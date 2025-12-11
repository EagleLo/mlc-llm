#!/usr/bin/env python3
"""
run_tensor_parallel.py - Tensor parallel LLM inference experiments.

This script runs MLC-LLM inference with tensor parallelism (tensor_parallel_shards=2)
to measure scaling efficiency, NCCL communication overhead, and GPU utilization.

Requirements:
- Same structure as run_baseline.py, but set tensor_parallel_shards=2
- Measure and print: Tokens/sec, Estimated NCCL overhead, GPU utilization
- If NCCL or multiple GPUs not available, simulate realistic measurements

15-418 Parallel Computer Architecture Project
"""

import argparse
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics
import threading

# Check if we can import MLC-LLM
MLC_AVAILABLE = False
try:
    from mlc_llm import MLCEngine
    from mlc_llm.serve import EngineConfig

    MLC_AVAILABLE = True
except ImportError:
    MLCEngine = None
    EngineConfig = None


# =============================================================================
# GPU and NCCL Utilities
# =============================================================================


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def check_nccl_available() -> bool:
    """Check if NCCL is available for multi-GPU communication."""
    # Check for NCCL library
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "libnccl" in result.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check environment variable
    if "NCCL_DEBUG" in os.environ or "NCCL_HOME" in os.environ:
        return True

    return False


def get_gpu_utilization() -> List[Dict[str, float]]:
    """Get GPU utilization for all GPUs.

    Returns list of dicts with utilization metrics per GPU.
    """
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "gpu_util_pct": float(parts[2]),
                        "mem_util_pct": float(parts[3]),
                        "memory_used_mb": float(parts[4]),
                        "memory_total_mb": float(parts[5]),
                        "power_w": float(parts[6]) if parts[6] != "[N/A]" else 0,
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return gpus


def estimate_nccl_overhead(
    data_size_mb: float,
    num_gpus: int,
    bandwidth_gbps: float = 300.0,  # NVLink bandwidth
) -> float:
    """Estimate NCCL all-reduce overhead in milliseconds.

    Args:
        data_size_mb: Size of data to communicate in MB
        num_gpus: Number of GPUs involved
        bandwidth_gbps: Interconnect bandwidth in Gbps

    Returns:
        Estimated overhead in milliseconds
    """
    # All-reduce communication volume: 2 * (n-1) / n * data_size
    # For ring all-reduce algorithm
    comm_volume_mb = 2 * (num_gpus - 1) / num_gpus * data_size_mb

    # Convert bandwidth to MB/s
    bandwidth_mbps = bandwidth_gbps * 1000 / 8  # Gbps to MB/s

    # Time = volume / bandwidth
    time_ms = (comm_volume_mb / bandwidth_mbps) * 1000

    # Add latency overhead (typical NCCL latency ~5-10 microseconds per operation)
    latency_ms = 0.01 * num_gpus

    return time_ms + latency_ms


# =============================================================================
# Mock Classes for Testing Without Multi-GPU
# =============================================================================


@dataclass
class MockUsage:
    """Mock usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MockMessage:
    """Mock message."""

    content: str = ""
    role: str = "assistant"


@dataclass
class MockChoice:
    """Mock choice."""

    index: int = 0
    message: Optional[MockMessage] = None
    finish_reason: Optional[str] = None


@dataclass
class MockCompletionResponse:
    """Mock completion response."""

    id: str = "mock-tp-response"
    choices: List[MockChoice] = field(default_factory=list)
    usage: MockUsage = field(default_factory=MockUsage)


class MockTPChatCompletions:
    """Mock chat completions for tensor parallel simulation."""

    def __init__(
        self,
        tensor_parallel_shards: int = 2,
        base_tokens_per_second: float = 30.0,
    ):
        self.tensor_parallel_shards = tensor_parallel_shards
        self.base_tokens_per_second = base_tokens_per_second

        # Simulate speedup with diminishing returns due to communication
        # Typical efficiency: ~85% for 2 GPUs, ~70% for 4 GPUs, ~55% for 8 GPUs
        self.efficiency = {
            1: 1.0,
            2: 0.85,
            4: 0.70,
            8: 0.55,
        }.get(tensor_parallel_shards, 0.5)

        self.effective_tps = base_tokens_per_second * tensor_parallel_shards * self.efficiency

        self._sample_responses = [
            "Tensor parallelism distributes model layers across multiple GPUs, "
            "enabling inference of models too large for a single GPU. Each GPU "
            "processes a portion of the tensor operations, with NCCL handling "
            "the all-reduce communication between devices.",
            "In tensor parallel inference, matrix multiplications are split across "
            "GPUs. This reduces memory per device but introduces communication "
            "overhead for synchronizing intermediate results via collective operations.",
        ]

    def create(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        stream: bool = False,
        **kwargs,
    ):
        """Create a mock tensor parallel completion."""
        response_text = random.choice(self._sample_responses)
        words = response_text.split()
        approx_tokens = min(len(words), max_tokens)
        response_text = " ".join(words[:approx_tokens])

        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = len(prompt_text.split())

        # Simulate generation time with TP speedup
        generation_time = approx_tokens / self.effective_tps

        # Add simulated NCCL overhead (per-token)
        # Assume ~0.5MB per token for hidden states, 2 GPUs
        nccl_overhead_per_token = estimate_nccl_overhead(
            0.5, self.tensor_parallel_shards
        ) / 1000  # Convert to seconds
        generation_time += nccl_overhead_per_token * approx_tokens * 0.1  # Partial overlap

        time.sleep(generation_time)

        usage = MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=approx_tokens,
            total_tokens=prompt_tokens + approx_tokens,
        )

        return MockCompletionResponse(
            choices=[MockChoice(message=MockMessage(content=response_text))],
            usage=usage,
        )


class MockTPChat:
    """Mock chat interface for tensor parallel."""

    def __init__(self, tensor_parallel_shards: int = 2):
        self.completions = MockTPChatCompletions(tensor_parallel_shards)


class MockTPMLCEngine:
    """Mock MLC Engine for tensor parallel testing."""

    def __init__(self, model: str, tensor_parallel_shards: int = 2, **kwargs):
        self.model = model
        self.tensor_parallel_shards = tensor_parallel_shards
        self.chat = MockTPChat(tensor_parallel_shards)
        self._is_mock = True
        print(f"[MOCK TP MODE] Simulating model: {model}")
        print(f"[MOCK TP MODE] Tensor parallel shards: {tensor_parallel_shards}")

    def terminate(self):
        """Terminate the mock engine."""
        pass


# =============================================================================
# NCCL Overhead Tracker
# =============================================================================


class NCCLOverheadTracker:
    """Tracks and estimates NCCL communication overhead."""

    def __init__(self, num_gpus: int = 2):
        self.num_gpus = num_gpus
        self.samples: List[float] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None

        # Preset realistic overhead values (in ms) for different configurations
        # Based on typical all-reduce latencies for transformer hidden states
        self._preset_overheads = {
            2: {"mean": 0.8, "std": 0.2},   # 2 GPUs: ~0.8ms per sync
            4: {"mean": 1.5, "std": 0.4},   # 4 GPUs: ~1.5ms per sync
            8: {"mean": 3.0, "std": 0.8},   # 8 GPUs: ~3.0ms per sync
        }

    def get_estimated_overhead_ms(self) -> Dict[str, float]:
        """Get estimated NCCL overhead in milliseconds."""
        if self.samples:
            return {
                "mean": statistics.mean(self.samples),
                "std": statistics.stdev(self.samples) if len(self.samples) > 1 else 0,
                "min": min(self.samples),
                "max": max(self.samples),
            }

        # Return preset values if no measurements
        preset = self._preset_overheads.get(self.num_gpus, {"mean": 2.0, "std": 0.5})
        return {
            "mean": preset["mean"],
            "std": preset["std"],
            "min": preset["mean"] - preset["std"],
            "max": preset["mean"] + preset["std"],
            "is_estimated": True,
        }

    def add_sample(self, overhead_ms: float):
        """Add a measured overhead sample."""
        self.samples.append(overhead_ms)

    def simulate_overhead(self) -> float:
        """Simulate a realistic NCCL overhead value."""
        preset = self._preset_overheads.get(self.num_gpus, {"mean": 2.0, "std": 0.5})
        return max(0.1, random.gauss(preset["mean"], preset["std"]))


# =============================================================================
# Tensor Parallel Runner
# =============================================================================

import os


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tensor parallel MLC-LLM inference experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tensor_parallel.py
  python run_tensor_parallel.py --tp-shards 4
  python run_tensor_parallel.py --mock --num-runs 5
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
        help="Model path or HuggingFace identifier",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain tensor parallelism in deep learning inference.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--tp-shards",
        type=int,
        default=2,
        help="Number of tensor parallel shards (GPUs)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of inference runs",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    return parser.parse_args()


def create_tp_engine(
    model: str,
    tensor_parallel_shards: int = 2,
    device: str = "auto",
    force_mock: bool = False,
) -> Tuple[Any, bool]:
    """Create a tensor parallel MLC engine or fall back to mock mode.

    Args:
        model: Model path
        tensor_parallel_shards: Number of TP shards
        device: Device to use
        force_mock: Force mock mode

    Returns:
        Tuple of (engine, is_mock)
    """
    if force_mock:
        print("\n[INFO] Mock mode forced by --mock flag")
        return MockTPMLCEngine(model, tensor_parallel_shards), True

    # Check GPU availability
    gpu_count = get_gpu_count()
    nccl_available = check_nccl_available()

    if gpu_count < tensor_parallel_shards:
        print(f"\n[INFO] Requested {tensor_parallel_shards} GPUs but only {gpu_count} available.")
        print("[INFO] Running in mock mode with simulated multi-GPU behavior.")
        return MockTPMLCEngine(model, tensor_parallel_shards), True

    if not nccl_available and tensor_parallel_shards > 1:
        print("\n[INFO] NCCL not detected. Running in mock mode.")
        return MockTPMLCEngine(model, tensor_parallel_shards), True

    if not MLC_AVAILABLE:
        print("\n[INFO] mlc_llm not installed. Running in mock mode.")
        return MockTPMLCEngine(model, tensor_parallel_shards), True

    # Try to create real engine with tensor parallelism
    try:
        print(f"\n[INFO] Creating MLCEngine with TP={tensor_parallel_shards}...")
        print(f"[INFO] Model: {model}")

        engine_config = EngineConfig(
            tensor_parallel_shards=tensor_parallel_shards,
            gpu_memory_utilization=0.85,
        )

        engine = MLCEngine(
            model=model,
            device=device,
            mode="local",
            engine_config=engine_config,
        )

        print(f"[INFO] Tensor parallel engine created with {tensor_parallel_shards} shards!")
        return engine, False

    except Exception as e:
        print(f"\n[WARNING] Failed to create TP engine: {e}")
        print("[INFO] Falling back to mock mode.")
        return MockTPMLCEngine(model, tensor_parallel_shards), True


def run_tp_inference(
    engine: Any,
    prompt: str,
    max_tokens: int,
    is_mock: bool,
    nccl_tracker: NCCLOverheadTracker,
) -> Dict[str, Any]:
    """Run a single tensor parallel inference.

    Args:
        engine: MLC engine or mock
        prompt: Input prompt
        max_tokens: Max tokens
        is_mock: Whether in mock mode
        nccl_tracker: NCCL overhead tracker

    Returns:
        Metrics dictionary
    """
    start_time = time.perf_counter()

    response = engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False,
    )

    end_time = time.perf_counter()
    latency = end_time - start_time

    metrics = {
        "latency_s": latency,
        "is_mock": is_mock,
    }

    if response.choices:
        metrics["output_tokens"] = response.usage.completion_tokens
        metrics["prompt_tokens"] = response.usage.prompt_tokens
        metrics["total_tokens"] = response.usage.total_tokens
        metrics["tokens_per_second"] = response.usage.completion_tokens / latency

        # Estimate or simulate NCCL overhead
        if is_mock:
            nccl_overhead = nccl_tracker.simulate_overhead()
        else:
            # Estimate based on token count and data size
            nccl_overhead = estimate_nccl_overhead(
                0.5 * response.usage.completion_tokens,
                nccl_tracker.num_gpus,
            )

        nccl_tracker.add_sample(nccl_overhead)
        metrics["nccl_overhead_ms"] = nccl_overhead

    return metrics


def run_tensor_parallel_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the complete tensor parallel experiment."""
    print("=" * 60)
    print("  MLC-LLM Tensor Parallel Performance Analysis")
    print("  15-418 Parallel Computer Architecture Project")
    print("=" * 60)

    # System info
    print(f"\n[System Info]")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {platform.python_version()}")

    # GPU info
    gpu_count = get_gpu_count()
    gpu_utils = get_gpu_utilization()
    print(f"\n[GPU Info]")
    print(f"  GPUs detected: {gpu_count}")
    print(f"  Requested TP shards: {args.tp_shards}")

    if gpu_utils:
        for gpu in gpu_utils:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Utilization: {gpu['gpu_util_pct']:.1f}%")
            print(f"    Memory: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB")

    # Create engine
    engine, is_mock = create_tp_engine(
        args.model, args.tp_shards, args.device, args.mock
    )

    # NCCL tracker
    nccl_tracker = NCCLOverheadTracker(args.tp_shards)

    if is_mock:
        print("\n" + "=" * 60)
        print("  RUNNING IN MOCK MODE - Simulated multi-GPU behavior")
        print("=" * 60)

    results = {
        "config": {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "tensor_parallel_shards": args.tp_shards,
            "num_runs": args.num_runs,
            "warmup_runs": args.warmup_runs,
            "is_mock": is_mock,
        },
        "system": {
            "platform": platform.system(),
            "gpu_count": gpu_count,
            "gpu_info": gpu_utils,
        },
        "runs": [],
        "summary": {},
    }

    try:
        # Warmup
        print(f"\n[Warmup] Running {args.warmup_runs} warmup iterations...")
        for i in range(args.warmup_runs):
            _ = run_tp_inference(engine, args.prompt, args.max_tokens, is_mock, nccl_tracker)
            print(f"  Warmup {i + 1}/{args.warmup_runs} complete")

        # Clear warmup samples
        nccl_tracker.samples = []

        # Measurement runs
        print(f"\n[Measurement] Running {args.num_runs} iterations...")
        latencies = []
        tokens_per_sec_list = []
        nccl_overheads = []

        for i in range(args.num_runs):
            # Get GPU utilization before
            gpu_before = get_gpu_utilization()

            metrics = run_tp_inference(
                engine, args.prompt, args.max_tokens, is_mock, nccl_tracker
            )

            # Get GPU utilization after
            gpu_after = get_gpu_utilization()
            if gpu_after:
                metrics["gpu_utilization"] = [g["gpu_util_pct"] for g in gpu_after]

            results["runs"].append(metrics)
            latencies.append(metrics["latency_s"])
            tokens_per_sec_list.append(metrics.get("tokens_per_second", 0))
            nccl_overheads.append(metrics.get("nccl_overhead_ms", 0))

            print(
                f"  Run {i + 1}/{args.num_runs}: "
                f"Latency={metrics['latency_s']:.3f}s, "
                f"Throughput={metrics.get('tokens_per_second', 0):.1f} tok/s, "
                f"NCCL={metrics.get('nccl_overhead_ms', 0):.2f}ms"
            )

        # Compute summary
        nccl_summary = nccl_tracker.get_estimated_overhead_ms()

        results["summary"] = {
            "mean_latency_s": statistics.mean(latencies),
            "std_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_latency_s": min(latencies),
            "max_latency_s": max(latencies),
            "mean_tokens_per_sec": statistics.mean(tokens_per_sec_list),
            "std_tokens_per_sec": (
                statistics.stdev(tokens_per_sec_list) if len(tokens_per_sec_list) > 1 else 0
            ),
            "nccl_overhead": nccl_summary,
            "tensor_parallel_shards": args.tp_shards,
        }

        # Get final GPU utilization
        final_gpu_utils = get_gpu_utilization()
        if final_gpu_utils:
            results["summary"]["final_gpu_utilization"] = [
                g["gpu_util_pct"] for g in final_gpu_utils
            ]

    finally:
        if hasattr(engine, "terminate"):
            engine.terminate()

    # Print summary
    print("\n" + "=" * 60)
    print("  TENSOR PARALLEL RESULTS SUMMARY")
    print("=" * 60)
    summary = results["summary"]
    print(f"\n  Tensor Parallel Shards: {args.tp_shards}")
    print(f"\n  Tokens/sec (mean):       {summary['mean_tokens_per_sec']:.1f}")
    print(f"  Tokens/sec (std):        {summary['std_tokens_per_sec']:.1f}")
    print(f"\n  Total Latency (mean):    {summary['mean_latency_s']:.3f} s")
    print(f"  Total Latency (std):     {summary['std_latency_s']:.3f} s")

    nccl = summary["nccl_overhead"]
    print(f"\n  NCCL Overhead (mean):    {nccl['mean']:.2f} ms")
    print(f"  NCCL Overhead (std):     {nccl['std']:.2f} ms")
    if nccl.get("is_estimated"):
        print(f"  [Note: NCCL overhead is estimated/simulated]")

    if gpu_utils:
        print(f"\n  GPU Utilization:")
        for gpu in gpu_utils:
            print(f"    GPU {gpu['index']}: {gpu['gpu_util_pct']:.1f}%")

    if is_mock:
        print(f"\n  [Note: Running in MOCK MODE - values are simulated]")

    print("=" * 60)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Results saved to: {args.output_file}]")

    return results


def main():
    """Main entry point."""
    args = parse_args()
    run_tensor_parallel_experiment(args)


if __name__ == "__main__":
    main()
