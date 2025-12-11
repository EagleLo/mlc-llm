#!/usr/bin/env python3
"""
run_baseline.py - Baseline LLM inference without parallelization optimizations.

This script runs MLC-LLM inference in standard single-GPU mode to establish
baseline performance metrics for comparison with tensor parallel and overlap experiments.

Requirements:
- Uses the existing Python API from mlc_llm (python/mlc_llm)
- Loads a model using tensor_parallel_shards=1
- Runs inference on a fixed prompt
- Measures wall-clock latency and tokens/sec
- Falls back to mock mode if GPU is unavailable
- Prints: Total latency, Tokens/sec, GPU memory usage

15-418 Parallel Computer Architecture Project
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# GPU Utilities
# =============================================================================


def check_gpu_available() -> Tuple[bool, str]:
    """Check if a GPU is available for inference.

    Returns:
        Tuple of (is_available, device_type)
        device_type can be: 'cuda', 'metal', 'rocm', 'vulkan', or 'none'
    """
    system = platform.system()

    # Check for CUDA (NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check for Metal (macOS)
    if system == "Darwin":
        # Metal is available on all modern Macs
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "Metal" in result.stdout:
                return True, "metal"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check for ROCm (AMD)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, "rocm"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, "none"


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory usage information.

    Returns a dict with memory info, or empty dict if unavailable.
    Works on CUDA (nvidia-smi) and falls back gracefully on other platforms.
    """
    info = {
        "available": False,
        "device_name": "Unknown",
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "memory_free_mb": 0,
        "utilization_pct": 0,
    }

    # Try nvidia-smi for CUDA GPUs
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            # Take first GPU
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 5:
                info["available"] = True
                info["device_name"] = parts[0]
                info["memory_used_mb"] = float(parts[1])
                info["memory_total_mb"] = float(parts[2])
                info["memory_free_mb"] = float(parts[3])
                info["utilization_pct"] = float(parts[4])
                return info
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # On macOS, try to get Metal device info
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse for GPU name
                for line in result.stdout.split("\n"):
                    if "Chipset Model:" in line:
                        info["device_name"] = line.split(":")[-1].strip()
                        info["available"] = True
                        break
                # Note: Memory info not easily available on Metal
                info["memory_used_mb"] = -1  # Unknown
                info["memory_total_mb"] = -1
                info["memory_free_mb"] = -1
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return info


# =============================================================================
# Mock Engine for Testing Without GPU
# =============================================================================


@dataclass
class MockUsage:
    """Mock usage statistics for testing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MockMessage:
    """Mock message for testing."""

    content: str = ""
    role: str = "assistant"


@dataclass
class MockDelta:
    """Mock delta for streaming."""

    content: str = ""


@dataclass
class MockChoice:
    """Mock choice for testing."""

    index: int = 0
    message: Optional[MockMessage] = None
    delta: Optional[MockDelta] = None
    finish_reason: Optional[str] = None


@dataclass
class MockCompletionResponse:
    """Mock completion response for testing."""

    id: str = "mock-response"
    choices: List[MockChoice] = field(default_factory=list)
    usage: MockUsage = field(default_factory=MockUsage)


class MockChatCompletions:
    """Mock chat completions API for testing without GPU."""

    def __init__(self, tokens_per_second: float = 30.0):
        self.tokens_per_second = tokens_per_second
        self._sample_responses = [
            "Parallel computing is a type of computation in which many calculations "
            "are carried out simultaneously. Large problems can be divided into smaller "
            "ones, which can then be solved concurrently. This approach is essential for "
            "modern high-performance computing, enabling faster processing of complex tasks "
            "like scientific simulations, machine learning, and data analysis.",
            "In parallel computing, multiple processors execute instructions simultaneously "
            "to solve problems faster. Key concepts include task decomposition, load balancing, "
            "and synchronization. Modern GPUs excel at parallel workloads due to their "
            "thousands of cores designed for concurrent execution.",
        ]

    def create(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        stream: bool = False,
        **kwargs,
    ):
        """Create a mock chat completion."""
        # Simulate realistic token generation time
        response_text = random.choice(self._sample_responses)

        # Truncate to max_tokens (approximate)
        words = response_text.split()
        approx_tokens = min(len(words), max_tokens)
        response_text = " ".join(words[:approx_tokens])

        # Calculate prompt tokens (approximate)
        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = len(prompt_text.split())

        # Simulate generation time
        generation_time = approx_tokens / self.tokens_per_second
        time.sleep(generation_time)

        usage = MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=approx_tokens,
            total_tokens=prompt_tokens + approx_tokens,
        )

        if stream:
            # Return iterator for streaming
            def stream_generator():
                for word in response_text.split():
                    yield MockCompletionResponse(
                        choices=[MockChoice(delta=MockDelta(content=word + " "))],
                        usage=usage,
                    )
                    time.sleep(1.0 / self.tokens_per_second)

            return stream_generator()
        else:
            return MockCompletionResponse(
                choices=[MockChoice(message=MockMessage(content=response_text))],
                usage=usage,
            )


class MockChat:
    """Mock chat interface."""

    def __init__(self):
        self.completions = MockChatCompletions()


class MockMLCEngine:
    """Mock MLC Engine for testing without GPU/MLC installed."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.chat = MockChat()
        self._is_mock = True
        print(f"[MOCK MODE] Simulating model: {model}")

    def terminate(self):
        """Terminate the mock engine."""
        pass


# =============================================================================
# Baseline Runner
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline MLC-LLM inference for performance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_baseline.py
  python run_baseline.py --model "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
  python run_baseline.py --max-tokens 512 --num-runs 10
  python run_baseline.py --mock  # Force mock mode for testing
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
        default="Explain the concept of parallel computing in 100 words.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of inference runs for averaging",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs before measurement",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even if GPU is available",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, metal, vulkan, cpu",
    )
    return parser.parse_args()


def create_engine(
    model: str, device: str = "auto", force_mock: bool = False
) -> Tuple[Any, bool]:
    """Create an MLC engine or fall back to mock mode.

    Args:
        model: Model path or identifier
        device: Device to use
        force_mock: Force mock mode even if GPU available

    Returns:
        Tuple of (engine, is_mock)
    """
    if force_mock:
        print("\n[INFO] Mock mode forced by --mock flag")
        return MockMLCEngine(model), True

    # Check GPU availability
    gpu_available, gpu_type = check_gpu_available()

    if not gpu_available:
        print(f"\n[INFO] No GPU detected. Running in mock mode.")
        return MockMLCEngine(model), True

    if not MLC_AVAILABLE:
        print("\n[INFO] mlc_llm not installed. Running in mock mode.")
        print("       Install with: pip install mlc-llm")
        return MockMLCEngine(model), True

    # Try to create real engine
    try:
        print(f"\n[INFO] Creating MLCEngine on {gpu_type}...")
        print(f"[INFO] Model: {model}")

        # Configure engine with tensor_parallel_shards=1 for baseline
        engine_config = EngineConfig(
            tensor_parallel_shards=1,  # Single GPU baseline
            gpu_memory_utilization=0.85,
        )

        # Determine device string
        if device == "auto":
            device = gpu_type if gpu_type != "none" else "auto"

        engine = MLCEngine(
            model=model,
            device=device,
            mode="local",
            engine_config=engine_config,
        )

        print("[INFO] Engine created successfully!")
        return engine, False

    except Exception as e:
        print(f"\n[WARNING] Failed to create MLCEngine: {e}")
        print("[INFO] Falling back to mock mode.")
        return MockMLCEngine(model), True


def run_single_inference(
    engine: Any, prompt: str, max_tokens: int, is_mock: bool
) -> Dict[str, Any]:
    """Run a single inference and collect metrics.

    Args:
        engine: MLC engine or mock engine
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        is_mock: Whether running in mock mode

    Returns:
        Dictionary with timing and token metrics
    """
    # Record start time
    start_time = time.perf_counter()

    # Run inference
    response = engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=False,
    )

    # Record end time
    end_time = time.perf_counter()
    latency = end_time - start_time

    # Extract metrics
    metrics = {
        "latency_s": latency,
        "is_mock": is_mock,
    }

    if response.choices:
        output_text = response.choices[0].message.content
        metrics["output_text"] = output_text
        metrics["output_tokens"] = response.usage.completion_tokens
        metrics["prompt_tokens"] = response.usage.prompt_tokens
        metrics["total_tokens"] = response.usage.total_tokens
        metrics["tokens_per_second"] = response.usage.completion_tokens / latency

    return metrics


def run_baseline_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the complete baseline experiment.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("  MLC-LLM Baseline Performance Analysis")
    print("  15-418 Parallel Computer Architecture Project")
    print("=" * 60)

    # System info
    print(f"\n[System Info]")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {platform.python_version()}")

    # Check GPU
    gpu_available, gpu_type = check_gpu_available()
    gpu_info = get_gpu_memory_info()
    print(f"\n[GPU Info]")
    print(f"  Available: {gpu_available} ({gpu_type})")
    if gpu_info["available"]:
        print(f"  Device: {gpu_info['device_name']}")
        if gpu_info["memory_total_mb"] > 0:
            print(f"  Memory: {gpu_info['memory_used_mb']:.0f} / {gpu_info['memory_total_mb']:.0f} MB")

    # Create engine
    engine, is_mock = create_engine(args.model, args.device, args.mock)

    if is_mock:
        print("\n" + "=" * 60)
        print("  RUNNING IN MOCK MODE - Simulated timing values")
        print("=" * 60)

    results = {
        "config": {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "num_runs": args.num_runs,
            "warmup_runs": args.warmup_runs,
            "device": args.device,
            "is_mock": is_mock,
        },
        "system": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "gpu_type": gpu_type,
            "gpu_info": gpu_info,
        },
        "runs": [],
        "summary": {},
    }

    try:
        # Warmup runs
        print(f"\n[Warmup] Running {args.warmup_runs} warmup iterations...")
        for i in range(args.warmup_runs):
            _ = run_single_inference(engine, args.prompt, args.max_tokens, is_mock)
            print(f"  Warmup {i + 1}/{args.warmup_runs} complete")

        # Get GPU memory after warmup (when model is loaded)
        gpu_after_load = get_gpu_memory_info()
        if gpu_after_load["available"] and gpu_after_load["memory_used_mb"] > 0:
            print(f"\n[GPU Memory After Load]")
            print(f"  Used: {gpu_after_load['memory_used_mb']:.0f} MB")
            print(f"  Free: {gpu_after_load['memory_free_mb']:.0f} MB")
            results["gpu_memory_after_load"] = gpu_after_load

        # Measurement runs
        print(f"\n[Measurement] Running {args.num_runs} iterations...")
        latencies = []
        tokens_per_sec_list = []

        for i in range(args.num_runs):
            metrics = run_single_inference(
                engine, args.prompt, args.max_tokens, is_mock
            )
            results["runs"].append(metrics)
            latencies.append(metrics["latency_s"])
            tokens_per_sec_list.append(metrics.get("tokens_per_second", 0))

            print(
                f"  Run {i + 1}/{args.num_runs}: "
                f"Latency={metrics['latency_s']:.3f}s, "
                f"Throughput={metrics.get('tokens_per_second', 0):.1f} tok/s"
            )

        # Compute summary statistics
        import statistics

        results["summary"] = {
            "mean_latency_s": statistics.mean(latencies),
            "std_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_latency_s": min(latencies),
            "max_latency_s": max(latencies),
            "median_latency_s": statistics.median(latencies),
            "mean_tokens_per_sec": statistics.mean(tokens_per_sec_list),
            "std_tokens_per_sec": (
                statistics.stdev(tokens_per_sec_list) if len(tokens_per_sec_list) > 1 else 0
            ),
        }

        # Get final GPU memory
        gpu_final = get_gpu_memory_info()
        if gpu_final["available"]:
            results["gpu_memory_final"] = gpu_final

    finally:
        # Cleanup
        if hasattr(engine, "terminate"):
            engine.terminate()

    # Print summary
    print("\n" + "=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    summary = results["summary"]
    print(f"\n  Total Latency (mean):    {summary['mean_latency_s']:.3f} s")
    print(f"  Total Latency (std):     {summary['std_latency_s']:.3f} s")
    print(f"  Total Latency (min/max): {summary['min_latency_s']:.3f} / {summary['max_latency_s']:.3f} s")
    print(f"\n  Tokens/sec (mean):       {summary['mean_tokens_per_sec']:.1f}")
    print(f"  Tokens/sec (std):        {summary['std_tokens_per_sec']:.1f}")

    if gpu_info["available"]:
        print(f"\n  GPU Device:              {gpu_info['device_name']}")
        if gpu_info["memory_total_mb"] > 0:
            print(f"  GPU Memory Used:         {gpu_info['memory_used_mb']:.0f} MB")
            print(f"  GPU Memory Total:        {gpu_info['memory_total_mb']:.0f} MB")
            print(f"  GPU Utilization:         {gpu_info['utilization_pct']:.0f}%")

    if is_mock:
        print(f"\n  [Note: Running in MOCK MODE - values are simulated]")

    print("=" * 60)

    # Save results if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # Remove output_text from runs to save space
            results_to_save = results.copy()
            for run in results_to_save["runs"]:
                run.pop("output_text", None)
            json.dump(results_to_save, f, indent=2)
        print(f"\n[Results saved to: {args.output_file}]")

    return results


def main():
    """Main entry point."""
    args = parse_args()
    run_baseline_experiment(args)


if __name__ == "__main__":
    main()
