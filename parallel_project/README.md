# Parallel LLM Inference Analysis - 15-418 Project

This directory contains the analysis and experimentation code for our 15-418 Parallel Computer Architecture project on parallelizing LLM inference.

## Project Structure

```
parallel_project/
├── run_baseline.py          # Baseline inference experiments
├── run_tensor_parallel.py   # Tensor parallelism experiments
├── overlap_experiment.py    # Compute/communication overlap experiments
├── profile_gpu.sh           # GPU profiling shell script
├── trace_utils.py           # Performance tracing utilities
├── README.md                # This file
├── runtime_patch/           # Runtime patches for optimization
│   ├── __init__.py
│   └── stream_overlap.py    # CUDA stream overlap implementation
└── results/                 # Experiment results (generated)
    ├── baseline/
    ├── tensor_parallel/
    ├── overlap/
    └── profiles/
```

## Overview

This project analyzes parallelization opportunities in MLC-LLM inference, focusing on:

1. **Baseline Performance**: Establishing single-GPU performance metrics
2. **Tensor Parallelism**: Scaling across multiple GPUs with TP
3. **Compute/Communication Overlap**: Hiding latency through prefetching and pipelining

## Prerequisites

- Python 3.10+
- CUDA 11.8+ with compatible NVIDIA GPU(s)
- MLC-LLM installed (`pip install mlc-llm`)
- NumPy (`pip install numpy`)

### Optional for advanced profiling:
- NVIDIA Nsight Systems
- NVIDIA Nsight Compute

## Quick Start

### 1. Run Baseline Experiments

```bash
cd parallel_project

# Basic run
python run_baseline.py --model "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"

# With custom parameters
python run_baseline.py \
    --model "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC" \
    --max-tokens 512 \
    --num-runs 10 \
    --warmup-runs 3 \
    --enable-trace
```

### 2. Run Tensor Parallel Experiments

```bash
# Test with different TP configurations
python run_tensor_parallel.py \
    --model "HF://mlc-ai/Llama-3-70B-Instruct-q4f16_1-MLC" \
    --tp-sizes 1 2 4 \
    --num-runs 5

# With communication profiling
python run_tensor_parallel.py \
    --profile-communication \
    --enable-trace
```

### 3. Run Overlap Experiments

```bash
# Test different overlap strategies
python overlap_experiment.py \
    --overlap-modes none prefetch pipeline \
    --num-streams 2 \
    --enable-trace
```

### 4. GPU Profiling

```bash
# Basic GPU profiling
./profile_gpu.sh -m basic -s run_baseline.py

# Detailed profiling with Nsight
./profile_gpu.sh -m nsight --nsight -s run_tensor_parallel.py

# Timeline profiling
./profile_gpu.sh -m timeline -d 60 -s overlap_experiment.py
```

## Experiments

### Baseline (`run_baseline.py`)

Measures single-GPU inference performance:
- Latency (mean, std, percentiles)
- Throughput (tokens/second)
- Memory usage

**Key metrics collected:**
- Time to First Token (TTFT)
- Inter-Token Latency (ITL)
- End-to-End Latency

### Tensor Parallelism (`run_tensor_parallel.py`)

Analyzes scaling efficiency across multiple GPUs:
- Strong scaling efficiency
- Communication overhead
- GPU utilization balance

**Key analysis:**
- Speedup vs. ideal speedup
- Scaling efficiency percentage
- Inter-GPU communication bandwidth

### Overlap Experiments (`overlap_experiment.py`)

Tests strategies to hide latency:
- **None**: Baseline sequential execution
- **Prefetch**: Prefetch next batch during compute
- **Pipeline**: Pipelined decode steps
- **Double Buffer**: Double buffering for continuous batching
- **Full Overlap**: Maximum overlap strategy

## Output Files

Results are saved in JSON format:

```json
{
  "config": { ... },
  "runs": [
    {"latency_s": 1.23, "tokens_per_second": 45.6, ...},
    ...
  ],
  "summary": {
    "mean_latency_s": 1.25,
    "std_latency_s": 0.05,
    "p95_latency_s": 1.32,
    "mean_tokens_per_sec": 44.8
  }
}
```

## Chrome Trace Visualization

When `--enable-trace` is used, a Chrome trace file is generated. View it by:

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load the `.json` trace file

## Project Goals

1. **Characterize** baseline LLM inference performance
2. **Measure** tensor parallelism scaling efficiency
3. **Identify** compute vs. communication bottlenecks
4. **Implement** and evaluate overlap strategies
5. **Propose** optimizations for improved throughput

## Team

- 15-418 Parallel Computer Architecture
- Carnegie Mellon University

## Notes

- This code is isolated from the main MLC-LLM codebase
- No modifications are made to existing MLC-LLM files
- Results may vary based on hardware configuration
- Ensure sufficient GPU memory for larger models

## References

- [MLC-LLM Documentation](https://llm.mlc.ai/docs/)
- [TVM Documentation](https://tvm.apache.org/docs/)
- 15-418 Lecture Notes on GPU Architecture and Parallelism

