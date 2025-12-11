#!/bin/bash
#
# profile_gpu.sh - GPU profiling script for MLC-LLM experiments
#
# This script profiles GPU utilization, memory usage, and power consumption
# during LLM inference experiments using nvidia-smi.
#
# If nvidia-smi is unavailable (e.g., on macOS), it generates a simulated
# gpu_profile.log with realistic values for testing purposes.
#
# 15-418 Parallel Computer Architecture Project
#

set -e

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE="${1:-gpu_profile.log}"
DURATION="${2:-60}"        # Duration in seconds
INTERVAL="${3:-1}"         # Sampling interval in seconds
SAMPLE_COUNT=$((DURATION / INTERVAL))

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  MLC-LLM GPU Profiler${NC}"
    echo -e "${BLUE}  15-418 Parallel Computing Project${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

check_nvidia_smi() {
    if command -v nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Generate a random float between min and max
random_float() {
    local min=$1
    local max=$2
    awk -v min="$min" -v max="$max" 'BEGIN{srand(); printf "%.1f", min+rand()*(max-min)}'
}

# Generate simulated GPU profile data (for macOS or systems without nvidia-smi)
generate_mock_profile() {
    echo -e "${YELLOW}[INFO] nvidia-smi not available. Generating simulated GPU profile...${NC}"
    echo ""

    # Write header comment
    {
        echo "# GPU Profile Log (SIMULATED)"
        echo "# Generated on $(get_timestamp)"
        echo "# Platform: $(uname -s) $(uname -m)"
        echo "# Duration: ${DURATION}s, Interval: ${INTERVAL}s"
        echo "# Note: This is simulated data for testing on non-NVIDIA systems"
        echo "#"
        echo "# Columns: timestamp, gpu_idx, gpu%, mem%, power_w, temp_c, mem_used_mb, mem_total_mb"
        echo "#"
    } > "$OUTPUT_FILE"

    # Simulate 2 GPUs with realistic workload patterns
    local gpu0_base_util=45
    local gpu1_base_util=42
    local base_mem_used=8500
    local mem_total=24576
    local base_power=180
    local base_temp=55

    echo -e "${GREEN}[INFO] Simulating ${SAMPLE_COUNT} samples over ${DURATION} seconds...${NC}"

    for ((i=0; i<SAMPLE_COUNT; i++)); do
        local timestamp=$(date -u "+%Y-%m-%d %H:%M:%S" -d "+$i seconds" 2>/dev/null || date -u "+%Y-%m-%d %H:%M:%S")

        # Simulate varying load (inference bursts)
        local burst_factor=1
        if (( i % 10 < 3 )); then
            burst_factor=2  # Simulate prefill burst
        fi

        # GPU 0
        local gpu0_util=$(random_float $((gpu0_base_util * burst_factor - 10)) $((gpu0_base_util * burst_factor + 15)))
        local gpu0_mem_pct=$(random_float 34 38)
        local gpu0_power=$(random_float $((base_power - 20)) $((base_power + 40)))
        local gpu0_temp=$(random_float $((base_temp - 3)) $((base_temp + 8)))
        local gpu0_mem_used=$(random_float $((base_mem_used - 200)) $((base_mem_used + 300)))

        # GPU 1
        local gpu1_util=$(random_float $((gpu1_base_util * burst_factor - 10)) $((gpu1_base_util * burst_factor + 15)))
        local gpu1_mem_pct=$(random_float 33 37)
        local gpu1_power=$(random_float $((base_power - 25)) $((base_power + 35)))
        local gpu1_temp=$(random_float $((base_temp - 2)) $((base_temp + 7)))
        local gpu1_mem_used=$(random_float $((base_mem_used - 250)) $((base_mem_used + 250)))

        # Clamp values
        gpu0_util=$(awk -v v="$gpu0_util" 'BEGIN{if(v>100)v=100;if(v<0)v=0;print v}')
        gpu1_util=$(awk -v v="$gpu1_util" 'BEGIN{if(v>100)v=100;if(v<0)v=0;print v}')

        # Write to log
        echo "${timestamp}, 0, ${gpu0_util}, ${gpu0_mem_pct}, ${gpu0_power}, ${gpu0_temp}, ${gpu0_mem_used}, ${mem_total}" >> "$OUTPUT_FILE"
        echo "${timestamp}, 1, ${gpu1_util}, ${gpu1_mem_pct}, ${gpu1_power}, ${gpu1_temp}, ${gpu1_mem_used}, ${mem_total}" >> "$OUTPUT_FILE"

        # Progress indicator
        if (( i % 10 == 0 )); then
            echo -ne "\r  Progress: $i / $SAMPLE_COUNT samples"
        fi

        # Small delay to make timestamps more realistic
        sleep 0.01
    done

    echo -e "\n${GREEN}[DONE] Simulated profile saved to: ${OUTPUT_FILE}${NC}"
}

# Run real nvidia-smi profiling
run_nvidia_profile() {
    echo -e "${GREEN}[INFO] nvidia-smi detected. Starting GPU profiling...${NC}"
    echo ""
    echo -e "  Output file: ${OUTPUT_FILE}"
    echo -e "  Duration: ${DURATION} seconds"
    echo -e "  Interval: ${INTERVAL} seconds"
    echo -e "  Samples: ${SAMPLE_COUNT}"
    echo ""

    # Write header
    {
        echo "# GPU Profile Log (nvidia-smi dmon)"
        echo "# Generated on $(get_timestamp)"
        echo "# Duration: ${DURATION}s, Interval: ${INTERVAL}s"
        echo "#"
    } > "$OUTPUT_FILE"

    # Run nvidia-smi dmon
    # -s: select metrics (p=power, u=utilization, c=proc/mem clocks, m=memory)
    # -d: delay between samples
    # -c: count of samples
    echo -e "${YELLOW}[INFO] Running: nvidia-smi dmon -s pucm -d ${INTERVAL} -c ${SAMPLE_COUNT}${NC}"

    nvidia-smi dmon -s pucm -d "$INTERVAL" -c "$SAMPLE_COUNT" >> "$OUTPUT_FILE" 2>&1 &
    local DMON_PID=$!

    # Show progress
    echo -e "${GREEN}[INFO] Profiling in progress (PID: ${DMON_PID})...${NC}"

    # Wait for completion with progress updates
    local elapsed=0
    while kill -0 $DMON_PID 2>/dev/null; do
        sleep 5
        elapsed=$((elapsed + 5))
        echo -ne "\r  Elapsed: ${elapsed}s / ${DURATION}s"
    done

    echo -e "\n${GREEN}[DONE] Profile saved to: ${OUTPUT_FILE}${NC}"
}

# Display summary of collected data
display_summary() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Profile Summary${NC}"
    echo -e "${BLUE}========================================${NC}"

    if [[ -f "$OUTPUT_FILE" ]]; then
        local line_count=$(wc -l < "$OUTPUT_FILE")
        local file_size=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "  File: $OUTPUT_FILE"
        echo "  Size: $file_size"
        echo "  Lines: $line_count"
        echo ""
        echo "  First 10 data lines:"
        grep -v "^#" "$OUTPUT_FILE" | head -10 | while read line; do
            echo "    $line"
        done
    else
        echo -e "${RED}  Error: Output file not found${NC}"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header

    echo "Configuration:"
    echo "  Output file: $OUTPUT_FILE"
    echo "  Duration: ${DURATION}s"
    echo "  Interval: ${INTERVAL}s"
    echo ""

    if check_nvidia_smi; then
        run_nvidia_profile
    else
        generate_mock_profile
    fi

    display_summary

    echo ""
    echo -e "${GREEN}GPU profiling complete!${NC}"
    echo "Use trace_utils.py to analyze the results:"
    echo "  python -c \"from trace_utils import parse_gpu_profile, print_summary; print_summary(parse_gpu_profile('$OUTPUT_FILE'))\""
}

# Run main
main
