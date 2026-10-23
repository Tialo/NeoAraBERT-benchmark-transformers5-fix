#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv run benchmark_tf4_57_6.py --device cpu
uv run benchmark_tf5_0_0.py --device cpu
uv run benchmark_tf5_20_2.py --device cpu
uv run compare_results.py
