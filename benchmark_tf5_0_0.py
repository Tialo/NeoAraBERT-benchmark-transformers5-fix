# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "torch==2.14.1",
#   "numpy==2.5.4",
#   "transformers==5.0.0",
#   "datasets==5.1.0",
#   "fast-disambig==0.4.0",
#   "huggingface-hub",
# ]
# ///
"""PR #3 baseline compatibility test with Transformers 5.0.0.

Run: uv run benchmark_tf5_0_0.py --device mps
"""

from benchmark_tf4_57_6 import run_benchmark

if __name__ == "__main__":
    run_benchmark("tf5_0_0", "refs/pr/3", patch_xformers=False)
