# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "torch==2.14.1",
#   "numpy==2.5.4",
#   "transformers==4.57.6",
#   "datasets==5.0.1",
#   "fast-disambig==0.4.0",
#   "huggingface-hub",
# ]
# ///
"""NeoAraBERT/Muradif full-test benchmark, shared by the three PEP 723 runners.

Run with: uv run benchmark_tf4_57_6.py [--device mps] [--batch-size 32]
The model card defines the embedding as last_hidden_state[:, 0, :].
Muradif accuracy is computed directly from the cosine similarities of
CLS embeddings for each anchor, synonym, and irrelevant sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from importlib.machinery import ModuleSpec
import sys
import time
from types import ModuleType
import zipfile

MODEL_ID = "U4RASD/NeoAraBERT"
DATASET_ID = "U4RASD/Muradif"
EXPECTED_ROWS = 38_554


def install_xformers_swiglu_shim() -> None:
    """Exact PR #3 fallback SwiGLU, exposed as the old xformers.ops.SwiGLU.

    Must run before importing Transformers or the model's remote model.py.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SwiGLU(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int | None = None,
            bias: bool = True,
        ) -> None:
            super().__init__()
            out_features = out_features or in_features
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.out_features = out_features or in_features
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    fake_xformers = ModuleType("xformers")
    fake_xformers.__path__ = []
    fake_xformers.__spec__ = ModuleSpec("xformers", loader=None, is_package=True)
    fake_ops = ModuleType("xformers.ops")
    fake_ops.__spec__ = ModuleSpec("xformers.ops", loader=None)
    fake_ops.SwiGLU = SwiGLU
    fake_xformers.ops = fake_ops
    sys.modules["xformers"] = fake_xformers
    sys.modules["xformers.ops"] = fake_ops


def sync_device(device: str, torch) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def save_weights_npz(model, path: Path) -> None:
    """Save ALL state_dict tensors as plain NumPy .npy members in an .npz.

    Streaming one tensor at a time avoids making a second copy of all weights.
    """
    import numpy as np

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, tensor in model.state_dict().items():
            values = tensor.detach().cpu().contiguous()
            if values.dtype == __import__("torch").bfloat16:
                raise TypeError(f"Unexpected bfloat16 state_dict tensor: {name}; expected fp32 weights")
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, values.numpy(), allow_pickle=False)


def load_rows(revision: str) -> tuple[list[dict], str]:
    from datasets import load_dataset

    data = load_dataset(DATASET_ID, split="test", revision=revision)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} Muradif rows; got {len(data)}")
    rows = list(data)
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        for field in ("context", "anchor_word", "syn_word", "irrelevant_word"):
            if not isinstance(row[field], str):
                raise ValueError(f"Row {index}: invalid {field}")
        if "_" not in row["context"]:
            raise ValueError(f"Row {index} has no underscore placeholder")
        digest.update(json.dumps([row[k] for k in ("context", "anchor_word", "syn_word", "irrelevant_word")],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return rows, digest.hexdigest()


def make_triplet(row: dict) -> tuple[str, str, str]:
    context = row["context"]
    # Some Muradif contexts contain multiple placeholders. Fill each one with
    # the same candidate to avoid leaving literal blanks in evaluated text.
    return tuple(context.replace("_", row[key])
                 for key in ("anchor_word", "syn_word", "irrelevant_word"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Allow model downloads to occur inside from_pretrained timing")
    return p.parse_args()


def run_benchmark(label: str, model_revision: str, *, patch_xformers: bool) -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive")

    # This must precede Transformers imports.
    if patch_xformers:
        install_xformers_swiglu_shim()

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import snapshot_download

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    np.random.seed(0)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this system")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this system")

    out = args.out_dir / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] loading all {EXPECTED_ROWS:,} Muradif rows ...", flush=True)
    rows, dataset_sha256 = load_rows(args.dataset_revision)

    # Warm the revision-specific HF cache first so model initialization is timed,
    # rather than the macOS network speed. Use same model_id + revision for loading.
    prefetch_s = 0.0
    if not args.no_prefetch:
        started = time.perf_counter()
        snapshot = snapshot_download(MODEL_ID, revision=model_revision)
        prefetch_s = time.perf_counter() - started
        resolved_model_commit = Path(snapshot).name
    else:
        resolved_model_commit = None
    effective_revision = resolved_model_commit or model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=effective_revision, trust_remote_code=True,
        local_files_only=not args.no_prefetch,
    )

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=effective_revision, trust_remote_code=True,
        local_files_only=not args.no_prefetch,
    )
    sync_device("cpu", torch)  # model is still on CPU during from_pretrained
    from_pretrained_s = time.perf_counter() - started
    model.eval()

    # Preserve float32 across all runs; comparing fp16/bf16 would add precision noise.
    started = time.perf_counter()
    # Do not pass dtype=float32 to .to(): rotary buffers are complex64 and
    # converting them to a real dtype would silently break the model.
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Expected full float32 parameters from the checkpoint")
    model = model.to(device=device)
    sync_device(device, torch)
    device_transfer_s = time.perf_counter() - started

    print(f"[{label}] saving full state_dict as weights.npz ...", flush=True)
    save_weights_npz(model, out / "weights.npz")

    tokenization_s = 0.0
    forward_s = 0.0

    @torch.inference_mode()
    def encode_batch(texts: list[str]) -> np.ndarray:
        nonlocal tokenization_s, forward_s
        started = time.perf_counter()
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=args.max_length).to(device)
        sync_device(device, torch)
        tokenization_s += time.perf_counter() - started

        started = time.perf_counter()
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS pooling
        sync_device(device, torch)
        forward_s += time.perf_counter() - started
        return embeddings.float().cpu().numpy()

    # Warmup is excluded from the benchmark and is identical in each runner.
    sample = make_triplet(rows[0])
    for _ in range(3):
        encode_batch(list(sample))
    forward_s = 0.0
    tokenization_s = 0.0

    # Disk-backed: [row, (anchor, synonym, irrelevant), CLS-dimension].
    dims = int(model.config.hidden_size)
    embeddings = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+",
                                          dtype="float32", shape=(len(rows), 3, dims))
    # Preserves all duplicates and row order: exactly 3 forward-encoded texts per row.
    texts = [sentence for row in rows for sentence in make_triplet(row)]
    total_texts = len(texts)
    print(f"[{label}] encoding all {total_texts:,} sentences on {device} ...", flush=True)
    start_encoding = time.perf_counter()
    for start in range(0, total_texts, args.batch_size):
        stop = min(start + args.batch_size, total_texts)
        encoded = encode_batch(texts[start:stop])
        embeddings.reshape(-1, dims)[start:stop] = encoded
        if (stop == total_texts) or (stop // args.batch_size) % 200 == 0:
            print(f"[{label}] {stop:,}/{total_texts:,}", flush=True)
    sync_device(device, torch)
    encode_wall_s = time.perf_counter() - start_encoding
    embeddings.flush()
    del texts

    # Every row: cos(anchor, synonym) > cos(anchor, irrelevant) => correct.
    # The model's CLS output is not assumed unit-normalized.
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    normalized = np.asarray(embeddings) / np.maximum(norms, 1e-12)
    positive = np.einsum("nd,nd->n", normalized[:, 0], normalized[:, 1])
    negative = np.einsum("nd,nd->n", normalized[:, 0], normalized[:, 2])
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Non-finite cosine score; check model/device compatibility")
    correct = positive > negative
    margins = positive - negative
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            anchor, synonym, irrelevant = make_triplet(row)
            value = {
                "row_index": i,
                "context": row["context"],
                "anchor_word": row["anchor_word"],
                "syn_word": row["syn_word"],
                "irrelevant_word": row["irrelevant_word"],
                "anchor_text": anchor,
                "synonym_text": synonym,
                "irrelevant_text": irrelevant,
                "cosine_synonym": float(positive[i]),
                "cosine_irrelevant": float(negative[i]),
                "margin": float(margins[i]),
                "prediction": "synonym" if correct[i] else "irrelevant_or_tie",
                "correct": bool(correct[i]),
            }
            f.write(json.dumps(value, ensure_ascii=False) + "\n")

    stats = {
        "label": label,
        "model_id": MODEL_ID,
        "requested_model_revision": model_revision,
        "resolved_model_commit": resolved_model_commit,
        "dataset_id": DATASET_ID,
        "dataset_revision": args.dataset_revision,
        "dataset_sha256": dataset_sha256,
        "versions": {
            name: importlib.metadata.version(name) for name in
            ("torch", "numpy", "transformers", "datasets")
        },
        "device": device,
        "dtype": "float32",
        "pooling": "CLS: last_hidden_state[:, 0, :]",
        "similarity": "cosine",
        "tokenizer_apply_stemming": getattr(tokenizer, "apply_stemming", None),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "row_count": len(rows),
        "sentences_encoded": total_texts,
        "accuracy": float(correct.mean()),
        "correct_count": int(correct.sum()),
        "incorrect_or_tied_count": int((~correct).sum()),
        "mean_positive_cosine": float(positive.mean()),
        "mean_negative_cosine": float(negative.mean()),
        "mean_cosine_margin": float(margins.mean()),
        "timing_seconds": {
            "prefetch": prefetch_s,
            "from_pretrained": from_pretrained_s,
            "transfer_to_device": device_transfer_s,
            "encode_wall": encode_wall_s,
            "tokenization_and_transfer": tokenization_s,
            "model_forward": forward_s,
        },
        "sentences_per_second_end_to_end": total_texts / encode_wall_s,
        "sentences_per_second_model_forward": total_texts / forward_s,
    }
    (out / "metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved {out/'metrics.json'}, {out/'predictions.jsonl'}, {out/'embeddings.npy'}, "
          f"{out/'weights.npz'}")


if __name__ == "__main__":
    run_benchmark("tf4_57_6", "main", patch_xformers=True)
