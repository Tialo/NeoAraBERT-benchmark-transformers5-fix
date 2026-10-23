# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["numpy==2.5.4"]
# ///
"""Compare all 3 NeoAraBERT Muradif runs: weights, answers, scores, embeddings.

Run: uv run --script compare_results.py [--root results] [--atol 1e-5 --rtol 1e-5]
Writes summary.json, weights comparison JSON and per-row CSVs for all pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

LABELS = ("tf4_57_6", "tf5_0_0", "tf5_20_2")
ROLES = ("anchor", "synonym", "irrelevant")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--chunk-rows", type=int, default=256)
    p.add_argument("--top", type=int, default=20, help="Number of largest disagreements to print")
    return p.parse_args()


def print_run_metrics(results: dict[str, dict]) -> None:
    print("\n=== TIMING / QUALITY ===")
    print(f"{'run':13} {'TF':9} {'load (s)':>10} {'encode (s)':>12} "
          f"{'forward (s)':>12} {'sent/s':>12} {'accuracy':>10} {'correct':>9}")
    for label, data in results.items():
        t = data["timing_seconds"]
        print(f"{label:13} {data['versions']['transformers']:9} "
              f"{t['from_pretrained']:10.3f} {t['encode_wall']:12.3f} "
              f"{t['model_forward']:12.3f} "
              f"{data['sentences_per_second_end_to_end']:12.2f} "
              f"{data['accuracy']:10.6f} {data['correct_count']:9}")
    print("\nSpeedup ratios relative to TF 4.57.6 (higher is faster):")
    base = results[LABELS[0]]
    for label in LABELS[1:]:
        m = results[label]
        print(f"  {label}: load {base['timing_seconds']['from_pretrained'] / m['timing_seconds']['from_pretrained']:.4f}x; "
              f"forward {base['timing_seconds']['model_forward'] / m['timing_seconds']['model_forward']:.4f}x; "
              f"end-to-end {base['timing_seconds']['encode_wall'] / m['timing_seconds']['encode_wall']:.4f}x")


def compare_weights(left: Path, right: Path, atol: float, rtol: float) -> dict:
    result = {
        "keys_only_in_left": [], "keys_only_in_right": [],
        "compared_tensors": 0, "total_elements": 0,
        "exact_elements": 0, "close_elements": 0,
        "max_abs_difference": 0.0, "max_relative_difference": 0.0,
        "different_tensors": [], "shape_or_dtype_mismatches": [],
    }
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        ak, bk = set(a.files), set(b.files)
        result["keys_only_in_left"] = sorted(ak - bk)
        result["keys_only_in_right"] = sorted(bk - ak)
        for key in sorted(ak & bk):
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                result["shape_or_dtype_mismatches"].append({
                    "name": key, "left": {"shape": x.shape, "dtype": str(x.dtype)},
                    "right": {"shape": y.shape, "dtype": str(y.dtype)},
                })
                continue
            result["compared_tensors"] += 1
            result["total_elements"] += x.size
            exact = int(np.count_nonzero(x == y))
            result["exact_elements"] += exact
            close = np.isclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
            result["close_elements"] += int(np.count_nonzero(close))
            if x.size:
                abs_diff = np.abs(x - y)
                max_abs = float(np.max(abs_diff))
                rel = abs_diff / np.maximum(np.abs(x), 1e-12)
                max_rel = float(np.max(rel))
                result["max_abs_difference"] = max(result["max_abs_difference"], max_abs)
                result["max_relative_difference"] = max(result["max_relative_difference"], max_rel)
            else:
                max_abs = max_rel = 0.0
            if exact != x.size:
                result["different_tensors"].append({
                    "name": key, "total": x.size, "exact": exact,
                    "within_tolerance": int(np.count_nonzero(close)),
                    "max_abs": max_abs, "max_rel": max_rel,
                })
    result["identical"] = (
        not result["keys_only_in_left"] and not result["keys_only_in_right"]
        and not result["shape_or_dtype_mismatches"]
        and result["exact_elements"] == result["total_elements"]
    )
    result["allclose"] = (
        not result["keys_only_in_left"] and not result["keys_only_in_right"]
        and not result["shape_or_dtype_mismatches"]
        and result["close_elements"] == result["total_elements"]
    )
    return result


def iter_predictions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def compare_rows(
    left: Path, right: Path, out_csv: Path, *, atol: float, rtol: float,
    chunk_rows: int, top_n: int,
) -> dict:
    x = np.load(left / "embeddings.npy", mmap_mode="r")
    y = np.load(right / "embeddings.npy", mmap_mode="r")
    if x.shape != y.shape or x.ndim != 3 or x.shape[1] != 3:
        raise ValueError(f"Embedding tensor shapes incompatible: {x.shape}, {y.shape}")

    count = x.shape[0]
    result = {
        "rows": count, "embedding_dims": x.shape[2],
        "answer_disagreements": 0, "correctness_disagreements": 0,
        "score_differences_outside_tolerance": 0,
        "embeddings_exact_elements": 0, "embeddings_close_elements": 0,
        "embeddings_total_elements": int(np.prod(x.shape)),
        "max_embedding_abs_difference": 0.0,
        "max_cosine_synonym_score_difference": 0.0,
        "max_cosine_irrelevant_score_difference": 0.0,
        "per_role_max_abs_difference": {role: 0.0 for role in ROLES},
        "per_role_mean_embedding_cosine": {},
        "largest_disagreements": [],
    }
    # Running aggregate avoids storing all vector pairs in RAM.
    role_cos_sum = np.zeros(3, dtype=np.float64)
    top = []
    fields = [
        "row_index", "context", "anchor_word", "syn_word", "irrelevant_word",
        "left_prediction", "right_prediction", "prediction_equal", "left_correct", "right_correct",
        "left_cosine_synonym", "right_cosine_synonym", "abs_diff_cosine_synonym",
        "left_cosine_irrelevant", "right_cosine_irrelevant", "abs_diff_cosine_irrelevant",
        "left_margin", "right_margin", "margin_abs_difference",
        "anchor_max_abs_diff", "synonym_max_abs_diff", "irrelevant_max_abs_diff",
        "anchor_embedding_cosine", "synonym_embedding_cosine", "irrelevant_embedding_cosine",
        "all_embeddings_exact", "all_embeddings_within_tolerance",
    ]
    pred_a = iter_predictions(left / "predictions.jsonl")
    pred_b = iter_predictions(right / "predictions.jsonl")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for start in range(0, count, chunk_rows):
            end = min(start + chunk_rows, count)
            # float64 intermediate prevents loss of precision during difference tests
            a = np.asarray(x[start:end], dtype=np.float64)
            b = np.asarray(y[start:end], dtype=np.float64)
            d = np.abs(a - b)
            close = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            result["embeddings_exact_elements"] += int(np.count_nonzero(a == b))
            result["embeddings_close_elements"] += int(np.count_nonzero(close))
            result["max_embedding_abs_difference"] = max(result["max_embedding_abs_difference"], float(d.max()))
            maxabs = np.max(d, axis=2)
            cos = np.einsum("nrd,nrd->nr", a, b) / np.maximum(
                np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-24
            )
            role_cos_sum += cos.sum(axis=0)
            for r, role in enumerate(ROLES):
                result["per_role_max_abs_difference"][role] = max(
                    result["per_role_max_abs_difference"][role], float(maxabs[:, r].max())
                )
            for j in range(end - start):
                i = start + j
                p, q = next(pred_a), next(pred_b)
                if p["row_index"] != i or q["row_index"] != i:
                    raise ValueError(f"Prediction row order differs at row {i}")
                for key in ("context", "anchor_word", "syn_word", "irrelevant_word"):
                    if p[key] != q[key]:
                        raise ValueError(f"Different benchmark input at row {i}: {key}")
                answers_equal = p["prediction"] == q["prediction"]
                result["answer_disagreements"] += int(not answers_equal)
                result["correctness_disagreements"] += int(p["correct"] != q["correct"])
                pos_diff = abs(p["cosine_synonym"] - q["cosine_synonym"])
                neg_diff = abs(p["cosine_irrelevant"] - q["cosine_irrelevant"])
                margin_diff = abs(p["margin"] - q["margin"])
                result["max_cosine_synonym_score_difference"] = max(result["max_cosine_synonym_score_difference"], pos_diff)
                result["max_cosine_irrelevant_score_difference"] = max(result["max_cosine_irrelevant_score_difference"], neg_diff)
                score_close = (
                    np.isclose(p["cosine_synonym"], q["cosine_synonym"], atol=atol, rtol=rtol)
                    and np.isclose(p["cosine_irrelevant"], q["cosine_irrelevant"], atol=atol, rtol=rtol)
                )
                result["score_differences_outside_tolerance"] += int(not score_close)
                all_exact = bool(np.all(a[j] == b[j]))
                all_close = bool(np.all(close[j]))
                row = {
                    "row_index": i,
                    "context": p["context"], "anchor_word": p["anchor_word"],
                    "syn_word": p["syn_word"], "irrelevant_word": p["irrelevant_word"],
                    "left_prediction": p["prediction"], "right_prediction": q["prediction"],
                    "prediction_equal": answers_equal,
                    "left_correct": p["correct"], "right_correct": q["correct"],
                    "left_cosine_synonym": p["cosine_synonym"], "right_cosine_synonym": q["cosine_synonym"],
                    "abs_diff_cosine_synonym": pos_diff,
                    "left_cosine_irrelevant": p["cosine_irrelevant"], "right_cosine_irrelevant": q["cosine_irrelevant"],
                    "abs_diff_cosine_irrelevant": neg_diff,
                    "left_margin": p["margin"], "right_margin": q["margin"],
                    "margin_abs_difference": margin_diff,
                    **{f"{role}_max_abs_diff": float(maxabs[j, r]) for r, role in enumerate(ROLES)},
                    **{f"{role}_embedding_cosine": float(cos[j, r]) for r, role in enumerate(ROLES)},
                    "all_embeddings_exact": all_exact,
                    "all_embeddings_within_tolerance": all_close,
                }
                writer.writerow(row)
                # Pick largest meaningful differences, prioritize answer flips.
                priority = (int(not answers_equal), margin_diff, float(maxabs[j].max()))
                top.append((priority, {"row_index": i, "answer_changed": not answers_equal,
                                       "left_prediction": p["prediction"], "right_prediction": q["prediction"],
                                       "left_margin": p["margin"], "right_margin": q["margin"],
                                       "max_embedding_abs_diff": float(maxabs[j].max()),
                                       "context": p["context"]}))
            top.sort(key=lambda item: item[0], reverse=True)
            del top[top_n:]
    try:
        next(pred_a)
        raise ValueError("Left predictions JSONL has additional rows")
    except StopIteration:
        pass
    try:
        next(pred_b)
        raise ValueError("Right predictions JSONL has additional rows")
    except StopIteration:
        pass

    result["largest_disagreements"] = [v for _, v in top]
    result["per_role_mean_embedding_cosine"] = {
        role: float(role_cos_sum[r] / count) for r, role in enumerate(ROLES)
    }
    result["embeddings_exact"] = (
        result["embeddings_exact_elements"] == result["embeddings_total_elements"]
    )
    result["embeddings_allclose"] = (
        result["embeddings_close_elements"] == result["embeddings_total_elements"]
    )
    result["scores_and_answers_match_within_tolerance"] = (
        result["answer_disagreements"] == 0 and result["score_differences_outside_tolerance"] == 0
    )
    return result


def main() -> None:
    args = get_args()
    if args.atol < 0 or args.rtol < 0 or args.chunk_rows < 1 or args.top < 1:
        raise ValueError("atol/rtol must be nonnegative; chunk-rows and top must be positive")
    runs = {label: args.root / label for label in LABELS}
    metrics = {label: json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
               for label, directory in runs.items()}
    digests = {v["dataset_sha256"] for v in metrics.values()}
    counts = {v["row_count"] for v in metrics.values()}
    if len(digests) != 1 or len(counts) != 1:
        raise ValueError("The three runs have different dataset hashes or row counts; comparisons would be invalid")
    configurations = {(v["batch_size"], v["max_length"], v["dtype"], v["pooling"], v["similarity"], v.get("tokenizer_apply_stemming"))
                      for v in metrics.values()}
    if len(configurations) != 1:
        raise ValueError("Different embedding settings detected; re-run with matching batch-size, max-length and pooling")
    devices = {v["device"] for v in metrics.values()}
    if len(devices) != 1:
        print("WARNING: Run devices differ; performance/embedding precision comparisons are not controlled", file=sys.stderr)

    print_run_metrics(metrics)
    comparison = {"tolerances": {"atol": args.atol, "rtol": args.rtol},
                  "dataset_sha256": next(iter(digests)),
                  "runs": metrics, "pairs": {}}
    for a, b in itertools.combinations(LABELS, 2):
        name = f"{a}__vs__{b}"
        print(f"\n=== {name} ===", flush=True)
        weights = compare_weights(runs[a] / "weights.npz", runs[b] / "weights.npz", args.atol, args.rtol)
        (args.root / f"weights_{name}.json").write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
        print(f"Weights: exact={weights['identical']}, allclose={weights['allclose']}, "
              f"different_tensors={len(weights['different_tensors'])}, max_abs={weights['max_abs_difference']:.9g}, "
              f"keys_missing={len(weights['keys_only_in_left']) + len(weights['keys_only_in_right'])}")
        rows = compare_rows(runs[a], runs[b], args.root / f"rows_{name}.csv",
                            atol=args.atol, rtol=args.rtol, chunk_rows=args.chunk_rows,
                            top_n=args.top)
        print(f"Answers disagree: {rows['answer_disagreements']:,}/{rows['rows']:,}; "
              f"scores outside tolerance: {rows['score_differences_outside_tolerance']:,}; "
              f"embeddings exact: {rows['embeddings_exact']}; allclose: {rows['embeddings_allclose']}; "
              f"maximum embedding absolute difference: {rows['max_embedding_abs_difference']:.9g}")
        print(f"Cosine score max abs deltas: positive={rows['max_cosine_synonym_score_difference']:.9g}; "
              f"negative={rows['max_cosine_irrelevant_score_difference']:.9g}")
        print("Mean embedding vector cosines: " + ", ".join(
            f"{role}={value:.10f}" for role, value in rows['per_role_mean_embedding_cosine'].items()
        ))
        if rows['answer_disagreements'] or rows['score_differences_outside_tolerance'] or not rows['embeddings_exact']:
            print("Examples with largest differences (full row detail saved in CSV):")
            for item in rows['largest_disagreements'][:min(5, args.top)]:
                print(f"  row {item['row_index']:5d}, answer flip={item['answer_changed']}, "
                      f"margins={item['left_margin']:.8f}/{item['right_margin']:.8f}, "
                      f"max embedding diff={item['max_embedding_abs_diff']:.8g}")
        comparison["pairs"][name] = {"weights": weights, "rows": rows}

    (args.root / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nWrote comparison_summary.json, 3 weight comparison JSONs, "
          "and 3 CSVs (one row per Muradif triplet per model pair).")


if __name__ == "__main__":
    main()
