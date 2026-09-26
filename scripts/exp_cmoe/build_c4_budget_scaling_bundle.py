#!/usr/bin/env python3
"""Build a pinned, tokenizer-specific C4 budget-scaling bundle.

The raw C4 document stream is selected once and consumed by every requested
model tokenizer. Each tokenizer receives an exact int32 [windows, seqlen]
memory-mapped training array. Calibration, train, development, and test raw
documents are disjoint. Small JSONL probes are materialized for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/cmoe_c4/budget_scaling_v1"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DEFAULT_TOKENIZERS = {
    "llama_2_7b_hf": Path("models/Llama-2-7b-hf"),
    "qwen2_5_7b": Path("models/Qwen2.5-7B"),
}
BUDGET_WINDOWS = {"40m": 20_480}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int_set(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    if not result:
        raise ValueError("empty shard selection")
    return sorted(result)


def parse_tokenizers(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_TOKENIZERS)
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--tokenizer must be NAME=/absolute/path")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or name in result:
            raise ValueError(f"invalid or repeated tokenizer name: {name!r}")
        result[name] = path
    return result


def c4_uri(split: str, shard: int, revision: str) -> str:
    if split == "train":
        return (
            f"hf://datasets/allenai/c4@{revision}/en/"
            f"c4-train.{shard:05d}-of-01024.json.gz"
        )
    if split == "validation":
        return (
            f"hf://datasets/allenai/c4@{revision}/en/"
            f"c4-validation.{shard:05d}-of-00008.json.gz"
        )
    raise ValueError(split)


def c4_rows(
    *,
    split: str,
    shards: list[int],
    revision: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to build the C4 bundle") from exc
    for shard in shards:
        uri = c4_uri(split, shard, revision)
        stream = load_dataset(
            "json",
            data_files={split: uri},
            split=split,
            streaming=True,
        )
        if shuffle_buffer > 1:
            stream = stream.shuffle(
                seed=seed + shard * 1009,
                buffer_size=shuffle_buffer,
            )
        for stream_index, row in enumerate(stream):
            text = str(row.get("text") or "")
            if not text.strip():
                continue
            yield {
                "split": split,
                "shard": shard,
                "stream_index": stream_index,
                "uri": uri,
                "url": str(row.get("url") or ""),
                "timestamp": str(row.get("timestamp") or ""),
                "text": text,
            }


@dataclass
class WindowWriter:
    path: Path
    windows: int
    seqlen: int

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.array = np.lib.format.open_memmap(
            self.path,
            mode="w+",
            dtype=np.int32,
            shape=(self.windows, self.seqlen),
        )
        self.row = 0
        self.position = 0
        self.current = np.empty(self.seqlen, dtype=np.int32)
        self.digest = hashlib.sha256()
        self.completed_source: dict[str, Any] | None = None

    @property
    def full(self) -> bool:
        return self.row >= self.windows

    def add(
        self,
        token_ids: list[int],
        eos_token_id: int,
        source: dict[str, Any],
    ) -> None:
        if self.full:
            return
        values = np.asarray([*token_ids, int(eos_token_id)], dtype=np.int64)
        if values.size and (
            int(values.min()) < np.iinfo(np.int32).min
            or int(values.max()) > np.iinfo(np.int32).max
        ):
            raise ValueError("token id does not fit int32")
        values = values.astype(np.int32, copy=False)
        offset = 0
        while offset < values.size and not self.full:
            count = min(self.seqlen - self.position, values.size - offset)
            self.current[self.position : self.position + count] = values[
                offset : offset + count
            ]
            self.position += count
            offset += count
            if self.position == self.seqlen:
                self.array[self.row] = self.current
                self.digest.update(self.current.astype("<i4", copy=False).tobytes())
                self.row += 1
                self.position = 0
                if self.full:
                    self.completed_source = {
                        key: source.get(key)
                        for key in ("split", "shard", "stream_index", "url", "text_sha256")
                    }

    def finish(self) -> dict[str, Any]:
        if not self.full:
            raise RuntimeError(
                f"{self.path}: produced {self.row}/{self.windows} complete windows"
            )
        self.array.flush()
        del self.array
        return {
            "path": str(self.path),
            "shape": [self.windows, self.seqlen],
            "dtype": "int32",
            "tokens": self.windows * self.seqlen,
            "token_stream_sha256": self.digest.hexdigest(),
            "file_sha256": sha256_file(self.path),
            "bytes": self.path.stat().st_size,
            "completed_source": self.completed_source,
        }


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or (ids and isinstance(ids[0], list)):
        raise ValueError("tokenizer did not return one flat input_ids list")
    return [int(value) for value in ids]


def raw_record(row: dict[str, Any], role: str) -> dict[str, Any]:
    text = str(row["text"])
    return {
        "role": role,
        "split": row["split"],
        "shard": row["shard"],
        "stream_index": row["stream_index"],
        "uri": row["uri"],
        "url": row["url"],
        "timestamp": row["timestamp"],
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def consume_role(
    *,
    rows: Iterator[dict[str, Any]],
    role: str,
    writers: dict[str, WindowWriter],
    tokenizers: dict[str, Any],
    eos_ids: dict[str, int],
    raw_handle: Any,
    seen_hashes: set[str],
    forbidden_hashes: set[str],
) -> dict[str, Any]:
    documents = 0
    duplicate_documents = 0
    forbidden_documents = 0
    raw_chars = 0
    while not all(writer.full for writer in writers.values()):
        try:
            row = next(rows)
        except StopIteration as exc:
            pending = {
                name: f"{writer.row}/{writer.windows}"
                for name, writer in writers.items()
                if not writer.full
            }
            raise RuntimeError(f"C4 source exhausted for {role}: {pending}") from exc
        record = raw_record(row, role)
        text_sha = str(record["text_sha256"])
        if text_sha in forbidden_hashes:
            forbidden_documents += 1
            continue
        if text_sha in seen_hashes:
            duplicate_documents += 1
            continue
        seen_hashes.add(text_sha)
        raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        documents += 1
        raw_chars += len(record["text"])
        source = {**row, "text_sha256": text_sha}
        for name, writer in writers.items():
            if writer.full:
                continue
            writer.add(
                tokenizer_ids(tokenizers[name], record["text"]),
                eos_ids[name],
                source,
            )
    return {
        "documents": documents,
        "duplicate_documents_skipped": duplicate_documents,
        "forbidden_overlap_documents_skipped": forbidden_documents,
        "raw_characters": raw_chars,
        "unique_text_hashes": len(seen_hashes),
    }


def write_token_jsonl(
    array_path: Path,
    output_path: Path,
    indices: Iterable[int],
) -> dict[str, Any]:
    array = np.load(array_path, mmap_mode="r")
    digest = hashlib.sha256()
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for index in indices:
            row = np.asarray(array[int(index)], dtype=np.int32)
            digest.update(row.astype("<i4", copy=False).tobytes())
            handle.write(
                json.dumps(
                    {"source_window_index": int(index), "input_ids": row.tolist()},
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    os.replace(tmp, output_path)
    return {
        "path": str(output_path),
        "windows": count,
        "tokens": count * int(array.shape[1]),
        "token_stream_sha256": digest.hexdigest(),
        "file_sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }


def fixed_probe_indices(prefix_windows: int, count: int, seed: int) -> list[int]:
    if count > prefix_windows:
        raise ValueError(f"probe count {count} exceeds prefix {prefix_windows}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(prefix_windows), count))


def load_tokenizers(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to build the C4 bundle") from exc
    tokenizers: dict[str, Any] = {}
    eos_ids: dict[str, int] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            use_fast=True,
            local_files_only=True,
        )
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError(f"{name} tokenizer has no EOS token")
        tokenizers[name] = tokenizer
        eos_ids[name] = int(eos)
    return tokenizers, eos_ids


def build(args: argparse.Namespace) -> Path:
    out_root = args.out_root.expanduser().resolve()
    if out_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing bundle root: {out_root}"
        )
    tokenizer_paths = parse_tokenizers(args.tokenizer)
    tokenizers, eos_ids = load_tokenizers(tokenizer_paths)
    train_shards = parse_int_set(args.train_shards)
    validation_shards = parse_int_set(args.validation_shards)
    if args.train_windows < max(BUDGET_WINDOWS.values()):
        raise ValueError("C4-40M requires at least 20480 train windows")
    if args.probe_windows > min(BUDGET_WINDOWS.values()):
        raise ValueError("probe windows exceed the 4M prefix")

    build_root = out_root.with_name(f".{out_root.name}.building.{os.getpid()}")
    if build_root.exists():
        raise FileExistsError(build_root)
    build_root.mkdir(parents=True)
    (build_root / ".building").write_text(
        json.dumps({"pid": os.getpid(), "target": str(out_root)}) + "\n",
        encoding="utf-8",
    )

    writer_groups: dict[str, dict[str, WindowWriter]] = {}
    for role, windows in (
        ("calibration", args.calibration_windows),
        ("train_master", args.train_windows),
        ("c4_dev", args.eval_windows),
        ("c4_test", args.eval_windows),
    ):
        writer_groups[role] = {
            name: WindowWriter(
                build_root / name / f"{role}_seed{args.seed}_n{windows}_seqlen{args.seqlen}.npy",
                windows,
                args.seqlen,
            )
            for name in tokenizer_paths
        }

    train_rows = c4_rows(
        split="train",
        shards=train_shards,
        revision=args.dataset_revision,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    validation_rows = c4_rows(
        split="validation",
        shards=validation_shards,
        revision=args.dataset_revision,
        seed=args.seed + 1_000_003,
        shuffle_buffer=args.shuffle_buffer,
    )
    raw_dir = build_root / "raw"
    raw_dir.mkdir(parents=True)
    train_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    role_stats: dict[str, Any] = {}
    with (raw_dir / "calibration.jsonl").open("w", encoding="utf-8") as handle:
        role_stats["calibration"] = consume_role(
            rows=train_rows,
            role="calibration",
            writers=writer_groups["calibration"],
            tokenizers=tokenizers,
            eos_ids=eos_ids,
            raw_handle=handle,
            seen_hashes=train_hashes,
            forbidden_hashes=set(),
        )
    with (raw_dir / "train_master.jsonl").open("w", encoding="utf-8") as handle:
        role_stats["train_master"] = consume_role(
            rows=train_rows,
            role="train_master",
            writers=writer_groups["train_master"],
            tokenizers=tokenizers,
            eos_ids=eos_ids,
            raw_handle=handle,
            seen_hashes=train_hashes,
            forbidden_hashes=set(),
        )
    with (raw_dir / "c4_dev.jsonl").open("w", encoding="utf-8") as handle:
        role_stats["c4_dev"] = consume_role(
            rows=validation_rows,
            role="c4_dev",
            writers=writer_groups["c4_dev"],
            tokenizers=tokenizers,
            eos_ids=eos_ids,
            raw_handle=handle,
            seen_hashes=validation_hashes,
            forbidden_hashes=train_hashes,
        )
    with (raw_dir / "c4_test.jsonl").open("w", encoding="utf-8") as handle:
        role_stats["c4_test"] = consume_role(
            rows=validation_rows,
            role="c4_test",
            writers=writer_groups["c4_test"],
            tokenizers=tokenizers,
            eos_ids=eos_ids,
            raw_handle=handle,
            seen_hashes=validation_hashes,
            forbidden_hashes=train_hashes,
        )

    writer_manifests: dict[str, dict[str, Any]] = {
        name: {} for name in tokenizer_paths
    }
    for role, writers in writer_groups.items():
        for name, writer in writers.items():
            writer_manifests[name][role] = writer.finish()

    for name, tokenizer_path in tokenizer_paths.items():
        model_dir = build_root / name
        files: dict[str, Any] = {}
        train_path = Path(writer_manifests[name]["train_master"]["path"])
        calibration_npy = Path(writer_manifests[name]["calibration"]["path"])
        dev_npy = Path(writer_manifests[name]["c4_dev"]["path"])
        test_npy = Path(writer_manifests[name]["c4_test"]["path"])

        files["train_master"] = writer_manifests[name]["train_master"]
        files["calibration_npy"] = writer_manifests[name]["calibration"]
        files["c4_dev"] = writer_manifests[name]["c4_dev"]
        files["c4_test_npy"] = writer_manifests[name]["c4_test"]
        files["calibration"] = write_token_jsonl(
            calibration_npy,
            model_dir
            / f"calibration_seed{args.seed}_n{args.calibration_windows}_seqlen{args.seqlen}.token_ids.jsonl",
            range(args.calibration_windows),
        )
        files["c4_dev_jsonl"] = write_token_jsonl(
            dev_npy,
            model_dir / f"c4_dev_seed{args.seed}_n{args.eval_windows}_seqlen{args.seqlen}.token_ids.jsonl",
            range(args.eval_windows),
        )
        files["c4_test"] = write_token_jsonl(
            test_npy,
            model_dir / f"c4_test_seed{args.seed}_n{args.eval_windows}_seqlen{args.seqlen}.token_ids.jsonl",
            range(args.eval_windows),
        )
        files["train_anchor"] = write_token_jsonl(
            train_path,
            model_dir
            / f"train_anchor_4m_seed{args.seed}_n{args.probe_windows}_seqlen{args.seqlen}.token_ids.jsonl",
            fixed_probe_indices(2_048, args.probe_windows, args.seed + 1701),
        )
        for offset, (budget, prefix) in enumerate(BUDGET_WINDOWS.items()):
            files[f"train_seen_{budget}"] = write_token_jsonl(
                train_path,
                model_dir
                / f"train_seen_{budget}_seed{args.seed}_n{args.probe_windows}_seqlen{args.seqlen}.token_ids.jsonl",
                fixed_probe_indices(
                    prefix,
                    args.probe_windows,
                    args.seed + 2003 + offset * 1009,
                ),
            )

        final_model_dir = out_root / name
        for row in files.values():
            row["path"] = str(
                final_model_dir / Path(str(row["path"])).relative_to(model_dir)
            )
        manifest = {
            "schema": "c4_budget_scaling_exact_tokens_v1",
            "seed": args.seed,
            "seqlen": args.seqlen,
            "tokenizer_name": name,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_eos_id": eos_ids[name],
            "dataset": "allenai/c4",
            "dataset_revision": args.dataset_revision,
            "train_shards": train_shards,
            "validation_shards": validation_shards,
            "shuffle_buffer": args.shuffle_buffer,
            "packing": "raw-document-order EOS concat then exact non-overlap windows",
            "train_windows": args.train_windows,
            "train_tokens": args.train_windows * args.seqlen,
            "budget_windows": BUDGET_WINDOWS,
            "files": files,
            "raw_role_stats": role_stats,
            "raw_train_validation_text_overlap": 0,
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    raw_files: dict[str, Any] = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        raw_files[path.stem] = {
            "path": str(out_root / "raw" / path.name),
            "file_sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    root_manifest = {
        "schema": "c4_budget_scaling_raw_selection_v1",
        "seed": args.seed,
        "seqlen": args.seqlen,
        "dataset": "allenai/c4",
        "dataset_revision": args.dataset_revision,
        "train_shards": train_shards,
        "validation_shards": validation_shards,
        "shuffle_buffer": args.shuffle_buffer,
        "tokenizers": {
            name: {
                "path": str(path),
                "manifest": str(out_root / name / "manifest.json"),
            }
            for name, path in tokenizer_paths.items()
        },
        "role_stats": role_stats,
        "raw_files": raw_files,
        "train_validation_text_overlap": 0,
    }
    (build_root / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (build_root / ".ready").write_text("ready\n", encoding="utf-8")
    (build_root / ".building").unlink()
    out_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(build_root, out_root)
    return out_root


def audit(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.out_root.expanduser().resolve()
    root_manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    checks: dict[str, Any] = {
        "root_ready": (out_root / ".ready").exists(),
        "schema": root_manifest.get("schema") == "c4_budget_scaling_raw_selection_v1",
        "revision": root_manifest.get("dataset_revision") == args.dataset_revision,
        "train_validation_text_overlap": (
            int(root_manifest.get("train_validation_text_overlap", -1)) == 0
        ),
    }
    models: dict[str, Any] = {}
    for name, row in root_manifest.get("tokenizers", {}).items():
        manifest_path = Path(row["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_checks: dict[str, bool] = {
            "manifest": manifest.get("schema") == "c4_budget_scaling_exact_tokens_v1",
            "seed": int(manifest.get("seed", -1)) == int(args.seed),
            "seqlen": int(manifest.get("seqlen", -1)) == int(args.seqlen),
            "overlap": int(manifest.get("raw_train_validation_text_overlap", -1)) == 0,
        }
        for key in (
            "train_master",
            "calibration_npy",
            "calibration",
            "c4_dev",
            "c4_test",
            "train_anchor",
            "train_seen_40m",
        ):
            file_row = manifest.get("files", {}).get(key, {})
            path = Path(str(file_row.get("path", "")))
            model_checks[f"{key}_exists"] = path.exists()
            if path.exists() and not args.fast:
                model_checks[f"{key}_sha256"] = (
                    sha256_file(path) == file_row.get("file_sha256")
                )
        train_path = Path(manifest["files"]["train_master"]["path"])
        if train_path.exists():
            train = np.load(train_path, mmap_mode="r")
            model_checks["train_shape"] = list(train.shape) == [
                args.train_windows,
                args.seqlen,
            ]
            model_checks["train_dtype"] = train.dtype == np.dtype("int32")
        models[name] = model_checks
    checks["models"] = models
    ok = all(value for key, value in checks.items() if key != "models") and all(
        all(model.values()) for model in models.values()
    )
    return {"ok": ok, "root": str(out_root), "checks": checks}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    tokenizers = parse_tokenizers(args.tokenizer)
    return {
        "schema": "c4_budget_scaling_plan_v1",
        "out_root": str(args.out_root.expanduser().resolve()),
        "dataset": "allenai/c4",
        "dataset_revision": args.dataset_revision,
        "train_shards": parse_int_set(args.train_shards),
        "validation_shards": parse_int_set(args.validation_shards),
        "seed": args.seed,
        "seqlen": args.seqlen,
        "train_windows": args.train_windows,
        "train_tokens": args.train_windows * args.seqlen,
        "calibration_windows": args.calibration_windows,
        "eval_windows": args.eval_windows,
        "probe_windows": args.probe_windows,
        "budget_windows": BUDGET_WINDOWS,
        "tokenizers": {name: str(path) for name, path in tokenizers.items()},
        "estimated_train_array_bytes_per_tokenizer": (
            args.train_windows * args.seqlen * np.dtype(np.int32).itemsize
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-revision", default=C4_REVISION)
    parser.add_argument("--train-shards", default="0-7")
    parser.add_argument("--validation-shards", default="0")
    parser.add_argument("--tokenizer", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--train-windows", type=int, default=20_480)
    parser.add_argument("--calibration-windows", type=int, default=8)
    parser.add_argument("--eval-windows", type=int, default=256)
    parser.add_argument("--probe-windows", type=int, default=256)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("command", choices=("plan", "build", "audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        payload = plan(args)
    elif args.command == "build":
        payload = {"ok": True, "root": str(build(args))}
    else:
        payload = audit(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload.get("ok", True):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
