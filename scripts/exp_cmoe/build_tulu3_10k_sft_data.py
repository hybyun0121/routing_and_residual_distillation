#!/usr/bin/env python3
"""Build a pinned, source-stratified Tulu-3 10K Qwen SFT stream."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.exp_cmoe.sft_tokenization import (
    sha256_path,
    tokenize_response_only,
    update_int_digest,
)


REPO_ID = "allenai/tulu-3-sft-mixture"
REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--selection-seed", type=int, default=20260922)
    parser.add_argument("--max-seqlen", type=int, default=2048)
    return parser.parse_args()


def stable_key(namespace: str, seed: int, source: str, row_id: str) -> str:
    value = f"{namespace}\0{seed}\0{source}\0{row_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target <= 0 or target > total:
        raise ValueError(f"target must be in [1, {total}], got {target}")
    quotas = {source: target * count // total for source, count in counts.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda source: (-(target * counts[source] % total), source),
    )
    for source in order[:remaining]:
        quotas[source] += 1
    assert sum(quotas.values()) == target
    assert all(0 <= quotas[source] <= counts[source] for source in counts)
    return quotas


def normalize_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty list")
    messages = []
    for message in raw:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"invalid message: {message!r}")
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("conversation has no assistant turn")
    return messages


def main() -> None:
    args = parse_args()
    if args.max_seqlen <= 0:
        raise ValueError("--max-seqlen must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "tulu3_10k_selected.jsonl"
    token_path = (
        output_dir
        / f"tulu3_10k_qwen25_response_only_seqlen{args.max_seqlen}.jsonl"
    )
    manifest_path = output_dir / "manifest.json"
    for path in (raw_path, token_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    dataset = load_dataset(
        REPO_ID,
        split="train",
        revision=args.revision,
    )
    if set(dataset.column_names) != {"id", "messages", "source"}:
        raise ValueError(f"unexpected columns: {dataset.column_names}")
    counts = Counter(str(source) for source in dataset["source"])
    quotas = proportional_quotas(dict(counts), args.rows)

    candidates: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    identity_counts: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(dataset):
        source, row_id = str(row["source"]), str(row["id"])
        identity_counts[(source, row_id)] += 1
        candidates[source].append(
            (
                stable_key(
                    "select-v1", args.selection_seed, source, f"{row_id}\0{index}"
                ),
                index,
                row_id,
            )
        )
    for source in candidates:
        candidates[source].sort()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    accepted: list[tuple[str, str, str]] = []
    rejected: Counter[str] = Counter()
    rows_truncated = 0
    assistant_turns = 0
    input_tokens = 0
    response_tokens = 0
    token_digest = hashlib.sha256()
    label_digest = hashlib.sha256()

    for source in sorted(quotas):
        source_accepted = 0
        for _, source_index, row_id in candidates[source]:
            if source_accepted == quotas[source]:
                break
            row = dataset[source_index]
            try:
                messages = normalize_messages(row["messages"])
                input_ids, labels, turns, untruncated = tokenize_response_only(
                    tokenizer, messages, args.max_seqlen
                )
            except (RuntimeError, TypeError, ValueError) as error:
                rejected[type(error).__name__] += 1
                continue
            active = sum(value != -100 for value in labels)
            if not input_ids or not active:
                rejected["no_active_response_after_truncation"] += 1
                continue
            raw_record = {
                "id": row_id,
                "source": source,
                "source_index": source_index,
                "messages": messages,
            }
            token_record = {
                "id": row_id,
                "source": source,
                "source_index": source_index,
                "input_ids": input_ids,
                "labels": labels,
                "input_token_count": len(input_ids),
                "response_token_count": active,
                "assistant_turn_count": turns,
            }
            order_key = stable_key(
                "order-v1", args.selection_seed, source, f"{row_id}\0{source_index}"
            )
            accepted.append(
                (
                    order_key,
                    json.dumps(raw_record, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(token_record, ensure_ascii=False, separators=(",", ":")),
                )
            )
            source_accepted += 1
            rows_truncated += int(untruncated > args.max_seqlen)
            assistant_turns += turns
            input_tokens += len(input_ids)
            response_tokens += active
        if source_accepted != quotas[source]:
            raise RuntimeError(
                f"source {source!r}: accepted {source_accepted}, expected {quotas[source]}"
            )

    accepted.sort()
    selected_digest = hashlib.sha256()
    with raw_path.open("x", encoding="utf-8") as raw_handle, token_path.open(
        "x", encoding="utf-8"
    ) as token_handle:
        for _, raw_line, token_line in accepted:
            raw_handle.write(raw_line + "\n")
            token_handle.write(token_line + "\n")
            record = json.loads(raw_line)
            token_record = json.loads(token_line)
            selected_digest.update(
                f"{record['source']}\0{record['id']}\0{record['source_index']}\n".encode(
                    "utf-8"
                )
            )
            update_int_digest(token_digest, token_record["input_ids"])
            update_int_digest(label_digest, token_record["labels"])

    manifest = {
        "schema": "tulu3_10k_qwen25_response_only_v1",
        "builder": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip(),
            "script_sha256": sha256_path(Path(__file__).resolve()),
        },
        "source": {
            "repo_id": REPO_ID,
            "revision": args.revision,
            "dataset_fingerprint": dataset._fingerprint,
            "rows": len(dataset),
            "columns": dataset.column_names,
            "counts_by_source": dict(sorted(counts.items())),
        },
        "selection": {
            "algorithm": "largest_remainder_source_quota_then_sha256_lowest_v1",
            "order": "sha256_order_v1",
            "seed": args.selection_seed,
            "target_rows": args.rows,
            "quotas_by_source": dict(sorted(quotas.items())),
            "duplicate_source_id_rows": sum(
                count - 1 for count in identity_counts.values() if count > 1
            ),
            "selected_source_id_order_sha256": selected_digest.hexdigest(),
            "rejected_before_quota_fill": dict(sorted(rejected.items())),
        },
        "tokenizer_path": str(Path(args.tokenizer_path).expanduser().resolve()),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "rows_written": len(accepted),
        "rows_truncated": rows_truncated,
        "assistant_turns": assistant_turns,
        "input_tokens": input_tokens,
        "response_tokens": response_tokens,
        "seqlen": args.max_seqlen,
        "train_on_prompt": False,
        "mask_policy": "all non-assistant-template positions are -100",
        "truncation": "right",
        "packing": False,
        "token_stream_sha256_int32le": token_digest.hexdigest(),
        "label_stream_sha256_int32le": label_digest.hexdigest(),
        "files": {
            "selected_messages": {
                "path": str(raw_path),
                "sha256": sha256_path(raw_path),
                "rows": len(accepted),
            },
            "train": {
                "path": str(token_path),
                "sha256": sha256_path(token_path),
                "rows": len(accepted),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
