from __future__ import annotations
import hashlib
import struct
from pathlib import Path
from typing import Any

def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_int_digest(digest: Any, values: list[int]) -> None:
    for value in values:
        digest.update(struct.pack("<i", int(value)))


def tokenize_response_only(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_seqlen: int,
) -> tuple[list[int], list[int], int, int]:
    full_ids: list[int] = []
    labels: list[int] = []
    assistant_turns = 0
    history: list[dict[str, str]] = []
    for message in messages:
        history.append(message)
        if message["role"] != "assistant":
            continue
        assistant_turns += 1
        prompt_text = tokenizer.apply_chat_template(
            history[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        turn_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not turn_text.startswith(prompt_text):
            raise RuntimeError(
                "Qwen rendered chat template prefix mismatch between generation "
                "prompt and completed assistant turn"
            )
        encoded = tokenizer(
            turn_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        turn_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        if turn_ids[: len(full_ids)] != full_ids:
            raise RuntimeError(
                "Qwen chat template is not prefix-stable across LIMA turns"
            )
        labels.extend([-100] * (len(turn_ids) - len(labels)))
        response_start = len(prompt_text)
        for index, (start, _) in enumerate(offsets):
            # Mask a byte-pair token if it straddles the prompt/response
            # boundary. This guarantees that no token containing prompt bytes
            # contributes to the objective.
            if start >= response_start:
                labels[index] = turn_ids[index]
        full_ids = turn_ids

    untruncated_length = len(full_ids)
    full_ids = full_ids[:max_seqlen]
    labels = labels[:max_seqlen]
    if len(full_ids) != len(labels):
        raise RuntimeError("token/label length mismatch after truncation")
    return full_ids, labels, assistant_turns, untruncated_length
