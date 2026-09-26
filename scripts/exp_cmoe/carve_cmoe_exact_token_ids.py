#!/usr/bin/env python3
"""Carve a CMoE model from exact pre-tokenized calibration windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


CMOE_ROOT = Path(__file__).resolve().parents[2] / "third_party/cmoe"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(path: Path, count: int, seqlen: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            ids = json.loads(line).get("input_ids")
            if not isinstance(ids, list) or len(ids) != seqlen:
                raise ValueError(f"{path}:{line_no} expected {seqlen} input_ids")
            rows.append(torch.tensor([int(token) for token in ids], dtype=torch.long))
    if len(rows) != count:
        raise ValueError(f"{path} provided {len(rows)} rows, expected exactly {count}")
    return torch.stack(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--calibration-token-ids", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--n-shared", type=int, required=True)
    parser.add_argument("--n-activated", type=int, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--calib-samples", type=int, required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--k-act", type=int, default=10)
    parser.add_argument("--bias-speed", type=float, default=0.001)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    routed_total = args.n_experts - args.n_shared
    if args.n_shared < 0 or routed_total <= 0 or args.n_activated > routed_total:
        raise ValueError(
            f"invalid topology S{args.n_shared}A{args.n_activated}E{args.n_experts}"
        )
    if (args.save_dir / "state_dict.pt").exists():
        raise FileExistsError(f"refusing to overwrite {args.save_dir / 'state_dict.pt'}")
    if args.calibration_manifest is not None:
        bundle = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
        if int(bundle.get("seed", -1)) != int(args.seed):
            raise ValueError(
                f"calibration manifest seed={bundle.get('seed')} does not match --seed={args.seed}"
            )
        if int(bundle.get("seqlen", -1)) != int(args.seqlen):
            raise ValueError("calibration manifest seqlen does not match carve seqlen")

    seed_everything(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if str(CMOE_ROOT) not in sys.path:
        sys.path.insert(0, str(CMOE_ROOT))
    import run_cmoe

    device = torch.device("cuda:0")
    run_cmoe.DEV = device
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    process_started = time.time()
    if "qwen" in args.base_model.lower():
        model = run_cmoe.get_qwen(args.base_model)
    else:
        model = run_cmoe.get_llama(args.base_model)
    model.seqlen = int(args.seqlen)
    model.eval()
    if int(model.config.intermediate_size) % int(args.n_experts) != 0:
        raise ValueError(
            f"intermediate_size={model.config.intermediate_size} is not divisible by E={args.n_experts}"
        )

    batch = load_rows(args.calibration_token_ids, args.calib_samples, args.seqlen)
    target = batch.clone()
    target[:, :-1] = -100
    dataloader = [(batch, target)]
    carve_args = SimpleNamespace(
        model=args.base_model,
        dataset=str(args.calibration_token_ids.resolve()),
        seed=args.seed,
        calib_samples=args.calib_samples,
        calib_bsz=args.calib_samples,
        nexperts=args.n_experts,
        nactivated=args.n_activated,
        nshared=args.n_shared,
        k_act=args.k_act,
        bias_speed=args.bias_speed,
        skip_internal_ppl=True,
    )
    started = time.time()
    carved, _, _, _ = run_cmoe.cmoe_sequential(model, dataloader, device, carve_args)
    state_tmp = args.save_dir / ".state_dict.pt.tmp"
    torch.save(carved.state_dict(), state_tmp)
    os.replace(state_tmp, args.save_dir / "state_dict.pt")
    manifest = {
        "base_model": str(Path(args.base_model).resolve()),
        "dataset": str(args.calibration_token_ids.resolve()),
        "dataset_sha256": sha256_file(args.calibration_token_ids),
        "calibration_bundle_manifest": (
            str(args.calibration_manifest.resolve()) if args.calibration_manifest else None
        ),
        "calibration_format": "exact_token_ids_jsonl",
        "calibration_text_roundtrip_used": False,
        "seed": int(args.seed),
        "calib_samples": int(args.calib_samples),
        "calib_bsz": int(args.calib_samples),
        "seqlen": int(args.seqlen),
        "nshared": int(args.n_shared),
        "nactivated": int(args.n_activated),
        "nexperts": int(args.n_experts),
        "k_act": int(args.k_act),
        "bias_speed": float(args.bias_speed),
        "nsamples": 0,
        "skip_sft": True,
        "seed_scope": ["python", "numpy", "torch_cpu_cuda", "calibration_artifact"],
        "elapsed_sec": time.time() - started,
        "process_elapsed_sec": time.time() - process_started,
        "cuda_peak_memory": {
            "max_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "max_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        },
    }
    manifest_tmp = args.save_dir / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(manifest_tmp, args.save_dir / "manifest.json")
    (args.save_dir / ".ready").write_text("ready\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
