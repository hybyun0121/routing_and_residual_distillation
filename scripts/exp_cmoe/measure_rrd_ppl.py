"""Measure paper-style token-level PPL for Qwen2.5 CMoE checkpoints.

This mirrors the metric used by CMoE's ``run_cmoe.cmoe_ppl_eval``:
non-overlapping 2048-token windows, shift-token cross entropy, then
``exp(total_nll / (num_windows * seqlen))``.  The difference from
``third_party/cmoe/scripts/measure_ppl_only.py`` is model loading:
that script instantiates ``LlamaForCausalLM`` directly, while these Qwen2.5
experiments need the local CMoE skeleton loader from ``lmeval_cmoe.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from scripts.exp_cmoe.lmeval_cmoe import (  # noqa: E402
    build_dtype,
    build_moe_skeleton_model,
    load_checkpoint_artifacts,
    load_manifest,
    resolve_moe_type,
)
from scripts.exp_cmoe.mlp_router_patch import (  # noqa: E402
    attach_mlp_probe_router,
    build_probe_from_dir,
    build_probe_from_manifest,
    is_mlp_router_arch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RRD CMoE token-level PPL."
    )
    parser.add_argument("save_dir", help="Directory with manifest.json + state_dict.pt.")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--datasets", nargs="+", default=["wikitext2"],
                        choices=["wikitext2", "ptb", "c4", "ptb-new", "c4-new"])
    parser.add_argument("--jsonl-path", default=None,
                        help="Optional JSONL with {'text': ...} or exact {'input_ids': [...]} rows. "
                             "When set, measure this stream instead of --datasets.")
    parser.add_argument("--jsonl-name", default="jsonl",
                        help="Metric key used for --jsonl-path output.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--cmoe-repo-path", default="third_party/cmoe")
    parser.add_argument("--strict-load", type=int, default=1, choices=[0, 1])
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--trainable-delta-path", default=None)
    parser.add_argument("--cmoe-extra-state-path", default=None)
    parser.add_argument("--source-moe-dir", default=None)
    parser.add_argument(
        "--model-loss",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use model(..., labels=batch).loss; parity audits mirror trainer validation.",
    )
    parser.add_argument(
        "--use-attention-mask",
        type=int,
        default=0,
        choices=[0, 1],
        help="Pass an all-ones attention mask; parity audits mirror trainer validation.",
    )
    parser.add_argument(
        "--use-cache",
        type=int,
        default=0,
        choices=[0, 1],
        help="Forward with model KV cache enabled; default 0 preserves PPL evaluation behavior.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size; use the training validation batch size for parity audits.",
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=0,
        help="Optional positive cap on non-overlapping evaluation windows; 0 evaluates all.",
    )
    return parser.parse_args()


def load_test_encoding(dataset: str, base_model_path: str, seqlen: int, cmoe_repo_path: str) -> Any:
    if cmoe_repo_path not in sys.path:
        sys.path.insert(0, cmoe_repo_path)
    import datautils  # noqa: E402

    # Some CMoE datautils revisions import load_dataset inside individual
    # loaders but miss it in get_c4(). Patch the module namespace locally
    # instead of editing the external CMoE checkout.
    if not hasattr(datautils, "load_dataset"):
        from datasets import load_dataset  # noqa: E402

        datautils.load_dataset = load_dataset

    _, testenc = datautils.get_loaders(
        dataset,
        nsamples=128,
        seed=0,
        seqlen=seqlen,
        model=base_model_path,
    )
    return testenc


def load_jsonl_encoding(jsonl_path: str, base_model_path: str) -> Any:
    from transformers import AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    buf: List[int] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            token_ids = row.get("input_ids")
            text = row.get("text")
            if isinstance(token_ids, list):
                if not token_ids:
                    raise ValueError(f"{jsonl_path}:{line_no} has empty input_ids")
                buf.extend(int(x) for x in token_ids)
            else:
                if not isinstance(text, str):
                    raise ValueError(
                        f"{jsonl_path}:{line_no} requires string 'text' or list 'input_ids'"
                    )
                buf.extend(tokenizer.encode(text, add_special_tokens=False))
                if eos_id is not None:
                    buf.append(int(eos_id))
    if not buf:
        raise ValueError(f"{jsonl_path} produced no tokens")

    class Encoding:
        pass

    enc = Encoding()
    enc.input_ids = torch.tensor(buf, dtype=torch.long).unsqueeze(0)
    return enc


@torch.no_grad()
def token_ppl_eval(
    model: Any,
    testenc: Any,
    device: torch.device,
    seqlen: int,
    batch_size: int = 1,
    limit_windows: int = 0,
    use_attention_mask: bool = False,
    use_cache: bool = False,
    model_loss: bool = False,
) -> float:
    input_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    input_ids = input_ids[:, : (input_ids.numel() // seqlen) * seqlen]
    nsamples = input_ids.numel() // seqlen
    if limit_windows > 0:
        nsamples = min(nsamples, int(limit_windows))
    if nsamples <= 0:
        raise ValueError(f"Not enough tokens for one seqlen={seqlen} window.")

    original_use_cache = model.config.use_cache
    model.config.use_cache = bool(use_cache)
    model.eval()
    # CMoE's load-balancing controller updates ``extra_bias`` whenever
    # ``cus_training`` is enabled, even after ``model.eval()``. Evaluation
    # must be state preserving, and the trainer's lightweight validation
    # disables the same flag before every validation pass.
    for module in model.modules():
        if hasattr(module, "cus_training"):
            module.cus_training = False
    model.to(device)

    loss_fct = nn.CrossEntropyLoss()
    nlls: List[torch.Tensor] = []
    for start in tqdm(range(0, nsamples, batch_size), desc="PPL windows"):
        end = min(start + batch_size, nsamples)
        chunks = [
            input_ids[:, i * seqlen: (i + 1) * seqlen]
            for i in range(start, end)
        ]
        batch = torch.cat(chunks, dim=0).to(device)
        forward_kwargs = {"use_cache": bool(use_cache)}
        if use_attention_mask:
            forward_kwargs["attention_mask"] = torch.ones_like(batch)
        if model_loss:
            outputs = model(batch, labels=batch, **forward_kwargs)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("model-loss parity requested but model returned no loss")
            nlls.append(loss.float() * seqlen * batch.size(0))
            del outputs, batch
        else:
            outputs = model(batch, **forward_kwargs)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen * batch.size(0))
            del outputs, shift_logits, shift_labels, batch

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    model.config.use_cache = original_use_cache
    return ppl


def main() -> None:
    args = parse_args()
    manifest_path = os.path.join(args.save_dir, "manifest.json")
    state_dict_path = os.path.join(args.save_dir, "state_dict.pt")
    manifest = load_manifest(manifest_path, moe_type="cmoe")
    base_model_path = args.base_model_path or manifest["base_model"]
    moe_type = resolve_moe_type(manifest, "cmoe")
    if moe_type != "cmoe":
        raise ValueError(f"This Qwen evaluator expects CMoE manifests, got {moe_type!r}.")

    dtype = build_dtype(args.dtype)
    t0 = time.time()
    model, _tokenizer = build_moe_skeleton_model(
        base_model_path=base_model_path,
        manifest=manifest,
        dtype=dtype,
        moe_type="cmoe",
    )
    router_arch = manifest.get("router_arch") or manifest.get("cpt_router_arch")
    if is_mlp_router_arch(router_arch):
        probe_dir = str(manifest.get("cpt_mlp_router_probe_dir") or "").strip()
        if probe_dir:
            probe, _probe_manifest = build_probe_from_dir(probe_dir, model, dtype=dtype)
        else:
            probe = build_probe_from_manifest(manifest, model, dtype=dtype)
        attach_mlp_probe_router(
            model,
            probe,
            aggregation=str(manifest.get("cpt_mlp_router_aggregation", "uniform")),
            layer_mask=str(manifest.get("cpt_mlp_router_layer_mask") or ""),
        )
    model, missing, unexpected, loaded_artifacts = load_checkpoint_artifacts(
        model,
        manifest,
        manifest_path,
        full_state_dict_path=state_dict_path if os.path.exists(state_dict_path) else None,
        adapter_path=args.adapter_path,
        trainable_delta_path=args.trainable_delta_path,
        cmoe_extra_state_path=args.cmoe_extra_state_path,
        source_moe_dir=args.source_moe_dir,
        strict_full=bool(args.strict_load),
    )
    model.seqlen = args.seqlen

    device = torch.device(args.device)
    results: Dict[str, float] = {}
    if args.jsonl_path:
        print(f"\n=== {args.jsonl_name} token-level PPL from JSONL (seqlen={args.seqlen}) ===")
        testenc = load_jsonl_encoding(args.jsonl_path, base_model_path)
        ppl = token_ppl_eval(
            model,
            testenc,
            device,
            args.seqlen,
            batch_size=args.batch_size,
            limit_windows=args.limit_windows,
            use_attention_mask=bool(args.use_attention_mask),
            use_cache=bool(args.use_cache),
            model_loss=bool(args.model_loss),
        )
        print(f"{args.jsonl_name}: PPL = {ppl:.6f}")
        results[args.jsonl_name] = ppl
    else:
        for dataset in args.datasets:
            print(f"\n=== {dataset} token-level PPL (seqlen={args.seqlen}) ===")
            testenc = load_test_encoding(dataset, base_model_path, args.seqlen, args.cmoe_repo_path)
            ppl = token_ppl_eval(
                model,
                testenc,
                device,
                args.seqlen,
                batch_size=args.batch_size,
                limit_windows=args.limit_windows,
                use_attention_mask=bool(args.use_attention_mask),
                use_cache=bool(args.use_cache),
                model_loss=bool(args.model_loss),
            )
            print(f"{dataset}: PPL = {ppl:.6f}")
            results[dataset] = ppl

    out_path = args.output_json or os.path.join(args.save_dir, "ppl_token_post_sft.json")
    payload = {
        "metric": "token_level_ppl_cmoe_compatible",
        "definition": "non-overlapping seqlen windows; shift CE; exp(sum(loss*seqlen)/num_windows/seqlen)",
        "save_dir": args.save_dir,
        "base_model_path": base_model_path,
        "datasets": args.datasets,
        "jsonl_path": args.jsonl_path,
        "jsonl_name": args.jsonl_name,
        "seqlen": args.seqlen,
        "limit_windows": int(args.limit_windows),
        "batch_size": int(args.batch_size),
        "use_attention_mask": bool(args.use_attention_mask),
        "use_cache": bool(args.use_cache),
        "model_loss": bool(args.model_loss),
        "dtype": args.dtype,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "loaded_artifacts": loaded_artifacts,
        "elapsed_sec": time.time() - t0,
        "ppl": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
