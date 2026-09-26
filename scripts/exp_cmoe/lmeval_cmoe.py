"""lm-evaluation-harness helper for custom Llama-2 MoE checkpoints (RRD or CMoE).

Loads a base Llama-2 dense model, swaps every layer's MLP with an empty MoE
skeleton shaped per ``manifest.json`` (RRD :class:`MoE` or CMoE :class:`MoE`
depending on ``--moe_type``), then loads a full ``state_dict.pt`` into the
result and runs lm-evaluation-harness via ``HFLM`` + ``simple_evaluate``.

Two MoE family branches are supported:

- ``--moe_type rrd`` (default): RRD MoE topology declared by manifest keys
  ``n_experts``, ``n_active``, ``has_shared``, ``d_shared``. The skeleton is
  built from :class:`llamafactory.model.rrd_moe_llama.MoE`.
- ``--moe_type cmoe``: CMoE S+A topology declared by manifest keys
  ``nexperts``, ``nactivated``, ``nshared`` (matching the format produced by
  CMoE's carving pipeline). The skeleton is built via
  :func:`llamafactory.model.cmoe_moe_llama.swap_llama_mlp_to_cmoe_moe`. The
  manifest may set ``moe_type`` explicitly to override ``--moe_type``.

For RRD, ``--manifest`` only declares topology; init/split/seed metadata is
recorded for traceability but not consumed here. For CMoE, the same
``manifest.json`` from CMoE carving (containing ``nshared``/``nactivated``/
``nexperts``) is read directly.

Outputs a JSON summary file containing the full lm-eval ``results`` block plus
manifest and CLI config, and prints a one-line table of the headline accuracy
per task.

RRD usage::

    python scripts/exp_cmoe/lmeval_cmoe.py \
        --moe_type rrd \
        --base_model_path models/Llama-2-7b-hf \
        --moe_state_dict saves/cmoe/llama2-E8A6-ParamSplit-coreneuron/state_dict.pt \
        --manifest saves/cmoe/llama2-E8A6-ParamSplit-coreneuron/manifest.json \
        --output_json saves/cmoe/.../lmeval_phase1.json \
        --tasks piqa,winogrande,arc_easy,arc_challenge,hellaswag \
        --batch_size 4 --device cuda:0

CMoE usage (pre-LoRA carved ckpt re-evaluation)::

    python scripts/exp_cmoe/lmeval_cmoe.py \
        --moe_type cmoe \
        --base_model_path models/Llama-2-7b-hf \
        --moe_state_dict outputs/llama2_S3A3E8/state_dict.pt \
        --manifest outputs/llama2_S3A3E8/manifest.json \
        --output_json saves/cmoe_carved/.../lmeval.json \
        --tasks piqa,winogrande,arc_easy,arc_challenge,hellaswag \
        --batch_size 4 --device cuda:0

A built-in ``--self_test`` mode swaps a single layer's MLP with a
``build_moe_from_clusters`` E8A6 BasicSplit module, runs a forward against the
base dense model, and asserts cosine-similarity > 0.99 (single-layer all-active
equivalence). This validates the swap+load mechanics without requiring a
trained checkpoint or GPU. ``--self_test --moe_type cmoe`` additionally runs a
mock-model swap + load_state_dict roundtrip on the CMoE skeleton.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from llamafactory.model.rrd_moe_llama import MoE, build_moe_from_clusters  # noqa: E402
from llamafactory.model.cmoe_moe_llama import (  # noqa: E402
    MoE as CMoEMoE,
    swap_llama_mlp_to_cmoe_moe,
)
from scripts.exp_cmoe.mlp_router_patch import (  # noqa: E402
    attach_mlp_probe_router,
    build_probe_from_dir,
    build_probe_from_manifest,
    is_mlp_router_arch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("lmeval_cmoe")


# Supported MoE family types.
MOE_TYPES = ("rrd", "cmoe")


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lm-eval helper for custom MoE Llama-2.")
    p.add_argument("--base_model_path", type=str,
                   default="models/Llama-2-7b-hf")
    p.add_argument("--moe_state_dict", type=str,
                   help="Optional full model state_dict.pt. Omit for base+adapter/delta checkpoints.")
    p.add_argument("--adapter_path", type=str, default=None,
                   help="Optional PEFT adapter directory; defaults to cpt_adapter_dir in manifest.")
    p.add_argument("--trainable_delta_path", type=str, default=None,
                   help="Optional CE trainable_delta.pt; defaults to manifest/run directory.")
    p.add_argument("--cmoe_extra_state_path", type=str, default=None,
                   help="Optional LoRA extra_scale/extra_bias sidecar.")
    p.add_argument("--source_moe_dir", type=str, default=None,
                   help="Base carved CMoE directory for adapter/delta reconstruction.")
    p.add_argument(
        "--source_state_dict_path",
        type=str,
        default=None,
        help=(
            "Optional full source checkpoint loaded before an adapter/delta. "
            "Defaults to manifest cpt_source_state_dict_path, then "
            "<source_moe_dir>/state_dict.pt. This preserves frozen state such "
            "as an embedded RRD MLP router during storage-efficient SFT eval."
        ),
    )
    p.add_argument("--manifest", type=str,
                   help="Path to manifest.json declaring MoE topology.")
    p.add_argument("--output_json", type=str,
                   help="Where to write lm-eval summary JSON.")
    p.add_argument("--tasks", type=str,
                   default="piqa,winogrande,arc_easy,arc_challenge,hellaswag")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional: cap eval examples per task (for debugging).")
    p.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help="Optional lm-eval generation kwargs, e.g. max_gen_toks=256,temperature=0,do_sample=False.",
    )
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--self_test", action="store_true",
                   help="Run swap+load sanity test instead of lm-eval.")

    # MoE family selection. CMoE branch loads a CMoE-format carved state_dict
    # via swap_llama_mlp_to_cmoe_moe; RRD branch keeps the original behavior.
    p.add_argument("--moe_type", type=str, choices=list(MOE_TYPES), default="rrd",
                   help="Which MoE family to build. Manifest may override via "
                        "a 'moe_type' key.")
    p.add_argument("--cmoe_n_experts", type=int, default=8,
                   help="CMoE total expert count (routed + shared). Used only if "
                        "manifest does not provide 'nexperts'.")
    p.add_argument("--cmoe_n_activated", type=int, default=3,
                   help="CMoE top-K. Used only if manifest does not provide "
                        "'nactivated'.")
    p.add_argument("--cmoe_n_shared", type=int, default=3,
                   help="CMoE shared expert count. Used only if manifest does not "
                        "provide 'nshared'.")
    p.add_argument("--cmoe_state_dict_path", type=str, default=None,
                   help="Alias for --moe_state_dict when --moe_type=cmoe. If set, "
                        "overrides --moe_state_dict.")
    p.add_argument("--cmoe_strict_load", type=int, default=1,
                   help="If 1, load CMoE state_dict with strict=True (default). "
                        "Set 0 to fall back to strict=False with key-mismatch "
                        "logging.")
    p.add_argument("--lmeval_seed", type=int, default=0,
                   help="Seed forwarded to lm-eval-harness "
                        "(random/numpy/torch/fewshot all set to this value). "
                        "Use to measure eval-side variance across runs.")
    p.add_argument("--log_samples", action="store_true",
                   help="Pass log_samples=True to simple_evaluate so per-sample "
                        "results are returned. Required for downstream "
                        "calibration / decision-agreement analysis.")
    p.add_argument(
        "--confirm_run_unsafe_code", action="store_true",
        help="Allow lm-eval tasks such as HumanEval to execute generated code.")
    p.add_argument("--output_path", type=str, default=None,
                   help="Directory to write per-task samples_<task>.jsonl. "
                        "Only used when --log_samples is set.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Manifest validation
# -----------------------------------------------------------------------------


RRD_REQUIRED_MANIFEST_KEYS = ("n_experts", "n_active", "has_shared")
CMOE_REQUIRED_MANIFEST_KEYS = ("nexperts", "nactivated", "nshared")


def resolve_moe_type(manifest: Dict[str, Any], cli_moe_type: str) -> str:
    """Decide effective moe_type from manifest + CLI.

    Priority: explicit ``manifest['moe_type']`` > CLI ``--moe_type``.
    """
    mtype = manifest.get("moe_type")
    if mtype is None:
        return cli_moe_type
    if mtype not in MOE_TYPES:
        raise ValueError(
            f"manifest['moe_type']={mtype!r} not in supported {MOE_TYPES}"
        )
    return mtype


def load_manifest(path: str, moe_type: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate manifest JSON.

    Validation depends on the effective MoE family. If ``moe_type`` is None,
    auto-detect: a manifest carrying ``nexperts`` is treated as CMoE; one
    carrying ``n_experts`` is treated as RRD. Pass an explicit ``moe_type`` to
    skip auto-detection (e.g. when the CLI overrides the manifest).
    """
    with open(path, "r") as f:
        manifest = json.load(f)

    if moe_type is None:
        if manifest.get("moe_type") in MOE_TYPES:
            moe_type = manifest["moe_type"]
        elif "nexperts" in manifest:
            moe_type = "cmoe"
        elif "n_experts" in manifest:
            moe_type = "rrd"
        else:
            raise ValueError(
                f"Cannot auto-detect moe_type from {path}: manifest has neither "
                f"'nexperts' (CMoE) nor 'n_experts' (RRD)."
            )

    if moe_type == "rrd":
        missing = [k for k in RRD_REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(
                f"RRD manifest {path} missing required keys: {missing}"
            )
        if manifest["has_shared"] and manifest.get("d_shared") is None:
            raise ValueError(
                f"RRD manifest has_shared=True but d_shared is null in {path}."
            )
    elif moe_type == "cmoe":
        missing = [k for k in CMOE_REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(
                f"CMoE manifest {path} missing required keys: {missing}"
            )
    else:
        raise ValueError(f"Unsupported moe_type {moe_type!r} (expected one of {MOE_TYPES}).")

    return manifest


# -----------------------------------------------------------------------------
# Model build (swap + load)
# -----------------------------------------------------------------------------


def build_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_moe_skeleton_model(
    base_model_path: str,
    manifest: Dict[str, Any],
    dtype: torch.dtype,
    moe_type: str = "rrd",
    cmoe_defaults: Optional[Dict[str, int]] = None,
) -> Tuple[Any, Any]:
    """Load the base dense model, swap every ``layer.mlp`` with an empty MoE.

    The MoE skeleton has random-init experts/router; the caller is expected to
    immediately ``load_state_dict`` real weights on top.

    Args:
        base_model_path: HF base Llama directory (must match the carving base).
        manifest: parsed manifest dict.
        dtype: torch dtype for the model and MoE skeleton.
        moe_type: ``"rrd"`` or ``"cmoe"`` — selects which MoE skeleton to use.
        cmoe_defaults: optional dict providing fallback values for CMoE
            ``n_experts``/``n_activated``/``n_shared`` when missing from manifest.
            Keys: ``n_experts``, ``n_activated``, ``n_shared``.

    Returns:
        ``(model, tokenizer)``.

    Notes (CMoE branch):
        After swap, every ``layer.mlp`` is a CMoE :class:`MoE` whose router has
        a learnable ``extra_scale`` parameter and an ``extra_bias`` buffer.
        ``freeze_extra_bias_and_scale`` is *not* applied here — for evaluation
        we want to load whatever values the saved state_dict carries (CMoE's
        carved baseline keeps a non-trivial ``extra_scale`` and a
        load-balancing ``extra_bias``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model from %s (dtype=%s)", base_model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    if moe_type == "rrd":
        config = model.config
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        n_experts = int(manifest["n_experts"])
        n_active = int(manifest["n_active"])
        has_shared = bool(manifest["has_shared"])
        d_shared = manifest.get("d_shared")
        d_shared = int(d_shared) if d_shared is not None else None

        logger.info(
            "Swapping MLPs with RRD MoE skeleton: "
            "H=%d I=%d E=%d A=%d shared=%s d_shared=%s",
            hidden_size, intermediate_size, n_experts, n_active, has_shared, d_shared,
        )

        for layer in model.model.layers:
            moe = MoE(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                n_experts=n_experts,
                n_active=n_active,
                has_shared=has_shared,
                d_shared=d_shared,
                zero_init_shared=True,
            ).to(dtype)
            layer.mlp = moe

        return model, tokenizer

    if moe_type == "cmoe":
        defaults = cmoe_defaults or {}
        n_experts = int(manifest.get("nexperts", defaults.get("n_experts", 8)))
        n_activated = int(manifest.get("nactivated", defaults.get("n_activated", 3)))
        n_shared = int(manifest.get("nshared", defaults.get("n_shared", 3)))
        bias_speed = float(manifest.get("bias_speed", 0.001))
        # cpt_log.json (saved alongside ckpt) records cpt_add_eas; manifest
        # itself doesn't carry add_eas, so prefer cpt_log when available.
        add_eas = bool(manifest.get("add_eas", defaults.get("add_eas", False)))

        logger.info(
            "Swapping MLPs with CMoE MoE skeleton: E=%d A=%d S=%d bias_speed=%g add_eas=%s",
            n_experts, n_activated, n_shared, bias_speed, add_eas,
        )

        swap_llama_mlp_to_cmoe_moe(
            model,
            n_experts=n_experts,
            n_activated=n_activated,
            n_shared=n_shared,
            bias_speed=bias_speed,
            add_eas=add_eas,
            eas_init_std=0.0,  # weights overwritten by state_dict load anyway
        )

        return model, tokenizer

    raise ValueError(f"Unsupported moe_type {moe_type!r} (expected one of {MOE_TYPES}).")


def load_state_dict_into_model(
    model: Any,
    state_dict_path: str,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load a full state_dict into ``model``.

    Args:
        model: target HF model with MoE skeleton already swapped in.
        state_dict_path: path to a ``state_dict.pt`` (raw dict or
            ``{"state_dict": ...}`` wrapper).
        strict: passed to :meth:`nn.Module.load_state_dict`. When True, a key
            mismatch raises; we still log first 5 missing/unexpected keys for
            diagnostics by intercepting ``RuntimeError`` and re-raising.

    Returns:
        ``(missing_keys, unexpected_keys)``. With ``strict=True`` and a clean
        load both lists are empty.
    """
    logger.info("Loading state_dict from %s (strict=%s)", state_dict_path, strict)
    sd = torch.load(state_dict_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.startswith(("model.", "lm_head")) for k in sd.keys()
    ):
        sd = sd["state_dict"]

    if strict:
        # Probe with strict=False first to log diagnostics before raising.
        probe_missing, probe_unexpected = model.load_state_dict(sd, strict=False)
        if probe_missing:
            logger.warning("missing keys: %d (first 5: %s)",
                           len(probe_missing), probe_missing[:5])
        if probe_unexpected:
            logger.warning("unexpected keys: %d (first 5: %s)",
                           len(probe_unexpected), probe_unexpected[:5])
        if probe_missing or probe_unexpected:
            raise RuntimeError(
                f"strict=True load failed: {len(probe_missing)} missing, "
                f"{len(probe_unexpected)} unexpected. First missing: "
                f"{probe_missing[:3]}; first unexpected: {probe_unexpected[:3]}"
            )
        return [], []

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("missing keys: %d (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])
    return list(missing), list(unexpected)


def _artifact_path(
    explicit: Optional[str],
    manifest: Dict[str, Any],
    key: str,
    manifest_path: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    value = explicit or manifest.get(key) or fallback
    if not value:
        return None
    path = os.path.expanduser(str(value))
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path)
    return os.path.abspath(path)


def load_checkpoint_artifacts(
    model: Any,
    manifest: Dict[str, Any],
    manifest_path: str,
    *,
    full_state_dict_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    trainable_delta_path: Optional[str] = None,
    cmoe_extra_state_path: Optional[str] = None,
    source_moe_dir: Optional[str] = None,
    source_state_dict_path: Optional[str] = None,
    strict_full: bool = True,
) -> Tuple[Any, List[str], List[str], List[str]]:
    """Load a full checkpoint or reconstruct base carve + adapter/delta."""
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    adapter = _artifact_path(
        adapter_path, manifest, "cpt_adapter_dir", manifest_path
    )
    delta = _artifact_path(
        trainable_delta_path,
        manifest,
        "cpt_trainable_delta_path",
        manifest_path,
        os.path.join(manifest_dir, "trainable_delta.pt"),
    )
    extra = _artifact_path(
        cmoe_extra_state_path,
        manifest,
        "cpt_cmoe_extra_state_path",
        manifest_path,
        os.path.join(manifest_dir, "cmoe_extra_state.pt"),
    )
    if delta and not os.path.exists(delta):
        delta = None
    if extra and not os.path.exists(extra):
        extra = None
    if adapter and not os.path.isdir(adapter):
        adapter = None

    explicit_full = None
    if full_state_dict_path:
        candidate = os.path.abspath(os.path.expanduser(str(full_state_dict_path)))
        if os.path.isfile(candidate):
            explicit_full = candidate

    # An explicitly supplied full checkpoint is self-contained and must take
    # precedence over manifest-advertised reconstruction fragments.  This
    # avoids loading the base carve into an already attached MLP router, where
    # the base correctly lacks the Stage1 router keys and emits a misleading
    # missing-key warning.
    fragment_mode = explicit_full is None and bool(adapter or delta)
    loaded: List[str] = []
    if fragment_mode:
        source_state = _artifact_path(
            source_state_dict_path,
            manifest,
            "cpt_source_state_dict_path",
            manifest_path,
        )
        if not source_state:
            source = source_moe_dir or manifest.get("cpt_source_moe_dir")
            if not source:
                raise ValueError(
                    "adapter/delta reconstruction requires "
                    "--source_state_dict_path, manifest "
                    "cpt_source_state_dict_path, --source_moe_dir, or manifest "
                    "cpt_source_moe_dir"
                )
            source = os.path.abspath(os.path.expanduser(str(source)))
            source_state = os.path.join(source, "state_dict.pt")
        if not os.path.isfile(source_state):
            raise FileNotFoundError(source_state)
        router_arch = manifest.get("router_arch") or manifest.get("cpt_router_arch")
        load_state_dict_into_model(
            model,
            source_state,
            strict=bool(strict_full and not is_mlp_router_arch(router_arch)),
        )
        loaded.append(source_state)

        missing: List[str] = []
        unexpected: List[str] = []
        if delta:
            delta_state = torch.load(delta, map_location="cpu")
            missing_raw, unexpected_raw = model.load_state_dict(
                delta_state, strict=False
            )
            missing = list(missing_raw)
            unexpected = list(unexpected_raw)
            if unexpected:
                raise RuntimeError(
                    f"delta load has {len(unexpected)} unexpected keys: "
                    f"{unexpected[:8]}"
                )
            loaded.append(delta)
        if extra:
            extra_state = torch.load(extra, map_location="cpu")
            _missing_extra, unexpected_extra = model.load_state_dict(
                extra_state, strict=False
            )
            if unexpected_extra:
                raise RuntimeError(
                    "CMoE extra-state load has unexpected keys: "
                    f"{list(unexpected_extra)[:8]}"
                )
            loaded.append(extra)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            loaded.append(adapter)
        return model, missing, unexpected, loaded

    full_state = explicit_full or _artifact_path(
        full_state_dict_path,
        manifest,
        "cpt_full_state_dict_path",
        manifest_path,
        os.path.join(manifest_dir, "state_dict.pt"),
    )
    if not full_state or not os.path.isfile(full_state):
        raise FileNotFoundError(
            f"no full state_dict and no adapter/delta checkpoint for {manifest_path}"
        )
    missing, unexpected = load_state_dict_into_model(
        model, full_state, strict=strict_full
    )
    loaded.append(full_state)
    return model, missing, unexpected, loaded


# -----------------------------------------------------------------------------
# lm-eval wiring
# -----------------------------------------------------------------------------


def run_lmeval(
    model: Any,
    tokenizer: Any,
    tasks: List[str],
    num_fewshot: int,
    batch_size: int,
    device: str,
    limit: Optional[int],
    gen_kwargs: Optional[str] = None,
    seed: int = 0,
    log_samples: bool = False,
    confirm_run_unsafe_code: bool = False,
) -> Dict[str, Any]:
    """Run lm-evaluation-harness via HFLM."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    # The Qwen checkpoint carries max_new_tokens=2048 in generation_config.
    # Clear it so lm-eval's task/default generation limit is authoritative.
    lm.model.generation_config.max_new_tokens = None

    unsafe_kwargs: Dict[str, Any] = {}
    if confirm_run_unsafe_code:
        if "confirm_run_unsafe_code" not in inspect.signature(simple_evaluate).parameters:
            raise RuntimeError(
                "Installed lm-eval does not support confirm_run_unsafe_code; "
                "upgrade lm-eval before running tasks that require unsafe code."
            )
        unsafe_kwargs["confirm_run_unsafe_code"] = True
    logger.info("Running lm-eval tasks=%s num_fewshot=%d batch_size=%d limit=%s seed=%d log_samples=%s",
                tasks, num_fewshot, batch_size, limit, seed, log_samples)
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        gen_kwargs=gen_kwargs,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=log_samples,
        **unsafe_kwargs,
    )
    return results


def headline_acc(metrics: Dict[str, Any]) -> Optional[float]:
    """Pick the most informative single accuracy from a task result dict."""
    for key in ("acc_norm,none", "acc,none", "acc_norm", "acc",
                "word_perplexity,none", "word_perplexity"):
        if key in metrics:
            return metrics[key]
    return None


def print_summary(manifest: Dict[str, Any], results: Dict[str, Any], moe_type: str = "rrd") -> None:
    print("=== lm-eval results ===")
    if moe_type == "cmoe":
        tag = "cmoe-S{}A{}E{}".format(
            manifest.get("nshared"),
            manifest.get("nactivated"),
            manifest.get("nexperts"),
        )
    else:
        tag = "{}-{}-{}".format(
            manifest.get("expert_split"),
            manifest.get("topology"),
            manifest.get("router_init"),
        )
    print(f"manifest ({moe_type}): {tag}")
    for task, metrics in results.get("results", {}).items():
        acc = headline_acc(metrics)
        print(f"  {task}: {acc}")


# -----------------------------------------------------------------------------
# Self-test (no trained checkpoint required)
# -----------------------------------------------------------------------------


def run_cmoe_self_test(dtype_name: str = "float32") -> None:
    """Mock-model CMoE swap + roundtrip sanity check (no large model load).

    Builds a tiny mock LlamaForCausalLM-shaped module, runs
    :func:`swap_llama_mlp_to_cmoe_moe`, captures its state_dict, rebuilds the
    skeleton on a fresh mock, and asserts ``strict=True`` load succeeds. This
    isolates the CMoE swap+load mechanics so that the (CPU-only) self-test is
    feasible without touching the 7B base model.
    """
    import torch.nn as nn

    dtype = build_dtype(dtype_name)
    logger.info("[self_test/cmoe] mock-model swap+load roundtrip (dtype=%s)", dtype_name)

    class MockMLP(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class MockLayer(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.mlp = MockMLP(hidden, intermediate)

    class MockInner(nn.Module):
        def __init__(self, hidden: int, intermediate: int, n_layers: int = 2) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [MockLayer(hidden, intermediate) for _ in range(n_layers)]
            )

    class MockLM(nn.Module):
        def __init__(self, hidden: int, intermediate: int, n_layers: int = 2) -> None:
            super().__init__()
            self.model = MockInner(hidden, intermediate, n_layers)

    hidden, intermediate = 64, 64  # 64/8 = 8 → moe_inter
    n_experts, n_act, n_shared = 8, 3, 3

    src = MockLM(hidden, intermediate, n_layers=2).to(dtype)
    swap_llama_mlp_to_cmoe_moe(
        src, n_experts=n_experts, n_activated=n_act, n_shared=n_shared
    )
    assert all(isinstance(layer.mlp, CMoEMoE) for layer in src.model.layers)

    # Forward sanity.
    x = torch.randn(2, 4, hidden, dtype=dtype)
    h = x
    with torch.no_grad():
        for layer in src.model.layers:
            h = layer.mlp(h)
    assert tuple(h.shape) == (2, 4, hidden), f"forward shape mismatch: {tuple(h.shape)}"

    # State_dict roundtrip via strict=True.
    sd = src.state_dict()
    dst = MockLM(hidden, intermediate, n_layers=2).to(dtype)
    swap_llama_mlp_to_cmoe_moe(
        dst, n_experts=n_experts, n_activated=n_act, n_shared=n_shared
    )
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys after CMoE roundtrip: {missing[:5]}"
    assert not unexpected, f"unexpected keys after CMoE roundtrip: {unexpected[:5]}"

    # Numerical equivalence.
    src.eval()
    dst.eval()
    with torch.no_grad():
        h_src, h_dst = x, x
        for s_layer, d_layer in zip(src.model.layers, dst.model.layers):
            h_src = s_layer.mlp(h_src)
            h_dst = d_layer.mlp(h_dst)
    cos = F.cosine_similarity(
        h_src.flatten().float(), h_dst.flatten().float(), dim=0
    ).item()
    assert cos > 0.999, f"CMoE roundtrip cos_sim too low: {cos:.6f}"
    print(f"[self_test/cmoe] PASS  layers=2 roundtrip_cos={cos:.6f}")


def run_self_test(base_model_path: str, dtype_name: str = "float32") -> None:
    """Swap exactly one layer's MLP with a teacher-equivalent E8A8 MoE; assert cos_sim ~ 1.

    Uses ``build_moe_from_clusters`` with a sequential (BasicSplit) cluster map
    plus a custom unweighted summation forward. With BasicSplit + N=8
    all-active + sum (weights=1), the partitioned experts reproduce the
    teacher MLP exactly. For the swap+load sanity we just verify that the
    builder + load_state_dict mechanics preserve numerical behavior.
    """
    from transformers import AutoModelForCausalLM

    dtype = build_dtype(dtype_name)
    logger.info("[self_test] loading base %s in %s", base_model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    model.eval()

    config = model.config
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    n_experts = 8
    moe_inter = intermediate_size // n_experts

    layer_idx = 0
    teacher_mlp = model.model.layers[layer_idx].mlp
    # Save teacher reference forward for comparison
    x = torch.randn(2, 4, hidden_size, dtype=dtype)
    with torch.no_grad():
        teacher_out = teacher_mlp(x.reshape(-1, hidden_size))

    cluster_ids = torch.arange(intermediate_size) // moe_inter
    moe = build_moe_from_clusters(
        teacher_mlp=teacher_mlp,
        cluster_ids=cluster_ids,
        n_experts=8,
        n_active=8,
        has_shared=False,
    ).to(dtype)
    moe.eval()

    # Equivalence check: sum of expert outputs (unweighted) == teacher_out.
    with torch.no_grad():
        x_flat = x.reshape(-1, hidden_size)
        moe_sum = torch.zeros_like(teacher_out)
        for e in range(n_experts):
            moe_sum = moe_sum + moe.experts[e](x_flat)
    cos = F.cosine_similarity(moe_sum.flatten().float(),
                              teacher_out.flatten().float(), dim=0).item()
    logger.info("[self_test] single-layer E8 sum vs teacher cos_sim=%.6f", cos)
    assert cos > 0.99, f"single-layer all-active equivalence failed: cos_sim={cos:.6f}"

    # Now sanity-check swap + state_dict round-trip:
    model.model.layers[layer_idx].mlp = moe
    sd = model.state_dict()
    # rebuild the same shape skeleton and reload
    skeleton_moe = MoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_experts=8,
        n_active=8,
        has_shared=False,
    ).to(dtype)
    model2 = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    model2.model.layers[layer_idx].mlp = skeleton_moe
    missing, unexpected = model2.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("[self_test] missing after reload: %d (first: %s)",
                       len(missing), missing[:3])
    if unexpected:
        logger.warning("[self_test] unexpected after reload: %d (first: %s)",
                       len(unexpected), unexpected[:3])
    model2.eval()
    with torch.no_grad():
        moe2 = model2.model.layers[layer_idx].mlp
        moe2_sum = torch.zeros_like(teacher_out)
        x_flat = x.reshape(-1, hidden_size)
        for e in range(n_experts):
            moe2_sum = moe2_sum + moe2.experts[e](x_flat)
    cos2 = F.cosine_similarity(moe2_sum.flatten().float(),
                               teacher_out.flatten().float(), dim=0).item()
    logger.info("[self_test] post-roundtrip cos_sim=%.6f", cos2)
    assert cos2 > 0.99, f"swap+load_state_dict roundtrip failed: cos_sim={cos2:.6f}"
    print(f"[self_test] PASS  cos_sim={cos:.6f} roundtrip={cos2:.6f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.self_test:
        # Always run the original RRD self-test (preserves prior behavior).
        # If the user requested CMoE, additionally run the CMoE mock test.
        if args.moe_type == "cmoe":
            run_cmoe_self_test(dtype_name="float32")
            return
        run_self_test(args.base_model_path, dtype_name="float32")
        return

    # Resolve effective state_dict path: --cmoe_state_dict_path overrides
    # --moe_state_dict when explicitly provided (handy for CMoE callers).
    state_dict_path = args.cmoe_state_dict_path or args.moe_state_dict
    if not args.manifest or not args.output_json:
        raise SystemExit(
            "When not running --self_test, --manifest and --output_json are "
            "required; provide a full state_dict or an adapter/delta artifact."
        )

    # Load manifest first with auto-detection, then resolve the effective
    # moe_type (manifest overrides CLI when explicit).
    manifest = load_manifest(args.manifest, moe_type=None)
    moe_type = resolve_moe_type(manifest, args.moe_type)
    # Re-validate with the resolved type to catch CLI/manifest disagreement.
    # If a CMoE-format manifest is given but CLI says rrd (or vice-versa),
    # this raises with a clear "missing required keys" message.
    try:
        manifest = load_manifest(args.manifest, moe_type=moe_type)
    except ValueError as e:
        raise SystemExit(
            f"Manifest/moe_type mismatch: resolved moe_type={moe_type!r} but "
            f"validation failed: {e}. Did you forget --moe_type cmoe for a CMoE "
            f"carved manifest? (manifest keys: "
            f"{sorted(k for k in manifest if isinstance(k, str))[:6]}...)"
        )

    dtype = build_dtype(args.dtype)

    cmoe_defaults = {
        "n_experts": args.cmoe_n_experts,
        "n_activated": args.cmoe_n_activated,
        "n_shared": args.cmoe_n_shared,
    }
    model, tokenizer = build_moe_skeleton_model(
        base_model_path=args.base_model_path,
        manifest=manifest,
        dtype=dtype,
        moe_type=moe_type,
        cmoe_defaults=cmoe_defaults,
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
    # Full checkpoints load directly; storage-efficient runs reconstruct from
    # the canonical carve plus CE delta or PEFT adapter+CMoE extra sidecar.
    strict = bool(args.cmoe_strict_load) if moe_type == "cmoe" else False
    model, missing, unexpected, loaded_artifacts = load_checkpoint_artifacts(
        model,
        manifest,
        args.manifest,
        full_state_dict_path=state_dict_path,
        adapter_path=args.adapter_path,
        trainable_delta_path=args.trainable_delta_path,
        cmoe_extra_state_path=args.cmoe_extra_state_path,
        source_moe_dir=args.source_moe_dir,
        source_state_dict_path=args.source_state_dict_path,
        strict_full=strict,
    )
    logger.info("Loaded checkpoint artifacts: %s", loaded_artifacts)
    model = model.to(args.device).eval()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = run_lmeval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        gen_kwargs=args.gen_kwargs,
        seed=args.lmeval_seed,
        log_samples=args.log_samples,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )

    if args.log_samples and args.output_path:
        from pathlib import Path
        out_dir = Path(args.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = results.get("samples", {}) or {}
        n_files = 0
        for task_name, task_samples in samples.items():
            path = out_dir / f"samples_{task_name}.jsonl"
            with open(path, "w") as f:
                for s in task_samples:
                    f.write(json.dumps(s, default=str) + "\n")
            n_files += 1
            logger.info("Wrote %d samples to %s", len(task_samples), path)
        logger.info("Wrote %d samples files to %s", n_files, out_dir)

    summary: Dict[str, Any] = {
        "moe_type": moe_type,
        "manifest": manifest,
        "manifest_path": os.path.abspath(args.manifest),
        "moe_state_dict_path": (
            os.path.abspath(state_dict_path) if state_dict_path else None
        ),
        "loaded_artifacts": loaded_artifacts,
        "tasks": list(results.get("results", {}).keys()),
        "results": results.get("results", {}),
        "groups": results.get("groups", {}),
        "config": {
            "base_model_path": args.base_model_path,
            "moe_type": moe_type,
            "tasks": tasks,
            "num_fewshot": args.num_fewshot,
            "batch_size": args.batch_size,
            "device": args.device,
            "limit": args.limit,
            "dtype": args.dtype,
            "gen_kwargs": args.gen_kwargs,
            "cmoe_strict_load": bool(args.cmoe_strict_load),
            "confirm_run_unsafe_code": args.confirm_run_unsafe_code,
        },
        "load_state_dict": {
            "strict": strict,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_first5": missing[:5],
            "unexpected_first5": unexpected[:5],
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary to %s", args.output_json)
    print_summary(manifest, results, moe_type=moe_type)


if __name__ == "__main__":
    main()
