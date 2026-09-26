#!/usr/bin/env python3
"""Resumable C4-4M runner for RRD with final-logit distillation.

It uses CE + router CE + 2 * representation RMSE + logit KD on a sequential
C4 memory-mapped stream and saves an immutable 4M checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

from scripts.exp_cmoe import rrd_1_stage_2607 as base  # noqa: E402
from scripts.exp_cmoe import rrd_contract as rrd2  # noqa: E402
from scripts.exp_cmoe.cpt_train import (  # noqa: E402
    NpyTokenIdsChunk,
    _collate,
    _trainable_delta_state,
)

STRATEGY = "rrd_logit_kd_c4_4m"
SOURCE_STRATEGY = "rrd_1_stage_2607"
SEQLEN = 2048
BATCH_SIZE = 2
DEFAULT_WINDOWS = 2_048
DEFAULT_SAVE_STEPS = (1_024,)
DEFAULT_VALIDATION_EVERY = 100
DEFAULT_VALIDATION_BATCHES = 32


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_steps(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if any(step <= 0 for step in result):
        raise ValueError("save steps must be positive")
    return result


def make_contract(args: argparse.Namespace) -> rrd2.Contract:
    teacher = args.teacher.expanduser().resolve()
    moe_dir = args.moe_dir.expanduser().resolve()
    train = args.train_npy.expanduser().resolve()
    validation = args.validation_npy.expanduser().resolve()
    data_manifest = args.data_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    n_layers, hidden_size = rrd2.model_config(teacher)
    return rrd2.Contract(
        model_key=args.model,
        model_hf_id={"qwen2_5_7b": "Qwen/Qwen2.5-7B", "llama2_7b": "meta-llama/Llama-2-7b-hf"}[args.model],
        topology=rrd2.parse_topology(args.topology),
        teacher=teacher,
        moe_dir=moe_dir,
        train=train,
        valtest=validation,
        data_manifest=data_manifest,
        output_root=output,
        seed=int(args.seed),
        n_layers=n_layers,
        hidden_size=hidden_size,
        middle_layers=rrd2.middle_third(n_layers),
    )


def audit_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = make_contract(args)
    required = (
        contract.teacher / "config.json",
        contract.moe_dir / "state_dict.pt",
        contract.moe_dir / "manifest.json",
        contract.moe_dir / ".ready",
        contract.train,
        contract.valtest,
        contract.data_manifest,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"ok": False, "missing": missing}

    data = load_json(contract.data_manifest)
    carve = load_json(contract.moe_dir / "manifest.json")
    files = data.get("files", {})
    train_row = files.get("train_master", {})
    validation_row = files.get("c4_dev", {})
    calibration_row = files.get("calibration", {})
    train = np.load(contract.train, mmap_mode="r")
    validation = np.load(contract.valtest, mmap_mode="r")
    checks = {
        "data_schema": data.get("schema") == "c4_budget_scaling_exact_tokens_v1",
        "data_seed": int(data.get("seed", -1)) == int(args.seed),
        "data_seqlen": int(data.get("seqlen", -1)) == SEQLEN,
        "data_overlap_zero": int(data.get("raw_train_validation_text_overlap", -1)) == 0,
        "train_path": Path(str(train_row.get("path", ""))).resolve() == contract.train,
        "validation_path": (
            Path(str(validation_row.get("path", ""))).resolve() == contract.valtest
        ),
        "train_shape": list(train.shape) == [int(args.total_windows), SEQLEN],
        "train_dtype": train.dtype == np.dtype("int32"),
        "validation_shape": (
            validation.ndim == 2 and int(validation.shape[1]) == SEQLEN
        ),
        "carve_topology": (
            int(carve.get("nshared", -1)) == contract.topology.shared
            and int(carve.get("nactivated", -1)) == contract.topology.active
            and int(carve.get("nexperts", -1)) == contract.topology.total
        ),
        "carve_seed": int(carve.get("seed", -1)) == int(args.seed),
        "carve_seqlen": int(carve.get("seqlen", -1)) == SEQLEN,
        "carve_calibration_exact": (
            carve.get("calibration_format") == "exact_token_ids_jsonl"
            and not bool(carve.get("calibration_text_roundtrip_used", True))
        ),
        "carve_calibration_matches_bundle": (
            carve.get("dataset_sha256") == calibration_row.get("file_sha256")
        ),
        "total_windows_even": int(args.total_windows) % BATCH_SIZE == 0,
    }
    return {
        "ok": all(checks.values()),
        "strategy": STRATEGY,
        "source_strategy": SOURCE_STRATEGY,
        "contract": {
            "teacher": str(contract.teacher),
            "moe_dir": str(contract.moe_dir),
            "train_npy": str(contract.train),
            "validation_npy": str(contract.valtest),
            "data_manifest": str(contract.data_manifest),
            "output_dir": str(contract.output_root),
            "total_windows": int(args.total_windows),
            "total_tokens": int(args.total_windows) * SEQLEN,
            "total_steps": int(args.total_windows) // BATCH_SIZE,
            "save_steps": list(parse_steps(args.save_at_steps)),
            "seed": int(args.seed),
        },
        "train_token_stream_sha256": train_row.get("token_stream_sha256"),
        "validation_token_stream_sha256": validation_row.get("token_stream_sha256"),
        "calibration_token_stream_sha256": calibration_row.get("token_stream_sha256"),
        "checks": checks,
    }


def resume_contract_sha256(
    args: argparse.Namespace,
    contract: rrd2.Contract,
) -> str:
    payload = {
        "schema": "rrd_logit_kd_c4_4m_resume_v1",
        "source_strategy": SOURCE_STRATEGY,
        "teacher": str(contract.teacher),
        "moe_dir": str(contract.moe_dir),
        "moe_manifest_sha256": sha256_file(contract.moe_dir / "manifest.json"),
        "train": str(contract.train),
        "validation": str(contract.valtest),
        "data_manifest": str(contract.data_manifest),
        "data_manifest_sha256": sha256_file(contract.data_manifest),
        "seed": int(args.seed),
        "total_windows": int(args.total_windows),
        "batch_size": BATCH_SIZE,
        "seqlen": SEQLEN,
        "router_base_lr": float(args.router_base_lr),
        "expert_lr": float(base.EXPERT_LR),
        "router_outer_lr": float(base.ROUTER_OUTER_LR),
        "router_middle_lr": float(base.ROUTER_MIDDLE_LR),
        "loss_weights": [base.CE_WEIGHT, base.ROUTER_WEIGHT, base.JOINT_RRD_WEIGHT, base.KD_WEIGHT],
        "kd_temperature": base.KD_TEMPERATURE,
        "kd_chunk_tokens": base.KD_CHUNK_TOKENS,
        "optimizer": "bitsandbytes.Adam8bit",
        "optimizer_betas": [0.9, 0.95],
        "lr_schedule": "constant",
        "data_order": "sequential_exact_prefix",
        "research_git_commit": os.environ.get("RESEARCH_GIT_COMMIT", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_resume(
    *,
    path: Path,
    contract_sha256: str,
    completed_step: int,
    student: Any,
    optimizer: Any,
    train_history: list[dict[str, Any]],
    validation_history: list[dict[str, Any]],
    first_audit: dict[str, Any],
    elapsed_sec: float,
) -> None:
    payload = {
        "schema": "rrd_logit_kd_c4_4m_resume_v1",
        "contract_sha256": contract_sha256,
        "completed_step": int(completed_step),
        "trainable_state": {
            name: parameter.detach().cpu()
            for name, parameter in student.named_parameters()
            if parameter.requires_grad
        },
        "optimizer_state": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "train_history": list(train_history),
        "validation_history": list(validation_history),
        "first_audit": first_audit,
        "elapsed_sec": float(elapsed_sec),
    }
    atomic_torch(path, payload)


def restore_resume(
    *,
    path: Path,
    contract_sha256: str,
    student: Any,
    optimizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "rrd_logit_kd_c4_4m_resume_v1":
        raise ValueError(f"unsupported resume schema: {path}")
    if payload.get("contract_sha256") != contract_sha256:
        raise ValueError("resume contract mismatch; refusing to splice trajectories")
    current = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    saved = payload.get("trainable_state", {})
    if set(current) != set(saved):
        raise ValueError(
            "resume trainable key mismatch: "
            f"missing={sorted(set(current) - set(saved))[:8]} "
            f"unexpected={sorted(set(saved) - set(current))[:8]}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in payload["cuda_rng_state_all"]]
        )
    return payload


def checkpoint_manifest(
    *,
    contract: rrd2.Contract,
    args: argparse.Namespace,
    carve_manifest: dict[str, Any],
    step: int,
    counts: dict[str, int],
    optimizer_audit: dict[str, Any],
    first_audit: dict[str, Any],
    elapsed_sec: float,
    delta_path: Path,
    contract_sha256: str,
    final: bool,
) -> dict[str, Any]:
    return {
        **carve_manifest,
        "strategy": SOURCE_STRATEGY,
        "phase": STRATEGY,
        "run_name": args.output_dir.name,
        "cpt_mode": "one_stage_true_stf_joint_rrd_logit_kd",
        "cpt_source_moe_dir": str(contract.moe_dir),
        "cpt_teacher_model_path": str(contract.teacher),
        "cpt_calib_path": str(contract.train),
        "cpt_validation_calib_path": str(contract.valtest),
        "cpt_data_manifest": str(contract.data_manifest),
        "cpt_data_manifest_sha256": sha256_file(contract.data_manifest),
        "cpt_data_source": "token_ids_npy",
        "cpt_data_order": "sequential_exact_prefix",
        "cpt_seed": int(args.seed),
        "cpt_one_epoch": True,
        "cpt_per_device_bsz": BATCH_SIZE,
        "cpt_grad_accum_steps": 1,
        "cpt_seqlen": SEQLEN,
        "cpt_max_steps": int(step),
        "cpt_total_sequences": int(step) * BATCH_SIZE,
        "cpt_total_tokens": int(step) * BATCH_SIZE * SEQLEN,
        "cpt_optimizer": "bitsandbytes.Adam8bit",
        "cpt_optimizer_betas": [0.9, 0.95],
        "cpt_weight_decay": 0.0,
        "cpt_lr_schedule": "constant",
        "cpt_router_base_lr": float(args.router_base_lr),
        "cpt_expert_lr": float(base.EXPERT_LR),
        "cpt_router_outer_lr": float(base.ROUTER_OUTER_LR),
        "cpt_router_middle_lr": float(base.ROUTER_MIDDLE_LR),
        "cpt_router_middle_layers": list(contract.middle_layers),
        "cpt_optimizer_group_audit": optimizer_audit,
        "cpt_alpha_task": base.CE_WEIGHT,
        "cpt_alpha_router": base.ROUTER_WEIGHT,
        "cpt_alpha_residual": base.JOINT_RRD_WEIGHT,
        "cpt_alpha_kd": base.KD_WEIGHT,
        "cpt_kd_temperature": base.KD_TEMPERATURE,
        "cpt_kd_chunk_tokens": base.KD_CHUNK_TOKENS,
        "cpt_kd_loss_form": "temperature_squared_forward_KL_full_vocab_valid_next_token_mean",
        "cpt_ce_gradient_policy": "shared_and_routed_only",
        "cpt_joint_rrd_gradient_policy": "shared_and_routed_only",
        "cpt_router_gradient_policy": "true_stf_router_only",
        "cpt_shared_target_form": "joint",
        "cpt_residual_loss_form": "fp32_global_rmse",
        "cpt_residual_target": "shared+routed->teacher_dense_mlp",
        "cpt_router_loss_form": "activation_mass_topA_ce",
        "router_contribution_score": "activation_mass_sum",
        "router_target_oracle": "dense_teacher_recovered_cmoe_neuron_mapping",
        "cpt_router_input_source": "student_true_stf_trajectory_mlp_input",
        "cpt_stf_oracle_aggregation": "uniform_sum",
        "cpt_router_aggregation": f"hard top-{contract.topology.active} uniform sum",
        "cpt_mlp_router_aggregation": "uniform",
        "cpt_mlp_router_hidden_size": base.PROBE_HIDDEN,
        "cpt_mlp_router_random_init": True,
        "cpt_router_initialization": "random",
        "cpt_router_arch": f"mlp_h{base.PROBE_HIDDEN}",
        "router_arch": f"mlp_h{base.PROBE_HIDDEN}",
        "hard_inference_preserved": True,
        "cpt_freeze_attention": True,
        "cpt_freeze_lm_head": True,
        "cpt_cmoe_enable_load_balance": False,
        "cpt_router_quality_metric": f"top{contract.topology.active}_hit_count_average",
        "cpt_exact_set_match_reporting": "disabled",
        "cpt_trainable_parameter_counts": counts,
        "gradient_audit": first_audit,
        "cpt_train_log_every": int(args.log_every),
        "cpt_validation_every": int(args.validation_every),
        "cpt_validation_batches": int(args.validation_batches),
        "cpt_elapsed_sec": float(elapsed_sec),
        "cpt_gpu_hours": float(elapsed_sec) / 3600.0,
        "cpt_trainable_delta_path": str(delta_path.resolve()),
        "cpt_save_trainable_delta": True,
        "cpt_skip_full_state_dict": True,
        "cpt_checkpoint_format": "base_plus_trainable_delta",
        "cpt_resume_schema": "rrd_logit_kd_c4_4m_resume_v1",
        "cpt_resume_contract_sha256": contract_sha256,
        "cpt_intermediate_save": not final,
        "checkpoint_status": "saved",
    }


def save_inference_checkpoint(
    *,
    output_dir: Path,
    contract: rrd2.Contract,
    args: argparse.Namespace,
    student: Any,
    carve_manifest: dict[str, Any],
    step: int,
    counts: dict[str, int],
    optimizer_audit: dict[str, Any],
    first_audit: dict[str, Any],
    train_history: list[dict[str, Any]],
    validation_history: list[dict[str, Any]],
    elapsed_sec: float,
    contract_sha256: str,
    final: bool,
) -> None:
    if output_dir.exists():
        if final:
            collisions = [
                name
                for name in (
                    "manifest.json",
                    "cpt_log.json",
                    "trainable_delta.pt",
                    ".ready",
                )
                if (output_dir / name).exists()
            ]
            if collisions:
                raise FileExistsError(
                    f"refusing to overwrite final checkpoint files: {collisions}"
                )
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite checkpoint: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "trainable_delta.pt"
    atomic_torch(delta_path, _trainable_delta_state(student))
    manifest = checkpoint_manifest(
        contract=contract,
        args=args,
        carve_manifest=carve_manifest,
        step=step,
        counts=counts,
        optimizer_audit=optimizer_audit,
        first_audit=first_audit,
        elapsed_sec=elapsed_sec,
        delta_path=delta_path,
        contract_sha256=contract_sha256,
        final=final,
    )
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(
        output_dir / "cpt_log.json",
        {
            "strategy": SOURCE_STRATEGY,
            "max_steps": int(step),
            "elapsed_sec": float(elapsed_sec),
            "log_history": train_history,
            "validation_history": validation_history,
            "intermediate_save": not final,
        },
    )
    for name in (
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ):
        source = contract.moe_dir / name
        if source.is_file():
            shutil.copy(source, output_dir / name)
    atomic_text(output_dir / ".ready", "ready\n")


def validation_batches(
    path: Path,
    count: int,
) -> list[dict[str, torch.Tensor]]:
    dataset = NpyTokenIdsChunk(str(path), SEQLEN, data_split="all")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=_collate,
    )
    return [batch for index, batch in enumerate(loader) if index < count]


def train(args: argparse.Namespace, *, smoke_steps: int = 0) -> dict[str, Any]:
    base.configure_logit_kd(1.0, 1.0, 128)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gate = audit_contract(args)
    if not gate.get("ok"):
        raise RuntimeError(json.dumps(gate, indent=2))

    contract = make_contract(args)
    out = contract.output_root / "smoke" if smoke_steps else contract.output_root
    final_complete = all(
        (out / name).exists()
        for name in ("manifest.json", "cpt_log.json", "trainable_delta.pt", ".ready")
    )
    if final_complete and not smoke_steps:
        return {"status": "already_complete", "output_dir": str(out)}

    resume_path = out / "resume_latest.pt"
    if out.exists():
        unexpected = [
            path.name
            for path in out.iterdir()
            if path.name not in {"researchctl_metadata.json", "resume_latest.pt"}
            and not path.name.startswith("ckpt_step_")
        ]
        if unexpected and not smoke_steps:
            raise RuntimeError(f"partial output requires inspection: {unexpected}")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    base.configure_router_architecture("mlp")
    base.configure_learning_rates(float(args.router_base_lr))

    import scripts.exp_cmoe.rrd_router_targets as p120

    p120.TEACHER = contract.teacher
    student, _tokenizer, carve_manifest = base.load_cmoe_model(
        contract.moe_dir,
        device,
        "bfloat16",
        train=True,
    )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    probe = base.MLPProbeBank(
        contract.n_layers,
        contract.hidden_size,
        base.PROBE_HIDDEN,
        contract.topology.routed_total,
    ).to(device=device, dtype=torch.bfloat16)
    base.attach_mlp_probe_router(student, probe, aggregation="uniform")
    counts = base.set_trainable(student)
    teacher = base.load_teacher(device, "bfloat16")
    activation_map = base.build_activation_mass_neuron_index_cache_from_state_dict(
        teacher,
        str(contract.moe_dir / "state_dict.pt"),
        carve_manifest,
        device,
    )
    student.config.use_cache = False
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(student, "enable_input_require_grads"):
        student.enable_input_require_grads()

    components: dict[int, dict[str, torch.Tensor]] = {}
    base.wrap_student_for_capture(student, components)
    teacher_mlp_outs: dict[int, torch.Tensor] = {}
    teacher_post_ln: dict[int, torch.Tensor] = {}
    handles = base.install_teacher_hooks(
        teacher,
        teacher_mlp_outs,
        teacher_post_ln,
    )
    router_params = base.named_group(student, "router")
    shared_params = base.named_group(student, "shared")
    routed_params = base.named_group(student, "routed")
    expert_params = shared_params + routed_params
    optimizer, optimizer_audit = base.build_optimizer(
        contract,
        router_params,
        expert_params,
    )

    dataset = NpyTokenIdsChunk(str(contract.train), SEQLEN, data_split="all")
    total_steps = (
        int(smoke_steps)
        if smoke_steps
        else int(args.total_windows) // BATCH_SIZE
    )
    val_batches = validation_batches(
        contract.valtest,
        1 if smoke_steps else int(args.validation_batches),
    )
    contract_sha256 = resume_contract_sha256(args, contract)
    completed_step = 0
    train_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    first_audit: dict[str, Any] = {}
    elapsed_before = 0.0
    if resume_path.exists() and not smoke_steps:
        payload = restore_resume(
            path=resume_path,
            contract_sha256=contract_sha256,
            student=student,
            optimizer=optimizer,
            device=device,
        )
        completed_step = int(payload["completed_step"])
        train_history = list(payload.get("train_history", []))
        validation_history = list(payload.get("validation_history", []))
        first_audit = dict(payload.get("first_audit", {}))
        elapsed_before = float(payload.get("elapsed_sec", 0.0))
        if completed_step < 0 or completed_step > total_steps:
            raise ValueError("resume completed step is outside the run contract")

    started = time.time() - elapsed_before
    if completed_step == 0:
        validation_history.append(
            base.evaluate_validation(
                contract,
                student,
                teacher,
                val_batches,
                activation_map,
                carve_manifest,
                teacher_mlp_outs,
                teacher_post_ln,
                components,
                device,
                0,
            )
        )

    save_steps = set(parse_steps(args.save_at_steps))
    for step in range(completed_step + 1, total_steps + 1):
        offset = (step - 1) * BATCH_SIZE
        ids_cpu = torch.stack(
            [
                dataset[offset]["input_ids"],
                dataset[offset + 1]["input_ids"],
            ]
        )
        ids = ids_cpu.to(device)
        labels = ids.clone()
        teacher_logits: dict[str, torch.Tensor] = {}
        targets = base.teacher_targets(
            contract,
            teacher,
            ids,
            None,
            teacher_mlp_outs,
            teacher_post_ln,
            activation_map,
            carve_manifest,
            teacher_logits=teacher_logits,
        )
        optimizer.zero_grad(set_to_none=True)
        components.clear()
        output = student(input_ids=ids, labels=labels, use_cache=False)
        joint = base.joint_rmse(components, teacher_mlp_outs)
        kd = base.logit_kd_loss(
            output.logits,
            teacher_logits["logits"],
            labels,
            temperature=base.KD_TEMPERATURE,
            chunk_tokens=base.KD_CHUNK_TOKENS,
        )
        teacher_logits.clear()
        ce_grads = base.accumulate_loss_grads(
            output.loss,
            expert_params,
            scale=base.CE_WEIGHT,
            retain_graph=True,
        )
        joint_grads = base.accumulate_loss_grads(
            joint,
            expert_params,
            scale=base.JOINT_RRD_WEIGHT,
            retain_graph=True,
        )
        kd_grads = base.accumulate_loss_grads(
            kd,
            expert_params,
            scale=base.KD_WEIGHT,
            retain_graph=False,
        )
        weight_errors = [
            float(
                (
                    comp["topk_weights"].float().sum(dim=-1)
                    - contract.topology.active
                )
                .abs()
                .max()
                .item()
            )
            for comp in components.values()
            if comp.get("topk_weights") is not None
        ]
        with base.TrueSTFRouterPatch(
            student,
            targets,
            teacher_mlp_outs,
            contract.topology.active,
        ) as patch:
            student(input_ids=ids, use_cache=False)
            router = torch.stack(
                [patch.losses[layer] for layer in sorted(patch.losses)]
            ).mean()
            router_grads = base.accumulate_loss_grads(
                router,
                router_params,
                scale=base.ROUTER_WEIGHT,
                retain_graph=False,
            )
        optimizer.step()

        if step == 1:
            first_audit = {
                "ce_grad_norms": ce_grads,
                "joint_rrd_grad_norms": joint_grads,
                "logit_kd_grad_norms": kd_grads,
                "router_grad_norms": router_grads,
                "selected_weight_sum_reference": contract.topology.active,
                "selected_weight_sum_max_deviation_from_active": max(
                    weight_errors,
                    default=0.0,
                ),
            }
        if step == 1 or step % int(args.log_every) == 0 or step == total_steps:
            row = {
                "step": step,
                "loss_ce": float(output.loss.detach()),
                "loss_router": float(router.detach()),
                "loss_joint_rrd": float(joint.detach()),
                "loss_logit_kd": float(kd.detach()),
                "weighted_total": float(
                    output.loss.detach()
                    + router.detach()
                    + base.JOINT_RRD_WEIGHT * joint.detach()
                    + base.KD_WEIGHT * kd.detach()
                ),
                "stf_hit_count": base.mean_layer_metric(
                    patch.layer_metrics,
                    "hit_count",
                ),
                "stf_forced_cos": base.mean_layer_metric(
                    patch.layer_metrics,
                    "cos_forced_dense",
                ),
                "optimizer_group_lrs": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
            }
            train_history.append(row)
            atomic_json(
                out / "progress.json",
                {
                    "state": "training",
                    "step": step,
                    "total_steps": total_steps,
                    "row": row,
                },
            )
            print(json.dumps(row), flush=True)

        if (
            step == total_steps
            or step % int(args.validation_every) == 0
        ):
            validation_history.append(
                base.evaluate_validation(
                    contract,
                    student,
                    teacher,
                    val_batches,
                    activation_map,
                    carve_manifest,
                    teacher_mlp_outs,
                    teacher_post_ln,
                    components,
                    device,
                    step,
                )
            )

        elapsed = time.time() - started
        if not smoke_steps and step in save_steps:
            save_inference_checkpoint(
                output_dir=out / f"ckpt_step_{step}",
                contract=contract,
                args=args,
                student=student,
                carve_manifest=carve_manifest,
                step=step,
                counts=counts,
                optimizer_audit=optimizer_audit,
                first_audit=first_audit,
                train_history=train_history,
                validation_history=validation_history,
                elapsed_sec=elapsed,
                contract_sha256=contract_sha256,
                final=False,
            )
        if not smoke_steps and (
            step == total_steps
            or step in save_steps
            or (
                int(args.resume_save_every) > 0
                and step % int(args.resume_save_every) == 0
            )
        ):
            save_resume(
                path=resume_path,
                contract_sha256=contract_sha256,
                completed_step=step,
                student=student,
                optimizer=optimizer,
                train_history=train_history,
                validation_history=validation_history,
                first_audit=first_audit,
                elapsed_sec=elapsed,
            )

    for handle in handles:
        handle.remove()
    base.unwrap_student_capture(student)
    elapsed = time.time() - started
    if smoke_steps:
        payload = {
            "status": "passed",
            "steps": total_steps,
            "gradient_audit": first_audit,
            "elapsed_sec": elapsed,
        }
        atomic_json(out / "smoke.json", payload)
    else:
        save_inference_checkpoint(
            output_dir=out,
            contract=contract,
            args=args,
            student=student,
            carve_manifest=carve_manifest,
            step=total_steps,
            counts=counts,
            optimizer_audit=optimizer_audit,
            first_audit=first_audit,
            train_history=train_history,
            validation_history=validation_history,
            elapsed_sec=elapsed,
            contract_sha256=contract_sha256,
            final=True,
        )
        atomic_json(
            out / "progress.json",
            {"state": "trained", "step": total_steps, "total_steps": total_steps},
        )

    del student, teacher, probe, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "passed" if smoke_steps else "complete",
        "steps": total_steps,
        "elapsed_sec": elapsed,
        "output_dir": str(out),
    }


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("qwen2_5_7b", "llama2_7b"), default="qwen2_5_7b")
    parser.add_argument("--topology", default="S2A2E8")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--moe-dir", type=Path, required=True)
    parser.add_argument("--train-npy", type=Path, required=True)
    parser.add_argument("--validation-npy", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--save-at-steps",
        default=",".join(str(value) for value in DEFAULT_SAVE_STEPS),
    )
    parser.add_argument("--resume-save-every", type=int, default=1_000)
    parser.add_argument(
        "--router-base-lr",
        type=float,
        default=base.DEFAULT_ROUTER_BASE_LR,
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--validation-every",
        type=int,
        default=DEFAULT_VALIDATION_EVERY,
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=DEFAULT_VALIDATION_BATCHES,
    )
    parser.add_argument("--device", default="cuda:0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "smoke", "train"):
        child = sub.add_parser(name)
        add_common(child)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.configure_learning_rates(float(args.router_base_lr))
    if args.command == "audit":
        payload = audit_contract(args)
    elif args.command == "smoke":
        payload = train(args, smoke_steps=2)
    else:
        payload = train(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload.get("ok", True) and args.command == "audit":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
