"""Shared RRD contracts and evaluation helpers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STRATEGY = "RRD-2-stage-2607"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SEQLEN = 2048
TOTAL_WINDOWS = 2048
STAGE_WINDOWS = 1024
BATCH_SIZE = 2
PROBE_HIDDEN = 1024
STAGE1_OUTER_LR = 5e-4
STAGE1_MIDDLE_LR = 1e-3
STAGE2_LR = 3e-5
STAGE2_RRD_WEIGHT = 2.0
MODEL_ALIASES = {
    "qwen": "qwen2_5_7b",
    "qwen2.5-7b": "qwen2_5_7b",
    "qwen2_5_7b": "qwen2_5_7b",
    "qwen14": "qwen2_5_14b",
    "qwen2.5-14b": "qwen2_5_14b",
    "qwen2_5_14b": "qwen2_5_14b",
    "llama": "llama2_7b",
    "llama2-7b": "llama2_7b",
    "llama2_7b": "llama2_7b",
}
DATA_SLUGS = {
    "qwen2_5_7b": "qwen2_5_7b",
    "llama2_7b": "llama_2_7b_hf",
    "qwen2_5_14b": "qwen2_5_14b",
}
EVAL_TASKS = "piqa,winogrande,arc_easy,arc_challenge,hellaswag"
METRIC_DIRECTIONS = {
    "ppl_wt2_train": "min",
    "ppl_wt2_valtest": "min",
    "ppl_c4": "min",
    "hit_wt2_train": "max",
    "hit_wt2_valtest": "max",
    "hit_c4": "max",
    "cos_wt2_train": "max",
    "cos_wt2_valtest": "max",
    "cos_c4": "max",
    "lmeval_avg": "max",
}


@dataclass(frozen=True)
class Topology:
    shared: int
    active: int
    total: int

    @property
    def routed_total(self) -> int:
        return self.total - self.shared

    @property
    def name(self) -> str:
        return f"S{self.shared}A{self.active}E{self.total}"


@dataclass(frozen=True)
class Contract:
    model_key: str
    model_hf_id: str
    topology: Topology
    teacher: Path
    moe_dir: Path
    train: Path
    valtest: Path
    data_manifest: Path
    output_root: Path
    seed: int
    n_layers: int
    hidden_size: int
    middle_layers: tuple[int, ...]

    @property
    def middle_mask(self) -> str:
        return f"{self.middle_layers[0]}-{self.middle_layers[-1]}"


def parse_topology(value: str) -> Topology:
    match = re.fullmatch(r"S(\d+)A(\d+)E(\d+)", value.strip().upper())
    if not match:
        raise ValueError(f"invalid topology {value!r}; expected SxAyEz")
    topology = Topology(*(int(x) for x in match.groups()))
    if topology.shared < 0 or topology.total <= 0:
        raise ValueError(f"invalid topology: {topology.name}")
    if topology.routed_total <= 0 or topology.active > topology.routed_total:
        raise ValueError(
            f"{topology.name}: active={topology.active} exceeds "
            f"routed_total={topology.routed_total}"
        )
    return topology


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_registry() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError("PyYAML is required to read research/model_registry.yaml") from exc
    path = ROOT / "research/model_registry.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid model registry: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def model_config(teacher: Path) -> tuple[int, int]:
    cfg = load_json(teacher / "config.json")
    n_layers = int(cfg.get("num_hidden_layers", 0))
    hidden_size = int(cfg.get("hidden_size", 0))
    if n_layers <= 0 or hidden_size <= 0:
        raise ValueError(f"invalid model config: {teacher / 'config.json'}")
    return n_layers, hidden_size


def middle_third(n_layers: int) -> tuple[int, ...]:
    start = n_layers // 3
    stop = n_layers - (n_layers // 3)
    return tuple(range(start, stop))


def registry_carve_path(
    registry: dict[str, Any], model_key: str, topology: Topology, server: str
) -> Path:
    matches = []
    for item in registry.get("carve_targets", []):
        topo = item.get("topology", {})
        if item.get("dense_model") != model_key:
            continue
        if (
            int(topo.get("shared", -1)) == topology.shared
            and int(topo.get("routed_active", -1)) == topology.active
            and int(topo.get("total", -1)) == topology.total
        ):
            matches.append(item)
    # Several calibration studies can preserve ready carves for one topology.
    # An explicit default selects the canonical initialization independently of
    # registry order or which experiment-specific resource exists locally.
    defaults = [item for item in matches if item.get("default_for_rrd_2607") is True]
    if defaults:
        matches = defaults
    if len(matches) != 1:
        raise ValueError(
            f"registry must select exactly one default {model_key}/{topology.name} carve; "
            f"found {len(matches)}; disambiguate with default_for_rrd_2607 or --moe-dir"
        )
    item = matches[0]
    if item.get("status") != "ready":
        raise ValueError(f"registry carve is not ready: {item.get('id')} status={item.get('status')}")
    availability = item.get("availability", {})
    if server != "auto":
        candidate = availability.get(server)
        if not isinstance(candidate, str) or candidate in {"missing", "ready"}:
            raise FileNotFoundError(
                f"{item.get('id')} is unavailable on {server}: {candidate!r}"
            )
        return Path(candidate)
    local = [
        Path(value)
        for value in availability.values()
        if isinstance(value, str) and value not in {"missing", "ready"} and Path(value).exists()
    ]
    if not local:
        raise FileNotFoundError(
            f"no local registry path exists for {item.get('id')}; use the canonical owner server"
        )
    return local[0]


def resolve_contract(args: argparse.Namespace) -> Contract:
    model_key = MODEL_ALIASES.get(args.model.lower())
    if model_key is None:
        raise ValueError(f"unknown model alias: {args.model!r}")
    topology = parse_topology(args.topology)
    registry = load_registry()
    dense = registry.get("dense_models", {}).get(model_key)
    if not isinstance(dense, dict):
        raise ValueError(f"model missing from registry: {model_key}")

    teacher = Path(args.teacher).expanduser() if args.teacher else Path(dense["canonical_path"])
    moe_dir = (
        Path(args.moe_dir).expanduser()
        if args.moe_dir
        else registry_carve_path(registry, model_key, topology, args.server)
    )
    seeded_root = ROOT / "data/cmoe_wt2/seeded_paragraph_exact" / DATA_SLUGS[model_key] / f"seed{args.seed}"
    master_root = ROOT / "data/cmoe_wt2/phase145_master" / DATA_SLUGS[model_key] / f"seed{args.seed}"
    candidates = (
        [seeded_root, master_root]
        if model_key.startswith("qwen2_5_")
        else [master_root, seeded_root]
    )
    train_name = f"train_seed{args.seed}_n2048_seqlen2048.token_ids.jsonl"
    data_root = next((path for path in candidates if (path / train_name).exists()), candidates[0])
    train = Path(args.train).expanduser() if args.train else data_root / train_name
    valtest = Path(args.valtest).expanduser() if args.valtest else data_root / "validation_test_seqlen2048.token_ids.jsonl"
    data_manifest = Path(args.data_manifest).expanduser() if args.data_manifest else data_root / "manifest.json"
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else ROOT / "saves/rrd_2_stage_2607" / f"{model_key}_{topology.name.lower()}_seed{args.seed}"
    )
    n_layers, hidden_size = model_config(teacher)
    return Contract(
        model_key=model_key,
        model_hf_id=str(dense["hf_id"]),
        topology=topology,
        teacher=teacher.resolve(),
        moe_dir=moe_dir.resolve(),
        train=train.resolve(),
        valtest=valtest.resolve(),
        data_manifest=data_manifest.resolve(),
        output_root=output_root.resolve(),
        seed=int(args.seed),
        n_layers=n_layers,
        hidden_size=hidden_size,
        middle_layers=middle_third(n_layers),
    )


def audit_contract(contract: Contract) -> dict[str, Any]:
    required = [
        contract.teacher / "config.json",
        contract.moe_dir / "state_dict.pt",
        contract.moe_dir / "manifest.json",
        contract.moe_dir / ".ready",
        contract.train,
        contract.valtest,
        contract.data_manifest,
        ROOT / "scripts/exp_cmoe/cpt_train.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing strategy inputs: {missing}")

    carve = load_json(contract.moe_dir / "manifest.json")
    actual = Topology(
        shared=int(carve.get("nshared", -1)),
        active=int(carve.get("nactivated", -1)),
        total=int(carve.get("nexperts", -1)),
    )
    if actual != contract.topology:
        raise ValueError(f"topology mismatch: requested={contract.topology.name}, carve={actual.name}")
    if carve.get("calibration_format") != "exact_token_ids_jsonl":
        raise ValueError("RRD-2-stage-2607 requires an exact-token paragraph-calibrated carve")
    if bool(carve.get("calibration_text_roundtrip_used", True)):
        raise ValueError("carve used a forbidden calibration text round trip")
    if int(carve.get("seed", -1)) != contract.seed:
        raise ValueError(f"carve seed mismatch: {carve.get('seed')} != {contract.seed}")
    if int(carve.get("seqlen", -1)) != SEQLEN:
        raise ValueError(f"carve seqlen mismatch: {carve.get('seqlen')}")

    data_manifest = load_json(contract.data_manifest)
    if int(data_manifest.get("seed", -1)) != contract.seed:
        raise ValueError("training data seed does not match --seed")
    if int(data_manifest.get("seqlen", -1)) != SEQLEN:
        raise ValueError("training data manifest does not use seqlen=2048")

    rows = [line for line in contract.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != TOTAL_WINDOWS:
        raise ValueError(f"training file must contain exactly {TOTAL_WINDOWS} rows, got {len(rows)}")
    invalid_lengths = []
    for idx, line in enumerate(rows):
        payload = json.loads(line)
        if len(payload.get("input_ids", [])) != SEQLEN:
            invalid_lengths.append(idx)
            if len(invalid_lengths) >= 8:
                break
    if invalid_lengths:
        raise ValueError(f"non-{SEQLEN}-token training rows: {invalid_lengths}")

    return {
        "strategy": STRATEGY,
        "model_key": contract.model_key,
        "model_hf_id": contract.model_hf_id,
        "topology": contract.topology.name,
        "shared_experts": contract.topology.shared,
        "active_routed_experts": contract.topology.active,
        "total_experts": contract.topology.total,
        "routed_expert_domain": contract.topology.routed_total,
        "teacher": str(contract.teacher),
        "moe_dir": str(contract.moe_dir),
        "carve_manifest": str((contract.moe_dir / "manifest.json").resolve()),
        "carve_calibration_samples": int(carve.get("calib_samples", -1)),
        "carve_calibration_sha256": carve.get("dataset_sha256"),
        "train": str(contract.train),
        "train_sha256": sha256_file(contract.train),
        "validation_test": str(contract.valtest),
        "data_manifest": str(contract.data_manifest),
        "seed": contract.seed,
        "num_hidden_layers": contract.n_layers,
        "hidden_size": contract.hidden_size,
        "middle_third_rule": "[num_layers//3, num_layers-num_layers//3)",
        "middle_layer_mask": contract.middle_mask,
        "data": {
            "total_windows": TOTAL_WINDOWS,
            "stage1_windows": STAGE_WINDOWS,
            "stage2_windows": STAGE_WINDOWS,
            "sequence_length": SEQLEN,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": 1,
            "stage1_steps": STAGE_WINDOWS // BATCH_SIZE,
            "stage2_steps": STAGE_WINDOWS // BATCH_SIZE,
            "total_steps": TOTAL_WINDOWS // BATCH_SIZE,
            "token_exposure": TOTAL_WINDOWS * SEQLEN,
            "baseline_multiplier": 1.0,
        },
        "stage1": {
            "objective": "True-STF activation-mass top-A cross entropy",
            "router": f"hidden->{PROBE_HIDDEN}->{contract.topology.routed_total}",
            "target_topk": contract.topology.active,
            "trainable": ["MLP router"],
            "frozen": ["shared experts", "routed experts", "attention", "lm_head"],
            "optimizer": "bitsandbytes.Adam8bit",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "schedule": "constant",
            "outer_lr": STAGE1_OUTER_LR,
            "middle_lr": STAGE1_MIDDLE_LR,
        },
        "stage2": {
            "objective": "1.0*CE + 2.0*joint_RRD_RMSE",
            "joint_rrd": "RMSE(shared_out + routed_out, dense_teacher_mlp_out)",
            "kd_weight": 0.0,
            "router_loss_weight": 0.0,
            "trainable": ["shared experts", "routed experts"],
            "frozen": ["MLP router", "attention", "lm_head"],
            "optimizer": "bitsandbytes.Adam8bit",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "schedule": "constant",
            "lr": STAGE2_LR,
            "aggregation": f"hard top-{contract.topology.active} uniform sum",
        },
        "router_quality_reporting": "top-2 hit-count only; exact-set reporting disabled",
    }


def split_payload(contract: Contract) -> dict[str, Any]:
    indices = list(range(TOTAL_WINDOWS))
    random.Random(contract.seed).shuffle(indices)
    stage1 = indices[:STAGE_WINDOWS]
    stage2 = indices[STAGE_WINDOWS:]
    return {
        "protocol": "seeded_permutation_disjoint_halves",
        "source": str(contract.train),
        "source_sha256": sha256_file(contract.train),
        "seed": contract.seed,
        "sequence_length": SEQLEN,
        "stage1_indices": stage1,
        "stage2_indices": stage2,
        "stage1_indices_sha256": hashlib.sha256(json.dumps(stage1, separators=(",", ":")).encode()).hexdigest(),
        "stage2_indices_sha256": hashlib.sha256(json.dumps(stage2, separators=(",", ":")).encode()).hexdigest(),
        "overlap": len(set(stage1) & set(stage2)),
        "union": len(set(stage1) | set(stage2)),
    }


def ensure_split(contract: Contract, root: Path) -> tuple[Path, Path, dict[str, Any]]:
    payload = split_payload(contract)
    if payload["overlap"] != 0 or payload["union"] != TOTAL_WINDOWS:
        raise RuntimeError(f"invalid Stage1/Stage2 split: {payload}")
    split_root = root / "data_split"
    split_manifest = split_root / "split_manifest.json"
    if split_manifest.exists() and load_json(split_manifest) != payload:
        raise RuntimeError(f"existing split contract differs: {split_manifest}")
    lines = [line for line in contract.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage1_path = split_root / "stage1_n1024.token_ids.jsonl"
    stage2_path = split_root / "stage2_n1024.token_ids.jsonl"
    for path, key in ((stage1_path, "stage1_indices"), (stage2_path, "stage2_indices")):
        expected = "\n".join(lines[idx] for idx in payload[key]) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"existing split data differs: {path}")
        if not path.exists():
            atomic_write_text(path, expected)
    if not split_manifest.exists():
        atomic_json(split_manifest, payload)
    return stage1_path, stage2_path, payload



def ppl_command(contract: Contract, run_dir: Path, kind: str, device: str) -> list[str]:
    command = [
        str(PYTHON),
        "scripts/exp_cmoe/measure_rrd_ppl.py",
        str(run_dir),
        "--base-model-path", str(contract.teacher),
        "--device", device,
        "--dtype", "bfloat16",
        "--seqlen", str(SEQLEN),
        "--strict-load", "1",
        "--output-json", str(run_dir / f"ppl_{kind}.json"),
    ]
    if kind == "wt2_train":
        command.extend(["--jsonl-path", str(contract.train), "--jsonl-name", kind])
    elif kind == "wt2_valtest":
        command.extend(["--jsonl-path", str(contract.valtest), "--jsonl-name", kind])
    elif kind == "c4":
        command.extend(["--datasets", "c4", "--limit-windows", "256"])
    else:
        raise ValueError(f"unknown PPL split: {kind}")
    return command


def alignment_command(contract: Contract, run_dir: Path, kind: str, device: str) -> list[str]:
    if kind == "wt2_train":
        data_kind, data = "token_ids_jsonl", str(contract.train)
    elif kind == "wt2_valtest":
        data_kind, data = "token_ids_jsonl", str(contract.valtest)
    elif kind == "c4":
        data_kind, data = "cmoe_eval", "c4"
    else:
        raise ValueError(f"unknown alignment split: {kind}")
    return [
        str(PYTHON),
        "scripts/exp_cmoe/eval_phase100_activation_mass_controls.py",
        "--run-spec", f"{run_dir.name}={run_dir}",
        "--teacher", str(contract.teacher),
        "--init-dir", str(contract.moe_dir),
        "--data-kind", data_kind,
        "--data", data,
        "--out-json", str(run_dir / f"alignment_{kind}.json"),
        "--device", device,
        "--dtype", "bfloat16",
        "--max-seqlen", str(SEQLEN),
        "--batch-size", "1",
        "--limit-windows", "16",
        "--dataset-seed", "0",
        "--report-exact-set", "0",
        "--report-k", "2",
    ]


def lmeval_command(contract: Contract, run_dir: Path, device: str) -> list[str]:
    command = [
        str(PYTHON),
        "scripts/exp_cmoe/lmeval_cmoe.py",
        "--base_model_path", str(contract.teacher),
        "--manifest", str(run_dir / "manifest.json"),
        "--output_json", str(run_dir / "lmeval.json"),
        "--tasks", EVAL_TASKS,
        "--num_fewshot", "0",
        "--batch_size", "8",
        "--device", device,
        "--dtype", "bfloat16",
        "--moe_type", "cmoe",
        "--cmoe_strict_load", "1",
        "--lmeval_seed", str(contract.seed),
    ]
    state = run_dir / "state_dict.pt"
    if state.exists():
        command.extend(["--moe_state_dict", str(state)])
    else:
        delta = run_dir / "trainable_delta.pt"
        if not delta.exists():
            raise FileNotFoundError(
                f"evaluation requires state_dict.pt or trainable_delta.pt: {run_dir}"
            )
        command.extend([
            "--trainable_delta_path", str(delta),
            "--source_moe_dir", str(contract.moe_dir),
        ])
    return command


def ppl_value(run_dir: Path, kind: str) -> float | None:
    path = run_dir / f"ppl_{kind}.json"
    if not path.exists():
        return None
    values = load_json(path).get("ppl", {})
    value = values.get(kind) if isinstance(values, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def alignment_values(run_dir: Path, kind: str) -> tuple[float | None, float | None]:
    path = run_dir / f"alignment_{kind}.json"
    if not path.exists():
        return None, None
    runs = load_json(path).get("runs", {})
    row = runs.get(run_dir.name, {}) if isinstance(runs, dict) else {}
    hit = row.get("mean_activation_mass_hit_count")
    cosine = row.get("mean_cos_dense_teacher_mlp")
    return (
        float(hit) if isinstance(hit, (int, float)) else None,
        float(cosine) if isinstance(cosine, (int, float)) else None,
    )


def lmeval_values(run_dir: Path) -> dict[str, float]:
    path = run_dir / "lmeval.json"
    if not path.exists():
        return {}
    results = load_json(path).get("results", {})
    output: dict[str, float] = {}
    for task in ("piqa", "winogrande", "arc_easy", "arc_challenge", "hellaswag"):
        row = results.get(task, {}) if isinstance(results, dict) else {}
        keys = (
            ("acc,none", "acc")
            if task == "winogrande"
            else ("acc_norm,none", "acc_norm", "acc,none", "acc")
        )
        for key in keys:
            if key in row:
                output[task] = 100.0 * float(row[key])
                break
    if len(output) == 5:
        output["avg"] = sum(output.values()) / 5.0
    return output


def metrics_for(run_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for kind in ("wt2_train", "wt2_valtest", "c4"):
        ppl = ppl_value(run_dir, kind)
        if ppl is not None:
            metrics[f"ppl_{kind}"] = ppl
        hit, cosine = alignment_values(run_dir, kind)
        if hit is not None:
            metrics[f"hit_{kind}"] = hit
        if cosine is not None:
            metrics[f"cos_{kind}"] = cosine
    for task, value in lmeval_values(run_dir).items():
        metrics[f"lmeval_{task}"] = value
    return metrics


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="qwen2_5_7b",
        help="qwen2_5_7b, qwen2_5_14b, or llama2_7b",
    )
    parser.add_argument("--topology", default="S2A2E8", help="Registry topology, e.g. S2A2E8 or S3A3E8")
    parser.add_argument("--server", default="auto", choices=["auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", default="", help="Optional dense model path override")
    parser.add_argument("--moe-dir", default="", help="Optional exact-token carved CMoE path override")
    parser.add_argument("--train", default="", help="Optional exact 2048-window token_ids JSONL override")
    parser.add_argument("--valtest", default="", help="Optional validation+test token_ids JSONL override")
    parser.add_argument("--data-manifest", default="", help="Optional exact-token data manifest override")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--delta-only", action="store_true", help="Skip the final full Stage2 state_dict.pt")
