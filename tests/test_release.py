from argparse import Namespace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.exp_cmoe.prepare_c4_4m import BUDGET_WINDOWS
from scripts.exp_cmoe import rrd_1_stage_2607 as base
from scripts.exp_cmoe.logit_distillation import logit_kd_loss
from scripts.exp_cmoe.rrd_logit_kd_c4_4m import (
    DEFAULT_WINDOWS,
    make_contract,
    resume_contract_sha256,
)


def main() -> None:
    assert BUDGET_WINDOWS == {"4m": 2_048}
    assert DEFAULT_WINDOWS == 2_048
    student = torch.zeros(1, 3, 4, requires_grad=True)
    teacher = torch.tensor([[[2.0, 0.0, 0.0, 0.0]] * 3])
    loss = logit_kd_loss(student, teacher, torch.tensor([[1, 2, 3]]))
    loss.backward()
    assert loss.item() > 0
    assert student.grad is not None
    assert student.grad[0, :2].abs().sum().item() > 0
    assert student.grad[0, 2].abs().sum().item() == 0
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "config.json").write_text(
            json.dumps({"num_hidden_layers": 28, "hidden_size": 3584})
        )
        (path / "manifest.json").write_text("{}")
        for model, topology, active in (
            ("qwen2_5_7b", "S2A2E8", 2),
            ("llama2_7b", "S3A3E8", 3),
        ):
            args = Namespace(
                teacher=path,
                moe_dir=path,
                train_npy=path / "train.npy",
                validation_npy=path / "dev.npy",
                data_manifest=path / "manifest.json",
                output_dir=path / "out",
                seed=0,
                model=model,
                topology=topology,
                total_windows=DEFAULT_WINDOWS,
                router_base_lr=base.DEFAULT_ROUTER_BASE_LR,
            )
            contract = make_contract(args)
            assert contract.model_key == model
            assert contract.topology.active == active
        base.configure_logit_kd(0.0, 1.0, 128)
        without_kd = resume_contract_sha256(args, contract)
        base.configure_logit_kd(1.0, 1.0, 128)
        assert resume_contract_sha256(args, contract) != without_kd


if __name__ == "__main__":
    main()
