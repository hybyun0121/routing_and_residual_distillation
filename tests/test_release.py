from argparse import Namespace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.exp_cmoe.build_c4_budget_scaling_bundle import BUDGET_WINDOWS
from scripts.exp_cmoe.rrd_1_stage_2607_c4_scaling import make_contract


def main() -> None:
    assert BUDGET_WINDOWS == {"40m": 20_480}
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "config.json").write_text(
            json.dumps({"num_hidden_layers": 28, "hidden_size": 3584})
        )
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
            )
            contract = make_contract(args)
            assert contract.model_key == model
            assert contract.topology.active == active


if __name__ == "__main__":
    main()
