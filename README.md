# RRD

RRD (Routing and Residual Distillation) continues training a carved mixture-of-experts model with language-model cross entropy, a routing loss, a residual loss, and final-logit knowledge distillation: `CE + routing loss + 2 x residual loss + KL(teacher logit || student logit)`, with KD weight 1 and temperature 1. Attention, embeddings, norms, and the LM head stay frozen; shared and routed experts and the MLP router are updated. Routing uses hard top-A selection with uniform expert aggregation.

This repository contains the code needed for a Qwen2.5-7B S2A2E8 C4-4M run and response-only Tülu3-10K LoRA fine-tuning. It contains no data, model weights, checkpoints, or experiment logs. The data and model licenses apply separately. The CMoE carving source in `third_party/cmoe/` retains its MIT license; the adapted LLaMA-Factory modules retain the Apache-2.0 license.

## Setup

Use Python 3.11+ and a CUDA build of PyTorch suitable for your GPU. Install the remaining packages, then run from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/src"
export DISABLE_VERSION_CHECK=1
```

## 1. Data preparation

The C4 builder pins `allenai/c4` revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, seed 0, train shards 0–7, validation shard 0, and 2,048-token non-overlapping windows. It separates calibration, train, development, and test documents. The 4M training stream has 2,048 windows (4,194,304 input tokens). Stage the tokenizer before packing C4; section 2 adds the model weights.

```bash
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B").save_pretrained("models/Qwen2.5-7B")
PY
python -m scripts.exp_cmoe.prepare_c4_4m \
  --out-root data/c4 --tokenizer qwen2_5_7b=models/Qwen2.5-7B \
  --calibration-windows 64 build
python -m scripts.exp_cmoe.prepare_c4_4m \
  --out-root data/c4 --tokenizer qwen2_5_7b=models/Qwen2.5-7B \
  --calibration-windows 64 audit
```

For SFT, the builder pins `allenai/tulu-3-sft-mixture` revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`, selects 10,000 examples by source-stratified hashing, and masks all non-assistant tokens.

```bash
python -m scripts.exp_cmoe.build_tulu3_10k_sft_data \
  --tokenizer-path models/Qwen2.5-7B --output-dir data/tulu3
```

## 2. Model preparation

Download the licensed Qwen2.5-7B base model and tokenizer to `models/Qwen2.5-7B`. For example:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-7B", local_dir="models/Qwen2.5-7B")
PY
```

Keep the downloaded files outside Git. `data/`, `models/`, and `outputs/` are ignored.

## 3. Calibration

Carve an E8S2A2 model from the exact C4 calibration windows. The builder's calibration file name records the selected window count.

```bash
python -m scripts.exp_cmoe.carve_cmoe_exact_token_ids \
  --base-model models/Qwen2.5-7B \
  --calibration-token-ids data/c4/qwen2_5_7b/calibration_seed0_n64_seqlen2048.token_ids.jsonl \
  --calibration-manifest data/c4/qwen2_5_7b/manifest.json \
  --save-dir models/rrd-carve --n-shared 2 --n-activated 2 \
  --n-experts 8 --calib-samples 64 --seed 0
```

## 4. CPT

The C4 runner uses BF16, batch size 2, 2,048 tokens per window, constant Adam8bit, and 1,024 updates. It trains with CE, routing loss, residual loss, and full-vocabulary logit KD. It writes a trainable delta and a resumable state. Run its contract audit before training; a two-update smoke run is available with the `smoke` command.

```bash
RRD_ARGS=(
  --model qwen2_5_7b --topology S2A2E8
  --teacher models/Qwen2.5-7B --moe-dir models/rrd-carve
  --train-npy data/c4/qwen2_5_7b/train_master_seed0_n2048_seqlen2048.npy
  --validation-npy data/c4/qwen2_5_7b/c4_dev_seed0_n256_seqlen2048.npy
  --data-manifest data/c4/qwen2_5_7b/manifest.json
  --output-dir outputs/rrd4m --seed 0
)
python -m scripts.exp_cmoe.rrd_c4_4m audit "${RRD_ARGS[@]}"
python -m scripts.exp_cmoe.rrd_c4_4m train "${RRD_ARGS[@]}"
```

## 5. SFT

LoRA fine-tuning starts from the frozen C4-4M RRD checkpoint. The training target is response-only cross entropy; distillation losses are disabled during SFT. The reference LoRA settings are rank 8, alpha 32, dropout 0.1, learning rate `5.95e-5`, one epoch, and seeds 0–4. Run the command once per seed with a distinct output directory.

```bash
python -m scripts.exp_cmoe.cpt_train \
  --moe_dir models/rrd-carve --teacher_model_path models/Qwen2.5-7B \
  --calib_path data/tulu3/tulu3_10k_qwen25_response_only_seqlen2048.jsonl \
  --data_manifest data/tulu3/manifest.json --data_manifest_train_key train \
  --data_manifest_seed_independent --data_source supervised_token_ids_jsonl \
  --data_packing per_doc --data_shuffle 1 --one_epoch --max_seqlen 2048 \
  --per_device_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 5.95e-5 --min_learning_rate 1e-6 --lr_schedule constant \
  --max_grad_norm 1 --log_every 10 --dtype bfloat16 --device cuda:0 --seed 0 \
  --use_8bit_adam --alpha_task 1 --alpha_shared 0 --alpha_router 0 --alpha_kd 0 \
  --mode lora --lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
  --moe_type cmoe --cmoe_n_experts 8 --cmoe_n_activated 2 --cmoe_n_shared 2 \
  --freeze_router --router_arch mlp_h1024 --init_mlp_router_random \
  --mlp_router_aggregation uniform \
  --init_trainable_delta_path outputs/rrd4m/trainable_delta.pt \
  --skip_full_state_dict --write_ready_markers --output_dir outputs/sft/seed0
```

## 6. Evaluation

Evaluate C4 held-out perplexity and the five zero-shot likelihood tasks after CPT. For SFT, pass the SFT adapter and its manifest to the same evaluator; keep the original CPT delta available.

```bash
python -m scripts.exp_cmoe.measure_rrd_ppl outputs/rrd4m \
  --base-model-path models/Qwen2.5-7B --source-moe-dir models/rrd-carve \
  --trainable-delta-path outputs/rrd4m/trainable_delta.pt \
  --jsonl-path data/c4/qwen2_5_7b/c4_test_seed0_n256_seqlen2048.token_ids.jsonl \
  --jsonl-name c4_test --output-json outputs/rrd4m/ppl_c4.json
python -m scripts.exp_cmoe.lmeval_cmoe \
  --base_model_path models/Qwen2.5-7B --source_moe_dir models/rrd-carve \
  --manifest outputs/rrd4m/manifest.json \
  --trainable_delta_path outputs/rrd4m/trainable_delta.pt \
  --moe_type cmoe --cmoe_n_experts 8 --cmoe_n_activated 2 --cmoe_n_shared 2 \
  --tasks piqa,winogrande,arc_easy,arc_challenge,hellaswag \
  --num_fewshot 0 --output_json outputs/rrd4m/avg5.json
python -m scripts.exp_cmoe.lmeval_cmoe \
  --base_model_path models/Qwen2.5-7B --source_moe_dir models/rrd-carve \
  --manifest outputs/sft/seed0/manifest.json \
  --trainable_delta_path outputs/rrd4m/trainable_delta.pt \
  --adapter_path outputs/sft/seed0/adapter \
  --moe_type cmoe --cmoe_n_experts 8 --cmoe_n_activated 2 --cmoe_n_shared 2 \
  --tasks piqa,winogrande,arc_easy,arc_challenge,hellaswag \
  --num_fewshot 0 --output_json outputs/sft/seed0/avg5.json
```

## Resources used

| Stage | Recorded resource |
| --- | --- |
| RRD C4-4M CPT | One NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (about 96 GiB), BF16; 1,024 updates, 4,194,304 input tokens. |
| LoRA-SFT | Related 40M-source runs used one NVIDIA RTX A6000 48GB GPU per seed and peaked at about 41–42 GB. Memory for SFT from this 4M checkpoint has not been measured. |
| Data | The 4M training `int32` array alone is 16 MiB; base model, carve, optimizer state, and checkpoints require substantially more disk. Keep enough free space for atomic checkpoint writes. |

These are observed resource classes, not a guarantee that other GPU or software versions produce the same memory use or scores.

## Code provenance

The RRD w/o Logit-KD one-stage training module and RRD C4 runner derive from committed research snapshots `3e1a86098a4c78045f6c100034ac99734bf57b5b` and the LoRA-SFT trainer from `5d34737143eaf7deab2b3dabff95388bb6aac3ce`. This release removes machine-specific paths and experiment orchestration, narrows C4 preparation to 4M, and exposes model/topology arguments in the C4 runner. This adapted C4-4M RRD path has not been used for a new full GPU reproduction.
