import time

import torch
import torch.nn as nn

from tqdm import *

import os

import copy

from CMoE_utils import *
from CMoE_model import *
from zero_eval import *
from sft_utils import simple_sft

DEV = torch.device('cuda:0')

TABLE1_TASKS = [
    ("piqa", 0),
    ("winogrande", 0),
    ("arc_easy", 0),
    ("arc_challenge", 0),
    ("hellaswag", 0),
]

TABLE3_TASKS = [
    ("mmlu", 5),
]

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model


def get_qwen(model):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='auto')
    model.seqlen = 2048
    return model


def carve_only(model, dataloader, dev, args):
    """Replace each decoder layer's mlp with a freshly carved MoE module.

    Deterministic given the same calibration data, seed, and hyperparameters —
    callers can rebuild the empty MoE structure later and overwrite weights from a
    saved state_dict (see scripts/eval_carved.py)."""
    print('Starting ...')

    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = args.calib_bsz
    calib_batches = args.calib_samples // bsz

    inps = torch.zeros(
        (calib_batches, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
            if cache['i'] >= calib_batches:
                break

    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    model.cuda()
    layers.cuda()

    carve_inp = copy.deepcopy(inps[0])
    for layer in tqdm(layers, desc='Carving MoE layers...'):
        moe_out = construct_moe(
            layer, carve_inp, attention_mask, position_ids,
            n_experts=args.nexperts, n_activated=args.nactivated,
            n_shared=args.nshared, args=args,
            rotary_emb=getattr(model.model, "rotary_emb", None),
        )
        carve_inp = moe_out

    return model


def cmoe_sequential(model, dataloader, dev, args):
    use_cache = model.config.use_cache
    carve_only(model, dataloader, dev, args)

    tick_1 = time.time()

    if getattr(args, "skip_internal_ppl", False):
        model.eval()
        model.config.use_cache = use_cache
        return model, tick_1, tick_1, ["skipped"]

    print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, DEV, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")

    tick_2 = time.time()

    model.eval()
    model.config.use_cache = use_cache

    return model, tick_1, tick_2, pre_ppl


def default_lm_eval_model_name(model_path):
    model_path_l = model_path.lower()
    if 'llama-3' in model_path_l:
        return "meta-llama/Meta-Llama-3-8B"
    if 'llama-2' in model_path_l:
        return "meta-llama/Llama-2-7b-hf"
    return model_path


def run_eval_suite(model, tokenizer, model_name, suite, args, prefix):
    if args.eval_device.startswith('cuda'):
        model.cuda()
    else:
        model.to(args.eval_device)
    model.eval()

    results = {}
    for task_pattern, fewshot in suite:
        print(f"{prefix}: task={task_pattern}, fewshot={fewshot}")
        task_results = eval_zero_shot(
            model_name,
            model,
            tokenizer,
            task_list=[task_pattern],
            num_fewshot=fewshot,
            batch_size=args.zero_batch_size,
            device=args.eval_device,
        )
        results[f"{task_pattern}_{fewshot}shot"] = {
            "summary": summarize_results(task_results),
            "raw": task_results,
        }
    return results

@torch.no_grad()
def cmoe_ppl_eval(model, testenc, dev, eval_set, args):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers)), desc= 'Processing...'):

        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache

    return ppl.item()

def save_results(file_name, results):
    if not isinstance(results, str):
        results = str(results)
    results = results + '\n'
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(results)
    else:
        with open(file_name, "a") as file:
            file.write(results)


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str,
        help='Where to extract calibration data from. Use wikitext2/ptb/c4 or a JSONL file with {"text": ...} records.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(
        '--calib-samples', type=int, default=8,
        help='Number of calibration samples used for activation profiling.'
    )
    parser.add_argument(
        '--calib-bsz', type=int, default=8,
        help='Calibration batch size. Keep equal to --calib-samples for the current single-batch construction path.'
    )
    parser.add_argument(
        '--seqlen', type=int, default=None,
        help='Override model.seqlen for calibration construction.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--extra-lr',
        type=float, default=0.001,
        help='Initial learning rate for extra scale for router.'
    )
    parser.add_argument(
        '--k-act', type=int, default=10,
        help='TopK number for the ATopK. K_a in paper.'
    )
    parser.add_argument(
        '--bias-speed',
        type=float, default=0.001,
        help='Bias update speed for load balancing. Gamma in paper.'
    )
    parser.add_argument(
        '--nexperts', type=int, default=16,
        help='Total number of experts. N in paper.'
    )
    parser.add_argument(
        '--nactivated', type=int, default=2,
        help='Number of activated routed experts.'
    )
    parser.add_argument(
        '--nshared', type=int, default=2,
        help='Number of shared experts.'
    )
    parser.add_argument(
        '--epoch', type=int, default=1,
        help='SFT epoch for CMoE.'
    )
    parser.add_argument(
        '--lora-lr',
        type=float, default=5.95e-5,
        help='LoRA learning rate used for CMoE fine-tuning.'
    )
    parser.add_argument(
        '--sft-bsz', type=int, default=2,
        help='SFT batch size for CMoE.'
    )
    parser.add_argument(
        '--skip-sft', action='store_true',
        help='Skip LoRA fine-tuning and only evaluate the analytically constructed model.'
    )
    parser.add_argument(
        '--skip-internal-ppl', action='store_true',
        help='Skip CMoE internal wikitext/c4 PPL evaluation. Use external lm-eval instead.'
    )
    parser.add_argument(
        '--eval-zero', action='store_true',
        help='Run the legacy downstream task evaluation preset.'
    )
    parser.add_argument(
        '--eval-table1', action='store_true',
        help='Evaluate the fine-tuned model on the Llama-2 7B Table 1 task set.'
    )
    parser.add_argument(
        '--eval-table3', action='store_true',
        help='Evaluate Table 3 MMLU-5shot for training-free and fine-tuned regimes.'
    )
    parser.add_argument(
        '--zero-batch-size', type=int, default=1,
        help='Batch size for lm-evaluation-harness tasks.'
    )
    parser.add_argument(
        '--eval-device', type=str, default='cuda:0',
        help='Device string for lm-evaluation-harness.'
    )
    parser.add_argument(
        '--lm-eval-model-name', type=str, default=None,
        help='Reference HF model name passed to lm-evaluation-harness. Defaults from model path.'
    )
    parser.add_argument(
        '--prefix', type=str, default=None,
        help='Prefix the results folder if needed.'
    )
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='If set, save the carved+SFT model state_dict and a manifest.json (carving hyperparameters) here so eval_carved.py can reuse the run without redoing LoRA.'
    )

    args = parser.parse_args()
    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be positive.")
    if args.calib_samples != args.calib_bsz:
        raise ValueError("--calib-samples must equal --calib-bsz in the current single-batch construction path.")

    if 'llava' in args.model.lower():
        model = get_llava(args.model)
    elif 'qwen' in args.model.lower():
        model = get_qwen(args.model)
    else:
        model = get_llama(args.model)
    if args.seqlen is not None:
        model.seqlen = int(args.seqlen)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.calib_samples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        bsz=args.calib_bsz,
    )

    print("number of fine-tuning data: ", args.nsamples)
    print("number of calibration data: ", args.calib_samples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tick = time.time()
    carved_model, tick_1, _, pre_ppl = cmoe_sequential(model, dataloader, DEV, args)
    rt_construct = tick_1 - tick
    print("Runtime of training-free construction: ", rt_construct)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    name = args.lm_eval_model_name or default_lm_eval_model_name(args.model)

    model_name = args.model.split("/")[-1]
    dataset_name = os.path.basename(args.dataset.rstrip("/"))
    dataset_name = dataset_name.replace("/", "_").replace(" ", "_")
    file_name = f"{model_name}_{dataset_name}_calib_{args.calib_samples}_sft_{args.nsamples}_epoch_{args.epoch}_S{args.nshared}_A{args.nactivated}_E{args.nexperts}.txt"
    dir_path = os.path.join('./result_logs', args.prefix) if args.prefix is not None else './result_logs'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.join(dir_path, file_name)

    save_results(file_name, f"pre_ppl: {str(pre_ppl)}")

    pre_zero = None
    if args.eval_table3:
        pre_zero = run_eval_suite(
            carved_model,
            tokenizer,
            name,
            TABLE3_TASKS,
            args,
            prefix="table3_training_free",
        )
        save_results(file_name, f"table3_training_free: {pre_zero}")

    ft_start = time.time()
    if args.nsamples > 0 and not args.skip_sft:
        for layer in carved_model.model.layers:
            layer.mlp.cus_training = True

        carved_model.cuda()
        carved_model = simple_sft(carved_model, args, epoch=args.epoch)

        for layer in carved_model.model.layers:
            layer.mlp.cus_training = False
        carved_model.eval()
    ft_time = time.time() - ft_start
    rt = rt_construct + ft_time
    print("Runtime of fine-tuning construction: ", rt)

    if args.save_dir is not None:
        import json
        os.makedirs(args.save_dir, exist_ok=True)
        sd_path = os.path.join(args.save_dir, "state_dict.pt")
        manifest_path = os.path.join(args.save_dir, "manifest.json")
        torch.save(carved_model.state_dict(), sd_path)
        manifest = {
            "base_model": args.model,
            "dataset": args.dataset,
            "seed": args.seed,
            "calib_samples": args.calib_samples,
            "calib_bsz": args.calib_bsz,
            "nshared": args.nshared,
            "nactivated": args.nactivated,
            "nexperts": args.nexperts,
            "k_act": args.k_act,
            "bias_speed": args.bias_speed,
            "nsamples": args.nsamples,
            "epoch": args.epoch,
            "sft_bsz": args.sft_bsz,
            "lora_lr": args.lora_lr,
            "extra_lr": args.extra_lr,
            "skip_sft": args.skip_sft,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved state_dict + manifest to {args.save_dir}")

    if args.skip_internal_ppl:
        ppl = ["skipped"]
    else:
        datasets = ['wikitext2', 'c4-new']
        ppl = []
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            eval_set = dataset
            ppl_i = cmoe_ppl_eval(carved_model, testloader, DEV, eval_set, args)
            ppl.append(f"{dataset}: {ppl_i}")
    save_results(file_name, f"ft_ppl: {str(ppl)}")

    table1_zero = None
    if args.eval_table1:
        table1_zero = run_eval_suite(
            carved_model,
            tokenizer,
            name,
            TABLE1_TASKS,
            args,
            prefix="table1_fine_tuned",
        )
        save_results(file_name, f"table1_fine_tuned: {table1_zero}")

    table3_ft_zero = None
    if args.eval_table3:
        table3_ft_zero = run_eval_suite(
            carved_model,
            tokenizer,
            name,
            TABLE3_TASKS,
            args,
            prefix="table3_fine_tuned",
        )
        save_results(file_name, f"table3_fine_tuned: {table3_ft_zero}")
    save_results(file_name, f"runtime_construct: {rt_construct}")
    save_results(file_name, f"runtime_all: {rt}")

    if args.eval_zero:
        task_list = ["winogrande"]
        results_1 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=5, batch_size=args.zero_batch_size, device=args.eval_device)
        save_results(file_name, results_1)

        task_list = ["arc_challenge"]
        results_2 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=25, batch_size=args.zero_batch_size, device=args.eval_device)
        save_results(file_name, results_2)

        task_list = ["hellaswag"]
        results_3 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=10, batch_size=args.zero_batch_size, device=args.eval_device)
        save_results(file_name, results_3)

        task_list = ["sciq","piqa"]
        results_4 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=0, batch_size=args.zero_batch_size, device=args.eval_device)
        save_results(file_name, results_4)

        task_list = ["boolq"]
        results_5 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=32, batch_size=args.zero_batch_size, device=args.eval_device)
        save_results(file_name, results_5)


    print("number of fine-tuning data: ", args.nsamples)
    print("number of calibration data: ", args.calib_samples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)
