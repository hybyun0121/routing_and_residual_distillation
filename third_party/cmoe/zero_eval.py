import fnmatch


def _pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(task_names)


def _expanded_patterns(task_list):
    patterns = []
    for task in task_list:
        if task == "mmlu":
            patterns.append("mmlu")
        else:
            patterns.append(task)
    return patterns


def _available_tasks_legacy():
    from lm_eval import tasks

    return list(tasks.ALL_TASKS)


def _available_tasks_modern():
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    return task_manager, list(task_manager.all_tasks)


def _resolve_task_names(task_list, available_tasks):
    task_names = _pattern_match(_expanded_patterns(task_list), available_tasks)
    return task_names if task_names else task_list


def _eval_zero_shot_modern(model, tokenizer, task_names, num_fewshot, batch_size, device, limit):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    return evaluator.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        check_integrity=False,
        task_manager=task_manager,
    )


def _eval_zero_shot_legacy(
    model_name,
    model,
    tokenizer,
    task_names,
    num_fewshot,
    batch_size,
    device,
    limit,
    use_accelerate,
    add_special_tokens,
):
    from lm_eval import evaluator

    model_args = f"pretrained={model_name}"
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"

    return evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
    )


def eval_zero_shot(
    model_name,
    model,
    tokenizer,
    task_list=None,
    num_fewshot=0,
    use_accelerate=False,
    add_special_tokens=False,
    batch_size=1,
    device="cuda:0",
    limit=None,
):
    if task_list is None:
        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if limit is None and ("70b" in model_name.lower() or "65b" in model_name.lower()):
        limit = 2000

    try:
        _, available_tasks = _available_tasks_modern()
        task_names = _resolve_task_names(task_list, available_tasks)
        return _eval_zero_shot_modern(model, tokenizer, task_names, num_fewshot, batch_size, device, limit)
    except (ImportError, AttributeError, TypeError):
        available_tasks = _available_tasks_legacy()
        task_names = _resolve_task_names(task_list, available_tasks)
        return _eval_zero_shot_legacy(
            model_name,
            model,
            tokenizer,
            task_names,
            num_fewshot,
            batch_size,
            device,
            limit,
            use_accelerate,
            add_special_tokens,
        )


def _pick_metric(task_name, metrics):
    task_l = task_name.lower()
    if "hellaswag" in task_l or "arc_" in task_l or "arc-" in task_l:
        candidates = ["acc_norm,none", "acc_norm", "acc,none", "acc"]
    else:
        candidates = ["acc,none", "acc", "exact_match,none", "exact_match"]

    for metric in candidates:
        if metric in metrics:
            return metric, metrics[metric]
    return None, None


def summarize_results(results):
    task_results = results.get("results", {})
    summary = {}
    mmlu_values = []

    for task_name, metrics in task_results.items():
        metric, value = _pick_metric(task_name, metrics)
        if metric is None:
            continue

        summary[task_name] = {
            "metric": metric,
            "value": value,
        }

        task_l = task_name.lower()
        if "mmlu" in task_l or "hendryckstest" in task_l:
            mmlu_values.append(value)

    if mmlu_values:
        summary["mmlu_avg"] = sum(mmlu_values) / len(mmlu_values)

    return summary
