"""
Script for PRM confidence experiment.
Loads questions and PRM scores from HuggingFace dataset, runs LLM inference with full reasoning, computes confidence (average entropy of last 10% tokens), and plots confidence vs PRM score.
"""

import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils.llm import load_llm_with_retries, get_prompt_format, get_sampling_params

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PRM confidence experiment with vLLM.")

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="young-j-park/prm_calibration",
        help="HuggingFace dataset name to use."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=1
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true"
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mihika_experiment_results"
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )
    return parser.parse_args()

def get_entropy_from_logprobs(logprobs):
    if not logprobs:
        return np.nan
    a = np.array(logprobs, dtype=np.float64)
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    Z = np.sum(exp_a)
    if Z == 0.0:
        return np.nan
    p = exp_a / Z
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(p * np.log(p))
    return float(H)

def extract_topk_logprobs_list(pos_entry, k=20):
    if pos_entry is None:
        return []
    vals = []
    for _, v in pos_entry.items():
        try:
            lp = float(v)
        except (TypeError, ValueError):
            try:
                lp = float(getattr(v, "logprob"))
            except Exception:
                continue
        vals.append(lp)
    vals.sort(reverse=True)
    return vals[:k]

def main():
    
    args = parse_args()

    # 1. Load only the math500/Qwen2.5-Math-7B-Instruct/data.json file from the PRM calibration dataset
    dataset_name = args.dataset_name
    dataset_path = "math500/Qwen2.5-Math-7B-Instruct/data.json"
    ds = load_dataset(dataset_name, data_files=dataset_path, split="train")

    import random
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), 3000)
    dataset = ds.select(indices)
    
    # 2. Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Extract data from loaded dataset
    questions = [sample["question"] for sample in dataset]
    if "success_prob" in dataset.column_names:
        prm_scores = [sample["success_prob"] for sample in dataset]
    elif "prm_score" in dataset.column_names:
        prm_scores = [sample["prm_score"] for sample in dataset]
    else:
        raise KeyError("Neither 'success_prob' nor 'prm_score' found in dataset columns.")

    # 4. Prepare prompts (prompt + reasoning prefix)
    prompt_format = get_prompt_format(args.model_id)
    formatted_prompts = [prompt_format.replace("{input}", q) for q in questions]

    # 5. Inference parameters: set max_tokens=1 to only get prompt logprobs
    sampling_params = get_sampling_params(
        model_id=args.model_id,
        n_generations=args.n_generations,
        temperature=args.temperature,
        max_tokens=1,
        prompt_logprobs=20,
    )

    # 6. Load LLM and batch inference for all prompts
    llm = load_llm_with_retries(args)
    results = llm.generate(formatted_prompts, sampling_params)

    # 7. Confidence calculation (average entropy of last 10% tokens in prompt+reasoning prefix)
    confidences = []
    for res in results:
        prompt_lp_list = getattr(res, "prompt_logprobs", None)
        if prompt_lp_list is not None and len(prompt_lp_list) > 0:
            n_tokens = len(prompt_lp_list)
            last_10pct = prompt_lp_list[int(n_tokens*0.9):]
            entropies = [get_entropy_from_logprobs(extract_topk_logprobs_list(pos_entry, k=20)) for pos_entry in last_10pct]
            avg_entropy = np.nanmean(entropies)
            confidences.append(avg_entropy)
        else:
            confidences.append(np.nan)

    # 8. Save results
    output_data = []
    for q, prm, conf in zip(questions, prm_scores, confidences):
        output_data.append({
            "question": q,
            "prm_score": prm,
            "confidence": conf
        })

    with open(os.path.join(args.output_dir, "experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 9. Plot confidence vs PRM score
    plt.figure()
    plt.scatter(confidences, prm_scores, alpha=0.7)
    plt.xlabel("Confidence (avg entropy, last 10% tokens)")
    plt.ylabel("PRM Score (ground-truth)")
    plt.title("Confidence vs PRM Score")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "confidence_vs_prm_score.png"), bbox_inches="tight", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
