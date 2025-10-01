"""
Script to run LLM forward inference using vLLM (Calibration).
Collects prompt top-k logprobs (k=20) as confidence traces and plots average entropy over the prompt.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.data import get_dataset
from utils.llm import (
    load_llm_with_retries,
    get_prompt_format,
    get_sampling_params,
    prioritize_boxed
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inference script with vLLM.")

    # Model / HF Hub
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID from Hugging Face Hub or local path."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face token for private models."
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500train",
        choices=["math500", "math500train", "aime2024", "aime2025", "aime2025-2"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Process only this chunk index from the dataset."
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=5,
        help="Total chunks to split the dataset into."
    )

    # Inference parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="calibration",
        choices=["calibration", "onepass"],
        help="Which mode to run on."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=8,
        help="Number of generation samples per prompt."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1,  # <-- we now generate exactly 1 token; prompt_logprobs still captured
        help="Max new tokens to generate."
    )

    # vLLM parameters
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory used by vLLM."
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="Use CUDA graph if possible."
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching for large LLMs in vLLM."
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        help="Enable chunked prefill for large LLMs in vLLM."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type (e.g., auto, float16, bf16, float32)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./infer_results",
        help="Directory to save inference results."
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode."
    )

    return parser.parse_args()


def _extract_topk_logprobs_list(pos_entry, k=20):
    """
    From a single prompt position's logprobs entry, return a sorted (desc) list of top-k logprob floats.
    vLLM prompt_logprobs entry is typically a dict[token_str -> float or ObjWithLogprob].
    We return only the floats (no tokens) to shrink JSON.
    """
    if pos_entry is None:
        return []
    vals = []
    for _, v in pos_entry.items():
        # v can be float or an object with .logprob
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


def _entropy_from_logprobs(logprobs):
    """
    Given a list of logprobs for top-k candidates at one position,
    compute entropy of the truncated distribution:
        H = -sum_i p_i * log(p_i), with p_i = softmax(logprobs)_i
    Returns np.nan if list is empty.
    """
    if not logprobs:
        return np.nan
    # stabilize
    a = np.array(logprobs, dtype=np.float64)
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    Z = np.sum(exp_a)
    if Z == 0.0:
        return np.nan
    p = exp_a / Z
    # natural-log entropy (nats)
    # If you want bits, divide by ln(2).
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(p * np.log(p))
    return float(H)


def main():
    """Main function for inference."""
    args = parse_args()

    # 1. Load model with retries
    llm = load_llm_with_retries(args)

    # 2. Load dataset
    dataset, q_key, a_key = get_dataset(
        dataset_name=args.dataset,
        chunk=args.chunk,
        total_chunks=args.total_chunks,
    )
    print(f"Loaded dataset: {args.dataset} | Number of samples: {len(dataset)}")

    # 3. Prepare sampling parameters (ensure utils.llm sets prompt_logprobs=20)
    if args.mode == "calibration":
        n_generations = int(args.n_generations * 1.25)
    else:
        n_generations = args.n_generations
    sampling_params = get_sampling_params(
        model_id=args.model_id,
        n_generations=n_generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,  # 1
        # make sure get_sampling_params passes prompt_logprobs=20 through to vLLM
    )

    # 4. Select the prompt format
    prompt_format = get_prompt_format(args.model_id)

    # 5. Prepare prompts
    formatted_prompts = []
    for sample in dataset:
        user_prompt = prompt_format.replace("{input}", sample[q_key])
        formatted_prompts.append(user_prompt)
        if args.debug and len(formatted_prompts) == 3:
            break

    # 6. Inference
    results = llm.generate(formatted_prompts, sampling_params)

    # 7. Post-process: collect generations (if any) and conf_traces = list of lists of logprobs
    all_results = []
    all_conf_traces = []  # per-prompt -> [per-position -> [top-20 logprob floats]]

    for res in results:
        if args.mode == "calibration":
            # you don't care about generation text, but keep behavior consistent
            final_list = prioritize_boxed(res.outputs, args.n_generations)
        else:
            final_list = [out.text for out in res.outputs]
        all_results.append(final_list)

        # Extract prompt confidence traces
        conf_trace = []
        prompt_lp_list = getattr(res, "prompt_logprobs", None)
        if prompt_lp_list is not None:
            for pos_entry in prompt_lp_list:
                conf_trace.append(_extract_topk_logprobs_list(pos_entry, k=20))
        all_conf_traces.append(conf_trace)

    print(f"Processed {len(all_conf_traces)} prompts, max_tokens={args.max_tokens}.")

    # 8. Save results JSON (conf_traces contains only floats)
    output_dir = os.path.join(
        args.output_dir,
        args.model_id.replace("/", "_"),
        args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)

    out_fname = f"inference_chunk_{args.chunk}.json" if args.mode == "calibration" \
                else f"inference_onepass_chunk_{args.chunk}.json"
    out_path = os.path.join(output_dir, out_fname)

    dataset_list = list(dataset)
    output_data = []
    for item, gens, conf in zip(dataset_list, all_results, all_conf_traces):
        output_data.append({
            "chunk": args.chunk,
            "total_chunks": args.total_chunks,
            "question": item[q_key],
            "gold_answer": item[a_key],
            "generations": gens,
            "generation_cnts": args.n_generations,
            # Each position is a list of up to 20 logprob floats (descending)
            "conf_traces": conf
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {out_path}")

    # 9. Plot aggregation: average ENTROPY vs prompt position (over all prompts)
    #    Entropy is computed from the truncated top-20 distribution at each position.
    entropy_traces = []
    for conf in all_conf_traces:
        ent = [_entropy_from_logprobs(lp_list) for lp_list in conf]
        entropy_traces.append(ent)

    if entropy_traces:
        max_len = max(len(t) for t in entropy_traces)
        padded = np.full((len(entropy_traces), max_len), np.nan, dtype=float)
        for i, t in enumerate(entropy_traces):
            padded[i, :len(t)] = t
        mean_entropy = np.nanmean(padded, axis=0)

        plt.figure()
        plt.plot(np.arange(1, len(mean_entropy) + 1), mean_entropy)
        plt.xlabel("Prompt token position")
        plt.ylabel("Average entropy (nats, top-20)")
        plt.title(f"Average Prompt Entropy vs. Position ({args.model_id}, {args.dataset})")
        plt.grid(True)

        plot_path = os.path.join(output_dir, f"avg_prompt_entropy_chunk_{args.chunk}.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        try:
            plt.show()
        except Exception:
            pass
        print(f"Entropy plot saved to {plot_path}")
    else:
        print("No prompt logprobs found to compute entropy.")


if __name__ == "__main__":
    main()
