# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Usage:
Single node (MATH-500):
    python inference.py \
            --model="Qwen/Qwen2.5-Math-1.5B" \
            --dataset="math" \
            --dp-size=2 \
            --batch-size=16 \
            --rollout-n=16 \
            --output_file=outputs/math500.jsonl

Single node (AMC23):
    python inference.py \
            --model="Qwen/Qwen2.5-Math-1.5B" \
            --dataset="amc" \
            --dp-size=2 \
            --batch-size=16 \
            --rollout-n=32 \
            --output_file=outputs/amc23.jsonl

Single node (AIME25):
    python inference.py \
            --model="Qwen/Qwen2.5-Math-1.5B" \
            --dataset="aime" \
            --dp-size=2 \
            --batch-size=16 \
            --rollout-n=32 \
            --output_file=outputs/aime25.jsonl
"""

import os
import sys
import time
import json
from time import sleep
from pathlib import Path


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        help="Dataset alias (math, amc, aime) or full Hugging Face path.",
    )
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument(
        "--temperature", type=float, default=1, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=1, help="Sampling top-p")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to merged JSONL output (per node; exact for single-node).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per DP rank for vLLM.generate.",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=1,
        help="Number of output sequences (completions) per prompt.",
    )
    return parser.parse_args()


def main(
    dataset,
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,  # currently unused, but kept for API symmetry
    dp_master_port,  # currently unused, but kept for API symmetry
    GPUs_per_dp_rank,
    output_file,
    batch_size,
    seq_len,
    temperature,
    top_p,
    n,
):
    # tp_size=1, dp_size=2:
    start_gpu_id = local_dp_rank * GPUs_per_dp_rank
    end_gpu_id = start_gpu_id + GPUs_per_dp_rank

    cuda_visible_devices = ",".join(str(i) for i in range(start_gpu_id, end_gpu_id))

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    print(
        f"DP Rank {global_dp_rank} (Local {local_dp_rank}) mapped to GPU(s): {cuda_visible_devices}"
    )

    # Clean up any existing distributed-related env vars that might interfere.
    keys_to_remove = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "VLLM_DP_RANK",
        "VLLM_DP_SIZE",
    ]
    for key in keys_to_remove:
        if key in os.environ:
            del os.environ[key]

    try:
        from vllm import LLM, SamplingParams
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"Import Error in process (vllm/torch/datasets/transformers): {e}")
        return

    print(f"[Rank {global_dp_rank}] Logical GPU count: {torch.cuda.device_count()}")

    dataset_map = {
        "math": "math-ai/math500",
        "amc": "math-ai/amc23",
        "aime": "math-ai/aime25",
    }
    # Resolve the short name to the full path.
    # If the user provides a full path (not in the map), it uses the input as is.
    dataset_path = dataset_map.get(dataset.lower(), dataset)
    print(f"[Rank {global_dp_rank}] Loading dataset: {dataset} -> {dataset_path}")
    # === MAPPING LOGIC END ===

    # 1. Load test set using the resolved path
    ds = load_dataset(dataset_path, split="test")

    # Use the resolved path name for schema detection
    name = dataset_path.lower()

    if name.endswith("math500") or "math500" in name:
        problems = ds["problem"]
        gold_answers = ds["answer"]
    elif name.endswith("amc23") or "amc23" in name:
        problems = ds["question"]
        gold_answers = ds["answer"]
    elif name.endswith("aime25") or "aime25" in name:
        problems = ds["problem"]
        gold_answers = ds["answer"]
    else:
        # Fallback: try common schemas automatically
        cols = set(ds.column_names)
        if {"problem", "answer"}.issubset(cols):
            problems = ds["problem"]
            gold_answers = ds["answer"]
        elif {"question", "answer"}.issubset(cols):
            problems = ds["question"]
            gold_answers = ds["answer"]
        else:
            raise ValueError(
                f"Unsupported dataset schema for '{dataset}'. "
                f"Columns found: {ds.column_names}"
            )
    # 2. Tokenizer from HF model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Prepare chat-style prompts: each problem as a single-turn user message
    chat_conversations = [
        [
            {
                "role": "user",
                "content": p,
            }
        ]
        for p in problems
    ]

    # Apply chat template to get the actual text prompts
    prompt_strs = []
    for conv in chat_conversations:
        # Try Qwen-style signature first (with enable_thinking),
        # then fall back to more generic signatures, then raw text.
        try:
            prompt_str = tokenizer.apply_chat_template(
                conversation=conv,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
        except Exception:
            prompt_str = tokenizer.apply_chat_template(
                conversation=conv,
                add_generation_prompt=True,
                tokenize=False,
            )
        prompt_strs.append(prompt_str)

    # 3. Data-parallel split: each DP rank processes a different slice.
    # We distribute the remainder across the lower ranks so that every rank
    # gets either floor(num_samples / dp_size) or ceil(num_samples / dp_size) samples.
    num_samples = len(prompt_strs)

    base = num_samples // dp_size
    rem = num_samples % dp_size

    if global_dp_rank < rem:
        # This rank gets (base + 1) samples.
        start = global_dp_rank * (base + 1)
        end = start + (base + 1)
    else:
        # Ranks >= rem start after all the larger chunks.
        start = rem * (base + 1) + (global_dp_rank - rem) * base
        end = start + base

    rank_prompt_strs = prompt_strs[start:end]
    rank_problems = problems[start:end]
    rank_gold_answers = gold_answers[start:end]

    if len(rank_prompt_strs) == 0:
        print(f"[Rank {global_dp_rank}] No prompts to process, exiting.")
        return

    print(
        f"[Rank {global_dp_rank}] Needs to process {len(rank_prompt_strs)} prompts "
        f"(indices [{start}, {end}))"
    )

    # Sampling params; you can tune these, or make them CLI args if needed.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=seq_len,
        n=n,
    )

    # Create an LLM.
    llm = LLM(
        model=model,
        dtype="half",
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=False,
        enable_expert_parallel=False,
        trust_remote_code=True,
        max_model_len=3072,
        gpu_memory_utilization=0.9,
    )

    base_path = Path(output_file)
    partial_path = base_path.with_name(
        base_path.stem + f".rank{global_dp_rank}.jsonl"
    )
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    num_prompts = len(rank_prompt_strs)
    print(
        f"[Rank {global_dp_rank}] Starting generation for {num_prompts} prompts "
        f"with batch_size={batch_size}"
    )

    with partial_path.open("w", encoding="utf-8") as f_out:
        # Batched generation
        for i in range(0, num_prompts, batch_size):
            batch_prompts = rank_prompt_strs[i : i + batch_size]
            batch_problems = rank_problems[i : i + batch_size]
            batch_gold = rank_gold_answers[i : i + batch_size]

            outputs = llm.generate(batch_prompts, sampling_params)

            for j, out in enumerate(outputs):
                local_idx = i + j
                global_idx = start + local_idx  # index in full dataset

                # vLLM returns up to `n` outputs per prompt in out.outputs
                for sample_id, seq in enumerate(out.outputs):
                    gen_text = seq.text

                    record = {
                        "id": global_idx,  # problem index
                        "sample_id": sample_id,  # which completion for this problem
                        "dp_rank": global_dp_rank,
                        "problem": batch_problems[j],
                        "gold": batch_gold[j],
                        "answer": gen_text,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[Rank {global_dp_rank}] "
                f"Processed {min(i + batch_size, num_prompts)}/{num_prompts} prompts "
                f"with n={n} completions each"
            )

    # Optionally print a few samples for debugging
    print(f"[Rank {global_dp_rank}] Saved partial JSONL to {partial_path}")
    sleep(1)


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    from vllm.utils import get_open_port

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    import torch
    physical_gpus = torch.cuda.device_count()
    required_gpus = dp_per_node * tp_size
    assert physical_gpus >= required_gpus, (
        f"Need at least {required_gpus} GPUs per node "
        f"for dp_per_node={dp_per_node}, tp_size={tp_size}, "
        f"but only found {physical_gpus}."
    )

    procs = []
    # Track the global DP ranks on this node
    global_dp_ranks = list(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    )

    start_time = time.time()

    for local_dp_rank, global_dp_rank in enumerate(global_dp_ranks):
        proc = multiprocessing.Process(
            target=main,
            args=(
                args.dataset,
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.output_file,
                args.batch_size,
                args.seq_len,
                args.temperature,
                args.top_p,
                args.rollout_n,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    end_time = time.time()
    print(f"Total time consuming is: {end_time - start_time:.2f}s")

    # Merge per-rank JSONL files on this node if all child processes succeeded
    if exit_code == 0:
        partial_files = []

        base_path = Path(args.output_file)
        for global_dp_rank in global_dp_ranks:
            part_path = base_path.with_name(
                base_path.stem + f".rank{global_dp_rank}.jsonl"
            )
            if part_path.exists():
                partial_files.append(part_path)

        if partial_files:
            if node_size == 1:
                merged_path = Path(args.output_file)
            else:
                # Per-node merged file; you can later merge node0/node1/... as needed.
                merged_path = Path(f"{args.output_file}.node{node_rank}.jsonl")

            merged_path.parent.mkdir(parents=True, exist_ok=True)
            with merged_path.open("w", encoding="utf-8") as out_f:
                for part in sorted(partial_files):
                    with part.open("r", encoding="utf-8") as in_f:
                        for line in in_f:
                            out_f.write(line)

            print(
                f"Merged {len(partial_files)} partial files into {merged_path.resolve()}"
            )

    sys.exit(exit_code)