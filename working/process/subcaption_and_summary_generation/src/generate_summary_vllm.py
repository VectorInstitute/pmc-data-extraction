#!/usr/bin/env python3
import argparse
import os
import re
import time

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

prompt = (
    "### INSTRUCTIONS:\n"
    "You will be provided with:\n"
    "1. A subcaption that describes a subfigure from a compound figure.\n"
    "2. The full caption of the compound figure.\n"
    "3. A context passage related to the compound figure.\n"
    "**Definition of compound figure:** A compound figure is a figure that contains multiple subfigures of the same topic (e.g., panels A, B, C, etc.).\n\n"
    "Your task is to summarize only the portions of the context passage "
    "that are most relevant to the given subcaption. The full caption \n"
    "is provided for additional information.\n"
    "The summary should:\n"
    "- Use both the subcaption and the full caption to determine context.\n"
    "- Be concise and focused on the subcaption's content.\n"
    "- Exclude unrelated information from the context passage.\n"
    "- Preserve key biomedical terminology exactly as it appears.\n"
    "- Output the summary only, without any labels or additional text in the following format:\n"
    "<summary>\n"
    "<YOUR SUMMARY OF THE CONTEXT PASSAGE RELEVANT TO THE SUBCAPTION AND FULL CAPTION>\n"
    "</summary>\n\n"
    "### INPUT:\n\n"
)


def build_chat(tokenizer, user_prompt: str, max_length: int = 32700):
    """
    Build chat-style input encoding for vLLM from user prompt.

    Args:
        tokenizer: The tokenizer to use.
        user_prompt (str): The user prompt string.
        max_length (int): Maximum token length for the input.

    Returns
    -------
        encoded inputs.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a biomedical image context summary generator.",
        },
        {"role": "user", "content": user_prompt},
    ]

    enc = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    token_ids = tokenizer.encode(enc, add_special_tokens=False)
    if len(token_ids) > max_length:
        enc = tokenizer.decode(token_ids[:max_length], skip_special_tokens=False)

    return enc


def _is_empty(x) -> bool:
    """
    Check if a response is empty (None, NaN, or empty string). Used to identify unprocessed rows.

    Args:
        x: The input to check.

    Returns
    -------
        bool: True if x is considered empty, False otherwise.
    """
    return (
        (x is None) or (isinstance(x, float) and pd.isna(x)) or (str(x).strip() == "")
    )


def _csv_overwrite(_df: pd.DataFrame, _path: str):
    """
    Safely overwrite a CSV file by writing to a temporary file first and then replacing the original.

    Args:
        _df (pd.DataFrame): DataFrame to save.
        _path (str): Path to the CSV file.
    """
    tmp = _path + ".tmp"
    _df.to_csv(tmp, index=False)
    os.replace(tmp, _path)


def process_data_batched_vllm(
    df: pd.DataFrame,
    llm: LLM,
    tokenizer,
    out_path: str,
    batch_size: int = 16,
    max_new_tokens: int = 192,
) -> None:
    """
    Process the DataFrame in batches using vLLM to generate summaries.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'caption', 'sub_caption', and 'image_context'.
        llm (LLM): The vLLM model instance.
        tokenizer: The tokenizer for building prompts.
        out_path (str): Path to save the output CSV.
        batch_size (int): Number of samples to process in each batch.
        max_new_tokens (int): Maximum number of new tokens to generate for each summary.
    """
    pattern = re.compile(
        r"<summary>\s*(.*?)\s*<\/summary>", re.DOTALL
    )  # Pattern to extract summary text

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens, temperature=0.0, top_p=1.0
    )
    t0_all = time.time()

    # Batch Processing Loop
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        idxs = [
            i for i in range(start, end) if _is_empty(df.loc[i, "summary"])
        ]  # Select unprocessed rows. This also allows resuming.
        if idxs:
            batch_prompts = []
            for i in idxs:
                # Prompt construction with full caption, subcaption, and context passage
                user_prompt = (
                    prompt
                    + f"Full Caption:\n{df.caption.iloc[i]}\n\n"
                    + f"Subcaption:\n{df.sub_caption.iloc[i]}\n\n"
                    + f"Context Passage:\n{df.image_context.iloc[i]}"
                )
                batch_prompts.append(build_chat(tokenizer, user_prompt))

            outs = llm.generate(batch_prompts, sampling_params)  # vLLM generation call

            for j, out in enumerate(outs):
                text = out.outputs[0].text
                m = pattern.search(text)
                df.loc[idxs[j], "summary"] = (
                    m.group(1).strip() if m else text.strip()
                )  # Extract summary or use full text if pattern not found

        # Overwrite CSV checkpoint every (batch size * 10) batches
        if start and (start % (10 * batch_size) == 0):
            _csv_overwrite(df, out_path)
            print(f"[ckpt] Saved at row {start} → {out_path}")

    # Final save after all batches are processed
    _csv_overwrite(df, out_path)
    print(f"Total time {time.time() - t0_all:.2f}s. Final saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data_path", required=True, help="CSV path to data")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path or HF id for the model (e.g., /model-weights/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="vLLM micro-batch size per generate() call",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=192, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="Tensor parallel size for vLLM"
    )
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.90,
        help="GPU memory utilization fraction for vLLM",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16"]
    )

    args = parser.parse_args()

    data_path = args.data_path
    model_dir = args.model_dir

    # Tokenizer used only to template chat → plain prompt string
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Init vLLM engine
    # Notes:
    # - tensor_parallel_size lets you span multiple GPUs if available.
    # - gpu_memory_utilization tunes how full vLLM packs the GPU.
    # - max_model_len can be set if you have very long contexts (defaults are fine for most).
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=None if args.dtype == "auto" else args.dtype,
    )

    df = pd.read_csv(data_path)  # Load input CSV

    process_data_batched_vllm(
        df=df,
        llm=llm,
        tokenizer=tokenizer,
        out_path=data_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Completed writing {len(df)} entries to: {data_path}")


if __name__ == "__main__":
    main()
