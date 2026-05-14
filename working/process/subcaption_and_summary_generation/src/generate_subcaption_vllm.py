#!/usr/bin/env python3
import argparse
import os
import re
import time
from typing import Any, Dict, List

import pandas as pd
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

prompt = (
    "### INSTRUCTIONS:\n"
    "You are an expert medical image captioning assistant. Your task is the following:\n"
    "1. You will be provided with a subfigure image that is part of a full image figure and the full figure caption in the input.\n"
    "2. The full caption contains descriptions for multiple subfigures (e.g., Subfigure-A, Subfigure-B, etc.).\n"
    "3. Your task is to identify the relevant subfigure caption corresponding to the provided subfigure image from the full caption exactly as it appears.\n"
    "4. If the subcaption is written jointly for two or more subfigures (e.g., A–C together, (A–C), Axial (A) and coronal (B), etc.), copy that combined description exactly as it appears.\n"
    "5. Do NOT rewrite, summarize, or generate new text. Copy the relevant portion exactly as it appears in the full caption.\n"
    "6. Here, 'exactly as it appears' mean the extracted caption must match word-for-word, character-for-character with the correct subfigure caption text from the full caption. It must be a verbatim copy, not paraphrased, summarized, or partially copied.\n"
    "7. If no relevant caption is found in the full caption, output the verbatim copy of the entire full caption.\n"
    "### OUTPUT FORMAT:\n"
    "<caption>\n"
    "<EXTRACTED SUBFIGURE CAPTION OR VERBATIM FULL CAPTION>\n"
    "</caption>\n\n"
    "### INPUT:\n\n"
)


def _is_empty(x) -> bool:
    """
    Check if a response is empty (None, NaN, or empty string). Used to identify unprocessed rows.

    Args:
        x: The input to check.

    Returns
    -------
        bool: True if x is considered empty, False otherwise.
    """
    return x is None or (isinstance(x, float) and pd.isna(x)) or (str(x).strip() == "")


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


def _load_rgb(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to RGB mode if necessary.

    Args:
        path (str): Path to the image file.

    Returns
    -------
        Image.Image: The loaded RGB image.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_messages(img: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    """
    Build the message structure for the vLLM compatible VLM input.

    Args:
        img (Image.Image): The input image.
        prompt (str): The text prompt.

    Returns
    -------
        List[Dict[str, Any]]: The constructed message list.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    return messages


def process_batched(
    df: pd.DataFrame,
    llm: LLM,
    processor,
    out_path: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> pd.DataFrame:
    """
    Process the DataFrame in batches to generate subcaptions using the provided vLLM model.

    Args:
        df (pd.DataFrame): Input DataFrame with image paths and captions.
        llm (LLM): The vLLM model instance.
        processor: The processor for preparing inputs.
        out_path (str): Path to save the output CSV.
        batch_size (int): Number of samples to process in each batch.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.

    Returns
    -------
        pd.DataFrame: The updated DataFrame with generated subcaptions.
    """
    image_col = "subfig_path"
    output_col = "sub_caption"

    # Sampling parameters for generation. Stop at </caption>.
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["</caption>"],
    )

    pattern = re.compile(
        r"<caption>\s*(.*?)\s*</caption>", re.DOTALL
    )  # to extract text within <caption> tags

    t0_all = time.time()
    n = len(df)
    total_loaded, total_failed, total_done = 0, 0, 0  # counters to track progress

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        idxs = [
            i for i in range(start, end) if _is_empty(df.at[i, output_col])
        ]  # Select unprocessed rows. This also allows resuming.
        if not idxs:
            continue  # skip if all rows in this batch are already processed

        t_img0 = time.time()
        requests = []
        idx_map = []

        # Load tqdm for progress tracking
        iterable = tqdm(
            idxs,
            desc=f"[prep] rows {start}-{end - 1}",
            leave=False,
            ncols=100,
            unit="row",
        )

        batch_loaded, batch_failed = 0, 0  # counters to track batch progress

        # Prepare inputs for each row in the batch
        for i in iterable:
            img_path = str(df.at[i, image_col]) if image_col in df.columns else ""
            text = f"{prompt}\n\n##Full Caption:\n{df.caption.iloc[i]}"  # Final text prompt containing full caption

            try:
                pil_img = _load_rgb(img_path)
                batch_loaded += 1
            except Exception:
                batch_failed += 1
                continue

            messages = build_messages(pil_img, text)  # Build vLLM message structure
            image_inputs, _videos = process_vision_info(
                messages
            )  # Process images for vLLM using qwen_vl_utils's process_vision_info function.

            # Apply chat template to format the prompt correctly
            fprompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Final request List for vLLM
            requests.append(
                {
                    "prompt": fprompt,
                    "multi_modal_data": {"image": image_inputs},
                }
            )
            idx_map.append(i)

        t_img = time.time() - t_img0
        total_loaded += batch_loaded
        total_failed += batch_failed

        print(
            f"[prep] batch {start}-{end - 1}: loaded={batch_loaded}, failed={batch_failed}, time={t_img:.2f}s"
        )

        if requests:
            t_gen0 = time.time()
            responses = llm.generate(requests, sampling)  # vLLM generation call
            t_gen = time.time() - t_gen0

            # Process and store outputs
            for j, res in enumerate(responses):
                out = res.outputs[0].text if res.outputs else ""
                m = pattern.search(out)
                df.at[idx_map[j], output_col] = (
                    m.group(1).strip() if m else out.replace("<caption>", "").strip()
                )  # Strip of extra caption tags if regex fails.

            total_done += len(responses)
            print(
                f"[gen ] batch {start}-{end - 1}: outputs={len(responses)}, time={t_gen:.2f}s"
            )

        # Checkpointing every 10 batches
        if start and ((start // batch_size) % 10 == 0):
            _csv_overwrite(df, out_path)
            elapsed = time.time() - t0_all
            print(
                f"[ckpt] saved at row {start} → {out_path} | elapsed={elapsed / 60:.1f}m | "
                f"done={total_done} | loaded={total_loaded} | failed={total_failed}"
            )

    # Final save after all batches are processed
    _csv_overwrite(df, out_path)
    print(
        f"Total time {time.time() - t0_all:.2f}s | done={total_done} | loaded={total_loaded} | failed={total_failed}. "
        f"Final saved → {out_path}"
    )
    return df


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data_path",
        required=True,
        help="CSV with at least two columns: image path + full caption.",
    )
    args.add_argument(
        "--model_dir",
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="HF id or local path to Qwen2.5-VL-32B-Instruct",
    )
    args.add_argument(
        "--batch_size", type=int, default=8, help="Keep modest; VLMs are memory heavy"
    )
    args.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max tokens to generate"
    )
    args.add_argument(
        "--tp_size",
        type=int,
        default=4,
        help="Tensor parallel degree for 32B (e.g., 4×A100-80GB)",
    )
    args.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.90,
        help="GPU memory utilization for vLLM",
    )
    args.add_argument(
        "--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16"]
    )
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--top_p", type=float, default=1.0)

    args_dct = args.parse_args()

    processor = AutoProcessor.from_pretrained(args_dct.model_dir)
    llm = LLM(
        model=args_dct.model_dir,
        tensor_parallel_size=args_dct.tp_size,
        gpu_memory_utilization=args_dct.gpu_mem_util,
        dtype=None if args_dct.dtype == "auto" else args_dct.dtype,
    )

    df = pd.read_csv(args_dct.data_path)  # Load input CSV

    # Process in batches and generate subcaptions
    df = process_batched(
        df=df,
        llm=llm,
        processor=processor,
        out_path=args_dct.data_path,
        batch_size=args_dct.batch_size,
        max_new_tokens=args_dct.max_new_tokens,
        temperature=args_dct.temperature,
        top_p=args_dct.top_p,
    )

    print(f"Completed writing {len(df)} rows → {args_dct.data_path}")


if __name__ == "__main__":
    main()
