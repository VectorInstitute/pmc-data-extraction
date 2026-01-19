#!/usr/bin/env python3
import os
import time
import argparse
import re
from tqdm import tqdm
from tqdm.auto import tqdm

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROMPT_MEDICAL_L2_ONLY = (
    "You are an expert in medical image modality classification. "
    "You are given a single image.\n\n"
    "Your task is to assign ONE fine-grained subclass label (L2) to the image.\n\n"
    "You must choose exactly ONE L2 label from the following allowed subclasses:\n"
    "- Radiology: [Ultrasound, Magnetic Resonance, Computerized Tomography, "
    "X-Ray, 2D Radiography, Angiography, PET, Combined modalities in one image]\n"
    "- Microscopy: [Light microscopy, Electron microscopy, Transmission microscopy, "
    "Fluorescence microscopy]\n"
    "- Visible Light Photography: [Dermatology, skin, Endoscopy, Other organs]\n"
    "- Other: [Other]\n\n"
    "If the image clearly does NOT belong to any medical modality above, choose \"Other\".\n"
    "If the image appears medical but you are unsure among subclasses, choose the most visually plausible one.\n\n"
    "OUTPUT FORMAT:\n"
    "Return your answer as a single JSON object with ONLY the L2 field:\n"
    "{\n"
    "  \"L2\": \"<one of the allowed subclasses above>\"\n"
    "}\n"
    "Do not include explanations, reasoning, or any additional text. Only output the JSON object."
)

# L2 Radiology label sets
L2_RADIOLOGY = {
    "ultrasound",
    "magnetic resonance",
    "computerized tomography",
    "x-ray",
    "2d radiography",
    "angiography",
    "pet",
    "combined modalities in one image",
}

# L2 Microscopy label sets
L2_MICROSCOPY = {
    "light microscopy",
    "electron microscopy",
    "transmission microscopy",
    "fluorescence microscopy",
}

# L2 Visible Light Photography label sets
L2_VLP = {
    "dermatology", "skin",
    "endoscopy",
    "other organs",
}

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# -------------------- Helpers --------------------
def _is_empty(x) -> bool:
    """
    Check if a response is empty (None, NaN, or empty string). Used to identify unprocessed rows.
    Args:
        x: The input to check.
    Returns:
        bool: True if x is considered empty, False otherwise.    
    """
    return x is None or (isinstance(x, float) and pd.isna(x)) or (str(x).strip() == "")

def _jsonl_overwrite(_df: pd.DataFrame, _path: str):
    """
    Safely overwrite a JSONL file by writing to a temporary file first and then replacing the original.
    Args:
        _df (pd.DataFrame): DataFrame to save.
        _path (str): Path to the JSONL file.
    """
    tmp = _path + ".tmp"
    _df.to_json(tmp, lines=True, orient="records")
    os.replace(tmp, _path)

def _load_rgb(path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to RGB mode if necessary.
    Args:
        path (str): Path to the image file.
    Returns:
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
    Returns:
        List[Dict[str, Any]]: The constructed message list.
    """
    
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

def extract_l2_label(text: str) -> Optional[str]:
    """
    Extract JSON {L2: "..."} from model text output.
    If parsing fails, return None.
    Args:
        text (str): The raw text output from the model.
    Returns:
        Optional[str]: The extracted L2 label, or None if parsing fails.
    """
    cleaned = text.strip()

    # strip Markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # keep only the JSON object part if there's extra text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        obj = json.loads(cleaned)
    except Exception:
        log.warning("JSON parse failed; storing raw text instead.")
        return None

    l2 = str(obj.get("L2") or obj.get("l2") or "").strip()
    return l2

def infer_from_l2(l2_raw: str) -> Tuple[str, str, str]:
    """
    Infer (L0, L1, L2) from an L2 string.
    L0 ∈ {Medical, Other}
    L1 ∈ {Radiology, Microscopy, Visible Light Photography, Other}
    L2 = original L2 text (possibly normalized upstream).
    Args:
        l2_raw (str): The raw L2 label.
    Returns:
        Tuple[str, str, str]: The inferred (L0, L1, L2) labels.
    """
    l2 = (l2_raw or "").strip()
    l2_norm = l2.lower()

    if l2_norm in L2_RADIOLOGY:
        l1 = "Radiology"
        l0 = "Medical"
    elif l2_norm in L2_MICROSCOPY:
        l1 = "Microscopy"
        l0 = "Medical"
    elif l2_norm in L2_VLP:
        l1 = "Visible Light Photography"
        l0 = "Medical"
    else:
        l1 = "Other"
        l0 = "Other"

    return l0, l1, l2

# -------------------- Batch processing --------------------
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
    Process the DataFrame in batches to generate modality labels using the provided vLLM model.
    Args:
        df (pd.DataFrame): Input DataFrame with image paths.
        llm (LLM): The vLLM model instance.
        processor: The processor for preparing inputs.
        out_path (str): Path to save the output CSV.
        batch_size (int): Number of samples to process in each batch.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.
    Returns:
        pd.DataFrame: The updated DataFrame with generated modality labels.
    """

    image_col = "subfig_path"
    label_cols = ["L0_label", "L1_label", "L2_label"]

    # ensure label columns exist, if not exist, create and store empty strings
    for col in label_cols:
        if col not in df.columns:
            df[col] = ""

    # Sampling parameters for generation. 
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    
    t0_all = time.time()
    n = len(df)
    total_loaded, total_failed, total_done = 0, 0, 0 # counters to track progress

    # rows needing inference = those with empty L0_label
    to_infer = sum(_is_empty(x) for x in df.get("L0_label", pd.Series([None] * n)))
    pbar = tqdm(total=to_infer, desc="inference", ncols=100, unit="img") # progress bar
    json_ok, json_fail = 0, 0

    log.info(f"Starting batched processing on {n:,} rows (to infer: {to_infer:,})")
    
    flag = False

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # Select unprocessed rows. This also allows resuming.
        idxs = [
            i for i in range(start, end)
            if any(_is_empty(df.at[i, col]) for col in label_cols)
        ]
        if not idxs:
            continue # skip if all rows in this batch are already processed

        t_img0 = time.time()
        requests = []
        idx_map = []

        # Load tqdm for progress tracking
        iterable = tqdm(
            idxs,
            desc=f"[prep] rows {start}-{end-1}",
            leave=False,
            ncols=100,
            unit="row",
        )

        batch_loaded, batch_failed = 0, 0

        # Prepare inputs for each row in the batch
        for i in iterable:
            img_path = str(df.at[i, image_col]) if image_col in df.columns else ""

            try:
                pil_img = _load_rgb(img_path)
                batch_loaded += 1
            except Exception as e:
                batch_failed += 1
                log.warning(f"Failed to load image at row {i}, path={img_path}: {e}")
                continue

            messages = build_messages(pil_img, PROMPT_MEDICAL_L2_ONLY) # Build vLLM message structure
            image_inputs, _videos = process_vision_info(messages) # Process images for vLLM using qwen_vl_utils's process_vision_info function.
            
            # Apply chat template to format the prompt correctly
            fprompt = processor.apply_chat_template( 
                messages, tokenize=False, add_generation_prompt=True
            )

            # Final request List for vLLM
            requests.append({
                "prompt": fprompt,
                "multi_modal_data": {"image": image_inputs},
            })
            idx_map.append(i)

        t_img = time.time() - t_img0
        total_loaded += batch_loaded
        total_failed += batch_failed

        log.info(
            f"[prep] batch {start}-{end-1}: loaded={batch_loaded}, "
            f"failed={batch_failed}, time={t_img:.2f}s"
        )

        if requests:
            t_gen0 = time.time()
            responses = llm.generate(requests, sampling) # vLLM generation call
            t_gen = time.time() - t_gen0

            # Process and store outputs
            for j, res in enumerate(responses):
                raw = res.outputs[0].text if res.outputs else ""
                l2_parsed = extract_l2_label(raw)

                if l2_parsed is not None:
                    l0, l1, l2 = infer_from_l2(l2_parsed)
                    json_ok += 1
                else:
                    # if JSON extraction fails, store full raw string in all labels
                    l0 = l1 = l2 = raw.strip()
                    json_fail += 1

                row_idx = idx_map[j]
                df.at[row_idx, "L0_label"] = l0
                df.at[row_idx, "L1_label"] = l1
                df.at[row_idx, "L2_label"] = l2

                pbar.update(1)

            total_done += len(responses)
            flag = True
            log.info(
                f"[gen ] batch {start}-{end-1}: outputs={len(responses)}, "
                f"time={t_gen:.2f}s | json_ok={json_ok}, json_fail={json_fail}"
            )

        # Checkpointing every 1000 batches
        if flag and start and ((start // batch_size) % 1000 == 0):
            _jsonl_overwrite(df, out_path)
            elapsed = time.time() - t0_all
            log.info(
                f"[ckpt] saved at row {start} → {out_path} | elapsed={elapsed/60:.1f}m | "
                f"done={total_done} | loaded={total_loaded} | failed_img={total_failed}"
            )
            flag = False

    # Final save after all batches are processed
    _jsonl_overwrite(df, out_path)
    pbar.close()
    log.info(
        f"Total time {time.time()-t0_all:.2f}s | done={total_done} | "
        f"loaded_img={total_loaded} | failed_img={total_failed} | "
        f"json_ok={json_ok} | json_fail={json_fail}. Final saved → {out_path}"
    )

    return df

# -------------------- Main --------------------
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", required=True, help="JSONL with column 'subfig_path'.")
    args.add_argument("--model_dir", default="Qwen/Qwen2.5-VL-32B-Instruct",
                      help="HF id or local path to Qwen2.5-VL-32B-Instruct")
    args.add_argument("--batch_size", type=int, default=8, help="Keep modest; VLMs are memory heavy")
    args.add_argument("--max_new_tokens", type=int, default=256)
    args.add_argument("--tp_size", type=int, default=4, help="Tensor parallel degree for 32B")
    args.add_argument("--gpu_mem_util", type=float, default=0.90)
    args.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16"])
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--top_p", type=float, default=1.0)

    args_dct = args.parse_args()

    log.info(f"Loading processor and model from {args_dct.model_dir}")
    processor = AutoProcessor.from_pretrained(args_dct.model_dir)
    llm = LLM(
        model=args_dct.model_dir,
        tensor_parallel_size=args_dct.tp_size,
        gpu_memory_utilization=args_dct.gpu_mem_util,
        dtype=None if args_dct.dtype == "auto" else args_dct.dtype,
    )

    log.info(f"Reading data from {args_dct.data_path}")
    df = pd.read_json(args_dct.data_path, lines=True)

    # Process in batches and generate modality labels
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

    log.info(f"Completed writing {len(df):,} rows → {args_dct.data_path}")

if __name__ == "__main__":
    main()


