## vLLM Inference Pipeline for Open-PMC-18M Subcaption, Image-context Summary generation, and Modality Labeling

This repo contains three vLLM inference stages, each launched via a Slurm bash script:

* **Stage 1 (Subcaption extraction, VLM):** `Qwen2.5-VL-32B-Instruct` generates a *verbatim* subfigure caption from a full figure caption + subfigure image. 
* **Stage 2 (Context summary, LLM):** `Qwen2.5-14B-Instruct` generates a focused summary of the context passage relevant to the subcaption. 
* **Stage 2 (Modality Labeling, VLM):** `Qwen2.5-VL-32B-Instruct` generates L2 labels, then L1 and L0 labels are inferred from a predefined set based on the generated L2 label. 

### Environment / Versions

This pipeline was run with:

* `vllm==0.8.2`
* `xformers==0.0.29.post2`
* `torch==2.6.0`

### Inputs

All scripts read and **overwrite** the same CSV or Jsonl (checkpointing is done by writing back to `--data_path`).

**Required columns**

* Subcaption stage (`generate_subcaption_vllm.py`):

  * `subfig_path` (path to subfigure image)
  * `caption` (full compound figure caption)
  * Output column: `sub_caption` 
* Summary stage (`generate_summary_vllm.py`):

  * `caption` (full compound figure caption)
  * `sub_caption` (subcaption for each subfigure)
  * `image_context` (image context related to subfigure)
  * Output column: `summary` 
* Modality Labeling stage (`generate_modality_labels_vllm.py`):

  * `subfig_path` (path to subfigure image)
  * Output column: `L0_label`, `L1_label`, and `L2_label` 

All stages support **resume** behavior: they skip rows where the output column is already filled (non-empty).

---

## How to Run (Slurm)

### 1) Subcaption generation (Qwen2.5-VL-32B-Instruct)

Edit the Slurm script to point to:

* your python file path
* your CSV path (`--data_path`)
* your model weights path (`--model_dir`)
* any desired batch/tp settings

Then submit:

```bash
sbatch run_vllm_subcaption_inference.sh
```

Slurm script reference: 

**What it does:** launches `generate_subcaption_vllm.py` with vLLM tensor parallelism and writes `sub_caption` back into the CSV.

---

### 2) Summary generation (Qwen2.5-14B-Instruct)

After Stage 1 finishes (CSV now has `sub_caption`), edit and submit:

```bash
sbatch run_vllm_summary_inference.sh
```

Slurm script reference: 

**What it does:** runs `generate_summary_vllm.py` and writes `summary` back into the same CSV.

---

### 3) Modality Label generation (Qwen2.5-VL-32B-Instruct)

Edit the Slurm script to point to:

* your python file path
* your CSV path (`--data_path`)
* your model weights path (`--model_dir`)
* any desired batch/tp settings

Then submit:

```bash
sbatch run_vllm_modality_inference.sh
```

Slurm script reference: 

**What it does:** runs `generate_modality_labels_vllm.py` and writes `L0`, `L1`, and `L2` labels back into the same jsonl file.

---

## Notes

* **Paths:** All Slurm scripts include placeholder paths like `/path/to/...` — replace them before submitting.
* **GPU selection:** All scripts set `CUDA_VISIBLE_DEVICES=0,1` and use `--tp_size 2` to shard across 2 GPUs.
* **Checkpointing:** All scripts allow periodic checkpointing. 
* **Outputs formatting:** subcaptions are extracted from `<caption>...</caption>`, and summaries from `<summary>...</summary>` (regex-based extraction).
