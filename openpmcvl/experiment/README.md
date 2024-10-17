## Benchmarking CLIP-style Methods on OpenPMC-VL Dataset

Please make sure to set the following environment variables:
```bash
export PMCVL_ROOT_DIR=/path/to/pmcvl/processed/
```
and activate the virtual environment where `mmlearn` is installed:
```bash
source /path/to/venv/bin/activate
```
Then, add the root directory of the repository to your `PYTHONPATH`:
```bash
cd root/of/repository
export PYTHONPATH="./"
```

To run an experiment (pretraining), use the following command:

**To Run Locally**:
```bash
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=vitb16_bert256_train_bs32 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    dataloader.train.batch_size=32 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2
```

**To Run on a SLURM Cluster**:
```bash
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip \
    experiment_name=vitb16_bert256_train_bs32 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    dataloader.train.batch_size=32 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2
```

### Evaluation

To run zero-shot retrieval evaluation on a pretrained model on the test split of OpenPMC-VL, use the following command:

**To Run Locally**:
```bash
mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip \
    experiment_name=biomedclip_retrieval_pmcvl \
    job_type=eval \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=0 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13571189/last.ckpt"
```

**To Run on a SLURM Cluster**:
```bash
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a100_arashaf \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip \
    experiment_name=biomedclip_retrieval_pmcvl \
    datasets.test.pmcvl.split=test_clean_1 \
    job_type=eval \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=0 \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13571189/last.ckpt"
```

### Evaluation with "med_benchmarking" models

To run zero-shot retrieval with `med_benchmarking` models (GPT/77 and ViT-B/16), use the following command:

**To Run Locally**:
```bash
mmlearn_run 'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' \
    +experiment=baseline \
    experiment_name=med_benchmarking_retrieval \
    job_type=eval \
    datasets@datasets.test=ROCO \
    datasets.test.split=test \
    datasets.test.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=77 \
    +datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    +datasets/transforms@datasets.test.transform=med_clip_vision_transform \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt
```

**To Run on a SLURM Cluster**:
```bash
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=900 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://projects.med_benchmarking.configs]' \
    +experiment=baseline \
    experiment_name=med_benchmarking_retrieval \
    job_type=eval \
    datasets@datasets.test=PMCVL \
    datasets.test.split=test_clean \
    datasets.test.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=77 \
    +datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    +datasets/transforms@datasets.test.transform=med_clip_vision_transform \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt
```

### Common Issues
* If your experiments run out of RAM (=cpu memory), try reducing the number of dataloader workers. The lowest possible value is `num_workers=0` which uses the least RAM and runs the slowest.
* Please submit your experiments to SLURM on a cpu node, not a gpu node. If you exist the gpu node after the experiment starts running, it might crash with an OOM error.
* Make sure the config package of your project is importable by running the following command:
```bash
python
import openpmcvl.experiment.configs
```
Moreover, check `pip freeze` and ensure that `mmlearn` is installed in your virtual environment.
* Make sure to set `strict_loading=False` when loading a checkpoint; currently, setting `strict_loading=True` raises an unnecessary error.
* You can find the full experiment configs in each experiment's output directory at `outputs/<experiment_name>/<date>/<time>/.hydra/config.yaml`. Alternatively, you can check out `mmlearn/conf/__init__.py` to see the definitions of all parts of experiment configs.

Hydra will compose the experiment configuration from all the configurations in the specified directory as well as all the
configurations in the `mmlearn` package. *Note the dot-separated path to the directory containing the experiment configuration
files.* Do not use `file://path/to/config/directory` notation since adding a searchpath with the `file://` notation does not run
the `__init__.py` file in `path/to/config/directory`, hence the configs defined in `__init__.py` will not be added to hydra's
external store.
