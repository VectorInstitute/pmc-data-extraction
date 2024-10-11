"""Check Retrieval Recall results.

Compute Retrieval Recall results by running clip_benchmark's retrieval pipeline on mmlearn-generated embeddings.
"""
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from openpmcvl.experiment.datasets.mimiciv_cxr import MIMICIVCXR
from openpmcvl.experiment.configs import biomedclip_vision_transform
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.data_collator import DefaultDataCollator
from mmlearn.tasks.contrastive_pretraining import ContrastivePretraining, ModuleKeySpec, EvaluationSpec
from openpmcvl.experiment.modules.encoders import BiomedCLIPText, BiomedCLIPVision
from mmlearn.modules.layers.normalization import L2Norm
from mmlearn.modules.layers.logit_scaling import LearnableLogitScaling
from mmlearn.modules.losses.contrastive_loss import CLIPLoss
from openpmcvl.experiment.modules.zero_shot_retrieval import ZeroShotCrossModalRetrievalEfficient, RetrievalTaskSpec


def instantiate_contrastive_pretraining():
    """Instantiate contrastive pretraining as in BiomedCLIP's experiment."""
    encoders = {"text": BiomedCLIPText(
                    model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    pretrained=True,
                    use_all_token_embeddings=False,
                    freeze_layers=False,
                    freeze_layer_norm=True,
                    model_config_kwargs=None),
                "rgb": BiomedCLIPVision(
                    model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    pretrained=True,
                    use_all_token_embeddings=False,
                    freeze_layers=False,
                    freeze_layer_norm=True,
                    model_config_kwargs=None),
                }
    heads = None
    postprocessors = {"norm_and_logit_scale":
                      {"norm": L2Norm(dim=-1),
                       "logit_scale": LearnableLogitScaling(logit_scale_init=14.285714285714285,
                                                            learnable=True,
                                                            max_logit_scale=100.0)}}
    modality_module_mapping = {"text": ModuleKeySpec(encoder_key=None,
                                                     head_key=None,
                                                     postprocessor_key="norm_and_logit_scale"),
                                "rgb": ModuleKeySpec(encoder_key=None,
                                                     head_key=None,
                                                     postprocessor_key="norm_and_logit_scale"),
                            }
    optimizier = None
    lr_scheduler = None
    loss = CLIPLoss(l2_normalize=False,
                    local_loss=True,
                    gather_with_grad=True,
                    cache_labels=False)
    modality_loss_pairs = None
    auxiliary_tasks = None
    log_auxiliary_tasks_loss = False
    compute_validation_loss = True
    compute_test_loss = True
    evaluation_tasks = {"retrieval": EvaluationSpec(task=ZeroShotCrossModalRetrievalEfficient(
                                      task_specs=[RetrievalTaskSpec(query_modality="text", target_modality="rgb", top_k=[10, 50, 200]),
                                                  RetrievalTaskSpec(query_modality="rgb", target_modality="text", top_k=[10, 50, 200])]
                                       ),
                                      run_on_validation=False,
                                      run_on_test=True)}

    return ContrastivePretraining(encoders=encoders,
                                  heads=heads,
                                  postprocessors=postprocessors,
                                  modality_module_mapping=modality_module_mapping,
                                  optimizer=optimizier,
                                  lr_scheduler=lr_scheduler,
                                  loss=loss,
                                  modality_loss_pairs=modality_loss_pairs,
                                  auxiliary_tasks=auxiliary_tasks,
                                  log_auxiliary_tasks_loss=log_auxiliary_tasks_loss,
                                  compute_validation_loss=compute_validation_loss,
                                  compute_test_loss=compute_test_loss,
                                  evaluation_tasks=evaluation_tasks)


def instantiate_mimic(batch_size, num_workers):
    """Instantiate the MIMIC-CXR test split and its dataloader."""
    # instantiate transform and tokenizer
    transform = biomedclip_vision_transform(image_crop_size=224, job_type="eval")
    tokenizer = HFTokenizer(model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                            max_length=256,
                            padding="max_length",
                            truncation=True,
                            clean_up_tokenization_spaces=False)
    # instantiate dataset
    dataset = MIMICIVCXR(root_dir=os.getenv("MIMICIVCXR_ROOT_DIR"),
                         split="test",
                         labeler="double_image",
                         transform=transform,
                         tokenizer=tokenizer)

    # instantiate data loader
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=None,
                      sampler=None,
                      batch_sampler=None,
                      num_workers=num_workers,
                      collate_fn=DefaultDataCollator(),
                      pin_memory=True,
                      drop_last=False,
                      timeout=0.0,
                      worker_init_fn=None,
                      multiprocessing_context=None,
                      generator=None,
                      prefetch_factor=None,
                      persistent_workers=False,
                      pin_memory_device="")


def embed_data(loader, task):
    """Embed image and text using a given mmlearn model and dataset."""
    embeddings = {"text_embedding": [], "rgb_embedding": []}
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="encoding"):
        outputs = task(batch)
        embeddings["text_embedding"].append(outputs["text_embedding"].detach().cpu())
        embeddings["rgb_embedding"].append(outputs["rgb_embedding"].detach().cpu())
    embeddings["text_embedding"] = torch.cat(embeddings["text_embedding"], axis=0).cpu()
    embeddings["rgb_embedding"] = torch.cat(embeddings["rgb_embedding"], axis=0).cpu()
    return embeddings


def save_embeddings(embeddings, root_dir="./"):
    """Save text and rgb embeddings on disk."""
    torch.save(embeddings, os.path.join(root_dir, "embeddings.pt"))
    print(f"Saved embeddings in {os.path.join(root_dir, 'embeddings.pt')}")


def load_embeddings(filename):
    """Load embeddings from file."""
    return torch.load(filename, weights_only=True)


if __name__ == "__main__":
    # set params
    batch_size = 16
    num_workers = 2
    # load mimic data
    print("Instantiating MIMIC-CXR...")
    loader = instantiate_mimic(batch_size, num_workers)

    # instantiate contrastive pretraining
    print("Instantiating ContrastivePretraining...")
    task = instantiate_contrastive_pretraining()

    # extract image and text embedding
    print("Embedding text and images...")
    embeddings = embed_data(loader, task)

    # save embeddings on disk
    save_embeddings(embeddings, root_dir="openpmcvl/experiment/diagnosis/")