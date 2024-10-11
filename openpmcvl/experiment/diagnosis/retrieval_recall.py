"""Check Retrieval Recall results.

Compute Retrieval Recall results by running clip_benchmark's
retrieval pipeline on mmlearn-generated embeddings.
"""

import os
from typing import Any, Dict, List, Union

import torch
from lightning import LightningModule
from mmlearn.datasets.core.data_collator import DefaultDataCollator
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from mmlearn.modules.encoders.clip_encoders import (
    HFCLIPTextEncoderWithProjection,
    HFCLIPVisionEncoderWithProjection,
)
from mmlearn.modules.layers.logit_scaling import LearnableLogitScaling
from mmlearn.modules.layers.normalization import L2Norm
from mmlearn.modules.losses.contrastive_loss import CLIPLoss
from mmlearn.tasks.contrastive_pretraining import (
    ContrastivePretraining,
    EvaluationSpec,
    ModuleKeySpec,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpmcvl.experiment.configs import (
    biomedclip_vision_transform,
    med_clip_vision_transform,
)
from openpmcvl.experiment.datasets.mimiciv_cxr import MIMICIVCXR
from openpmcvl.experiment.modules.encoders import BiomedCLIPText, BiomedCLIPVision
from openpmcvl.experiment.modules.zero_shot_retrieval import (
    RetrievalTaskSpec,
    ZeroShotCrossModalRetrievalEfficient,
)


def instantiate_contrastive_pretraining_biomedclip() -> ContrastivePretraining:
    """Instantiate contrastive pretraining as in BiomedCLIP's experiment."""
    encoders = {
        "text": BiomedCLIPText(
            model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            pretrained=True,
            use_all_token_embeddings=False,
            freeze_layers=False,
            freeze_layer_norm=True,
            model_config_kwargs=None,
        ),
        "rgb": BiomedCLIPVision(
            model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            pretrained=True,
            use_all_token_embeddings=False,
            freeze_layers=False,
            freeze_layer_norm=True,
            model_config_kwargs=None,
        ),
    }
    heads = None
    postprocessors = {
        "norm_and_logit_scale": {
            "norm": L2Norm(dim=-1),
            "logit_scale": LearnableLogitScaling(
                logit_scale_init=14.285714285714285,
                learnable=True,
                max_logit_scale=100.0,
            ),
        }
    }
    modality_module_mapping = {
        "text": ModuleKeySpec(
            encoder_key=None, head_key=None, postprocessor_key="norm_and_logit_scale"
        ),
        "rgb": ModuleKeySpec(
            encoder_key=None, head_key=None, postprocessor_key="norm_and_logit_scale"
        ),
    }
    optimizer = None
    lr_scheduler = None
    loss = CLIPLoss(
        l2_normalize=False, local_loss=True, gather_with_grad=True, cache_labels=False
    )
    modality_loss_pairs = None
    auxiliary_tasks = None
    log_auxiliary_tasks_loss = False
    compute_validation_loss = True
    compute_test_loss = True
    evaluation_tasks = {
        "retrieval": EvaluationSpec(
            task=ZeroShotCrossModalRetrievalEfficient(
                task_specs=[
                    RetrievalTaskSpec(
                        query_modality="text",
                        target_modality="rgb",
                        top_k=[10, 50, 200],
                    ),
                    RetrievalTaskSpec(
                        query_modality="rgb",
                        target_modality="text",
                        top_k=[10, 50, 200],
                    ),
                ]
            ),
            run_on_validation=False,
            run_on_test=True,
        )
    }

    return ContrastivePretraining(
        encoders=encoders,
        heads=heads,
        postprocessors=postprocessors,
        modality_module_mapping=modality_module_mapping,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=loss,
        modality_loss_pairs=modality_loss_pairs,
        auxiliary_tasks=auxiliary_tasks,
        log_auxiliary_tasks_loss=log_auxiliary_tasks_loss,
        compute_validation_loss=compute_validation_loss,
        compute_test_loss=compute_test_loss,
        evaluation_tasks=evaluation_tasks,
    )


def instantiate_contrastive_pretraining_neurips() -> ContrastivePretraining:
    """Instantiate contrastive pretraining as in our NeurIPS paper's experiment."""
    encoders = {
        "text": HFCLIPTextEncoderWithProjection(
            model_name_or_path="openai/clip-vit-base-patch16",
            pretrained=True,
            use_all_token_embeddings=False,
            freeze_layers=False,
            freeze_layer_norm=True,
            peft_config=None,
            model_config_kwargs=None,
        ),
        "rgb": HFCLIPVisionEncoderWithProjection(
            model_name_or_path="openai/clip-vit-base-patch16",
            pretrained=True,
            use_all_token_embeddings=False,
            patch_dropout_rate=0.0,
            patch_dropout_shuffle=False,
            patch_dropout_bias=None,
            freeze_layers=False,
            freeze_layer_norm=True,
            peft_config=None,
            model_config_kwargs=None,
        ),
    }
    heads = None
    postprocessors = {
        "norm_and_logit_scale": {
            "norm": L2Norm(dim=-1),
            "logit_scale": LearnableLogitScaling(
                logit_scale_init=14.285714285714285,
                learnable=True,
                max_logit_scale=100.0,
            ),
        }
    }
    modality_module_mapping = {
        "text": ModuleKeySpec(
            encoder_key=None, head_key=None, postprocessor_key="norm_and_logit_scale"
        ),
        "rgb": ModuleKeySpec(
            encoder_key=None, head_key=None, postprocessor_key="norm_and_logit_scale"
        ),
    }
    optimizer = None
    lr_scheduler = None
    loss = CLIPLoss(
        l2_normalize=False, local_loss=True, gather_with_grad=True, cache_labels=False
    )
    modality_loss_pairs = None
    auxiliary_tasks = None
    log_auxiliary_tasks_loss = False
    compute_validation_loss = True
    compute_test_loss = True
    evaluation_tasks = {
        "retrieval": EvaluationSpec(
            task=ZeroShotCrossModalRetrievalEfficient(
                task_specs=[
                    RetrievalTaskSpec(
                        query_modality="text",
                        target_modality="rgb",
                        top_k=[10, 50, 200],
                    ),
                    RetrievalTaskSpec(
                        query_modality="rgb",
                        target_modality="text",
                        top_k=[10, 50, 200],
                    ),
                ]
            ),
            run_on_validation=False,
            run_on_test=True,
        )
    }

    task = ContrastivePretraining(
        encoders=encoders,
        heads=heads,
        postprocessors=postprocessors,
        modality_module_mapping=modality_module_mapping,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=loss,
        modality_loss_pairs=modality_loss_pairs,
        auxiliary_tasks=auxiliary_tasks,
        log_auxiliary_tasks_loss=log_auxiliary_tasks_loss,
        compute_validation_loss=compute_validation_loss,
        compute_test_loss=compute_test_loss,
        evaluation_tasks=evaluation_tasks,
    )

    # load a checkpoint
    ckpt_path = "/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt"
    state_dict = torch.load(ckpt_path, weights_only=True)
    task.load_state_dict(state_dict["state_dict"], strict=False)
    return task


def instantiate_mimic_biomedclip(
    batch_size: int, num_workers: int
) -> DataLoader[Dict[str, Any]]:
    """Instantiate the MIMIC-CXR test split and its dataloader."""
    # instantiate transform and tokenizer
    transform = biomedclip_vision_transform(image_crop_size=224, job_type="eval")
    tokenizer = HFTokenizer(
        model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=256,
        padding="max_length",
        truncation=True,
        clean_up_tokenization_spaces=False,
    )
    # instantiate dataset
    dataset = MIMICIVCXR(
        root_dir=os.getenv("MIMICIVCXR_ROOT_DIR", ""),
        split="test",
        labeler="double_image",
        transform=transform,
        tokenizer=tokenizer,
    )

    # instantiate data loader
    return DataLoader(
        dataset=dataset,
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
        pin_memory_device="",
    )


def instantiate_mimic_neurips(
    batch_size: int, num_workers: int
) -> DataLoader[Dict[str, Any]]:
    """Instantiate the MIMIC-CXR test split and its dataloader.

    Transform and tokenizer match `med_benchmarking` experiment.
    """
    # instantiate transform and tokenizer
    transform = med_clip_vision_transform(image_crop_size=224, job_type="eval")
    tokenizer = HFTokenizer(
        model_name_or_path="openai/clip-vit-base-patch16",
        max_length=77,
        padding="max_length",
        truncation=True,
        clean_up_tokenization_spaces=False,
    )
    # instantiate dataset
    dataset = MIMICIVCXR(
        root_dir=os.getenv("MIMICIVCXR_ROOT_DIR", ""),
        split="test",
        labeler="double_image",
        transform=transform,
        tokenizer=tokenizer,
    )

    # instantiate data loader
    return DataLoader(
        dataset=dataset,
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
        pin_memory_device="",
    )


def embed_data(
    loader: DataLoader[Dict[str, Any]], task: LightningModule
) -> Dict[str, torch.Tensor]:
    """Embed image and text using a given mmlearn model and dataset."""
    embeddings: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
        "text_embedding": [],
        "rgb_embedding": [],
    }
    assert isinstance(embeddings["text_embedding"], list)
    assert isinstance(embeddings["rgb_embedding"], list)
    for _, batch in tqdm(enumerate(loader), total=len(loader), desc="encoding"):
        outputs = task(batch)
        embeddings["text_embedding"].append(outputs["text_embedding"].detach().cpu())
        embeddings["rgb_embedding"].append(outputs["rgb_embedding"].detach().cpu())
    embeddings["text_embedding"] = torch.cat(embeddings["text_embedding"], axis=0).cpu()  # type: ignore[call-overload]
    embeddings["rgb_embedding"] = torch.cat(embeddings["rgb_embedding"], axis=0).cpu()  # type: ignore[call-overload]
    assert isinstance(embeddings["text_embedding"], torch.Tensor)
    assert isinstance(embeddings["rgb_embedding"], torch.Tensor)
    return embeddings


def save_embeddings(
    embeddings: Dict[str, torch.Tensor], filename: str = "./embeddings.pt"
) -> None:
    """Save text and rgb embeddings on disk."""
    torch.save(embeddings, filename)
    print(f"Saved embeddings in {filename}")


def load_embeddings(filename: str) -> Any:
    """Load embeddings from file."""
    return torch.load(filename, weights_only=True)


if __name__ == "__main__":
    # set params
    batch_size = 16
    num_workers = 2
    # load mimic data
    print("Instantiating MIMIC-CXR...")
    loader = instantiate_mimic_neurips(batch_size, num_workers)

    # instantiate contrastive pretraining
    print("Instantiating ContrastivePretraining...")
    task = instantiate_contrastive_pretraining_neurips()

    # extract image and text embedding
    print("Embedding text and images...")
    embeddings = embed_data(loader, task)

    # save embeddings on disk
    save_embeddings(
        embeddings, filename="openpmcvl/experiment/diagnosis/embeddings_neurips.pt"
    )
