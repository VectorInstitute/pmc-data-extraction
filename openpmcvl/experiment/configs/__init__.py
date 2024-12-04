"""Project-specific configs to add the hydra's store."""

from typing import Any, Callable, Literal, TypeVar

from hydra_zen import builds
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from omegaconf import MISSING
from timm.data.transforms import ResizeKeepRatio
from torchvision import transforms

from openpmcvl.experiment.datasets.deepeyenet import DeepEyeNet
from openpmcvl.experiment.datasets.mimiciv_cxr import MIMICIVCXR
from openpmcvl.experiment.datasets.pmc2m_sum import PMC2MSum
from openpmcvl.experiment.datasets.pmcoa import PMCOA
from openpmcvl.experiment.datasets.pmcpatients import PMCPatients
from openpmcvl.experiment.datasets.pmcvl import PMCVL
from openpmcvl.experiment.datasets.quilt1m import Quilt
from openpmcvl.experiment.datasets.roco import ROCO
from openpmcvl.experiment.modules.contrastive_pretraining_ppr import (
    ContrastivePretrainingPPR,
)
from openpmcvl.experiment.modules.encoders import BiomedCLIPText, BiomedCLIPVision
from openpmcvl.experiment.modules.pmc_clip import (
    PmcClipText,
    PmcClipVision,
    pmc_clip_vision_transform,
)
from openpmcvl.experiment.modules.pubmedclip import (
    PubmedClipText,
    PubmedClipTokenizer,
    PubmedClipTransform,
    PubmedClipVision,
)
from openpmcvl.experiment.modules.scheduler import CosineAnnealingWarmupLR
from openpmcvl.experiment.modules.tokenizer import OpenClipTokenizerWrapper
from openpmcvl.experiment.modules.zero_shot_retrieval import (
    ZeroShotCrossModalRetrievalEfficient,
)


@external_store(group="datasets/transforms")
def med_clip_vision_transform(
    image_crop_size: int = 224, job_type: Literal["train", "eval"] = "train"
) -> transforms.Compose:
    """Return transforms for training/evaluating CLIP with medical images.

    Parameters
    ----------
    image_crop_size : int, default=224
        Size of the image crop.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    return transforms.Compose(
        [
            ResizeKeepRatio(  # type: ignore[no-untyped-call]
                512 if job_type == "train" else image_crop_size, interpolation="bicubic"
            ),
            transforms.RandomCrop(image_crop_size)
            if job_type == "train"
            else transforms.CenterCrop(image_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


@external_store(group="datasets/transforms")
def biomedclip_vision_transform(
    image_crop_size: int = 224, job_type: Literal["train", "eval"] = "train"
) -> transforms.Compose:
    """Return transforms for training/evaluating CLIP with medical images.

    Matching the transforms used in BiomedCLIP [1].

    Notes
    -----
    [1] https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

    Parameters
    ----------
    image_crop_size : int, default=224
        Size of the image crop.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    if job_type == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(image_crop_size, image_crop_size),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 4 / 3),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    size=image_crop_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                transforms.CenterCrop(image_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    return transform


external_store(
    HFTokenizer,
    name="BiomedCLIPTokenizer",
    group="datasets/tokenizers",
    model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    max_length=256,
    padding="max_length",
    truncation=True,
    clean_up_tokenization_spaces=False,
)

external_store(
    HFTokenizer,
    name="BiomedCLIPTokenizer512",
    group="datasets/tokenizers",
    model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    max_length=512,
    padding="max_length",
    truncation=True,
    clean_up_tokenization_spaces=False,
)

external_store(
    OpenClipTokenizerWrapper,
    name="BiomedCLIPTokenizerOG",
    group="datasets/tokenizers",
    model_name_or_path="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
)

external_store(
    HFTokenizer,
    name="PmcClipTokenizer",
    group="datasets/tokenizers",
    model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    max_length=77,
    padding="max_length",
    truncation=True,
    eturn_tensors="pt",
)

external_store(
    builds(
        CosineAnnealingWarmupLR,
        populate_full_signature=True,
        zen_partial=True,
        optimizer=MISSING,
        t_max=MISSING,
        warmup_length=0,
    ),
    name="CosineAnnealingWarmupLR",
    group="modules/lr_schedulers",
    provider="openpmcvl",
)

external_store(
    BiomedCLIPText,
    name="BiomedCLIPTextNormalized",
    group="modules/encoders",
    provider="openpmcvl",
    model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    normalize=True,
)

external_store(
    BiomedCLIPVision,
    name="BiomedCLIPVisionNormalized",
    group="modules/encoders",
    provider="openpmcvl",
    model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    normalize=True,
)

# add modalities for patient-to-patient retrieval
Modalities.register_modality(name="patient_q")
Modalities.register_modality(name="patient_t")
