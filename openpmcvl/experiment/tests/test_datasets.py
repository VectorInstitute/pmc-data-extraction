"""Tests for datasets added to openpmcvl project."""

import torch
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer

from openpmcvl.experiment.configs import biomedclip_vision_transform
from openpmcvl.experiment.datasets.pmcoa import PMCOA


def test_pmcoa():
    """Test PMC-OA dataset."""
    # test without transform and tokenizer
    root_dir = "/projects/multimodal/datasets/pmc_oa/"
    split = "train"
    transform = None
    tokenizer = None
    dataset = PMCOA(root_dir, split, transform, tokenizer)
    sample = dataset[0]
    assert isinstance(
        sample[Modalities.TEXT], str
    ), f"Expected to find `str` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT])}"
    assert isinstance(
        sample[Modalities.RGB], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.RGB` but found {type(sample[Modalities.RGB])}"
    assert (
        sample[Modalities.RGB].size(0) == 3
    ), f"Expected `Modalities.RGB` to have 3 channels but found {sample[Modalities.RGB].size(0)}"

    # test with transform and tokenizer
    transform = biomedclip_vision_transform(image_crop_size=224, job_type="train")
    tokenizer = HFTokenizer(
        model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=256,
        padding="max_length",
        truncation=True,
        clean_up_tokenization_spaces=False,
    )
    dataset = PMCOA(root_dir, split, transform, tokenizer)
    sample = dataset[0]
    assert isinstance(
        sample[Modalities.TEXT], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT])}"
    assert (
        sample[Modalities.TEXT].size() == torch.Size([256])
    ), f"Expected `Modalities.TEXT` to have shape {torch.Size([256])} but found {sample[Modalities.TEXT].size()}"
    assert (
        sample[Modalities.RGB].size() == torch.Size([3, 224, 224])
    ), f"Expected `Modalities.RGB` to have shape {torch.Size([3, 224, 224])} but found {sample[Modalities.RGB].size()}"
