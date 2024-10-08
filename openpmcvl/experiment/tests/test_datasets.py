"""Tests for datasets added to openpmcvl project."""

import os

import torch
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer

from openpmcvl.experiment.configs import biomedclip_vision_transform
from openpmcvl.experiment.datasets.deepeyenet import DeepEyeNet
from openpmcvl.experiment.datasets.pmcoa import PMCOA
from openpmcvl.experiment.datasets.quilt1m import Quilt


def test_pmcoa():
    """Test PMC-OA dataset."""
    root_dir = os.getenv("PMCOA_ROOT_DIR")
    assert (
        root_dir is not None
    ), "Please set PMCOA root directory in `PMCOA_ROOT_DIR` environment variable."

    # test without transform and tokenizer
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


def test_quilt():
    """Test Quilt-1M dataset."""
    root_dir = os.getenv("QUILT_ROOT_DIR")
    assert (
        root_dir is not None
    ), "Please set Quilt root directory in `QUILT_ROOT_DIR` environment variable."

    # test with all subsets and without transform and tokenizer
    split = "val"
    subsets = None
    transform = None
    tokenizer = None
    dataset = Quilt(root_dir, split, subsets, transform, tokenizer)
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
    assert (
        len(dataset) == 13559
    ), f"Expected 13559 entries in the dataset but found {len(dataset)}"

    # test with partial subsets
    subsets = ["openpath", "quilt", "laion"]
    dataset = Quilt(root_dir, split, subsets, transform, tokenizer)
    assert (
        len(dataset) == 11559
    ), f"Expected 11559 entries in three subsets but found {len(dataset)}"

    # test with transform and tokenizer
    transform = biomedclip_vision_transform(image_crop_size=224, job_type="train")
    tokenizer = HFTokenizer(
        model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=256,
        padding="max_length",
        truncation=True,
        clean_up_tokenization_spaces=False,
    )
    dataset = Quilt(root_dir, split, subsets, transform, tokenizer)
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


def test_deepeyenet():
    """Test DeepEyeNet dataset."""
    root_dir = os.getenv("DEY_ROOT_DIR")
    assert (
        root_dir is not None
    ), "Please set DeepEyeNet root directory in `DEY_ROOT_DIR` environment variable."

    # test without transform and tokenizer
    split = "test"
    transform = None
    tokenizer = None
    dataset = DeepEyeNet(root_dir, split, transform, tokenizer)
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
    dataset = DeepEyeNet(root_dir, split, transform, tokenizer)
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
