"""Tests for datasets added to openpmcvl project."""

import os

import torch
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from torch.utils.data.dataloader import DataLoader

from openpmcvl.experiment.configs import biomedclip_vision_transform
from openpmcvl.experiment.datasets.deepeyenet import DeepEyeNet
from openpmcvl.experiment.datasets.pmc2m_sum import PMC2MSum
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
        sample[Modalities.TEXT.name], str
    ), f"Expected to find `str` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert isinstance(
        sample[Modalities.RGB.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.RGB` but found {type(sample[Modalities.RGB.name])}"
    assert (
        sample[Modalities.RGB.name].size(0) == 3
    ), f"Expected `Modalities.RGB` to have 3 channels but found {sample[Modalities.RGB.name].size(0)}"

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
        sample[Modalities.TEXT.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert (
        sample[Modalities.TEXT.name].size() == torch.Size([256])
    ), f"Expected `Modalities.TEXT` to have shape {torch.Size([256])} but found {sample[Modalities.TEXT.name].size()}"
    assert (
        sample[Modalities.RGB.name].size() == torch.Size([3, 224, 224])
    ), f"Expected `Modalities.RGB` to have shape {torch.Size([3, 224, 224])} but found {sample[Modalities.RGB.name].size()}"


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
        sample[Modalities.TEXT.name], str
    ), f"Expected to find `str` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert isinstance(
        sample[Modalities.RGB.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.RGB` but found {type(sample[Modalities.RGB.name])}"
    assert (
        sample[Modalities.RGB.name].size(0) == 3
    ), f"Expected `Modalities.RGB` to have 3 channels but found {sample[Modalities.RGB.name].size(0)}"
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
        sample[Modalities.TEXT.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert (
        sample[Modalities.TEXT.name].size() == torch.Size([256])
    ), f"Expected `Modalities.TEXT` to have shape {torch.Size([256])} but found {sample[Modalities.TEXT.name].size()}"
    assert (
        sample[Modalities.RGB.name].size() == torch.Size([3, 224, 224])
    ), f"Expected `Modalities.RGB` to have shape {torch.Size([3, 224, 224])} but found {sample[Modalities.RGB.name].size()}"


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
        sample[Modalities.TEXT.name], str
    ), f"Expected to find `str` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert isinstance(
        sample[Modalities.RGB.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.RGB` but found {type(sample[Modalities.RGB.name])}"
    assert (
        sample[Modalities.RGB.name].size(0) == 3
    ), f"Expected `Modalities.RGB` to have 3 channels but found {sample[Modalities.RGB.name].size(0)}"

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
        sample[Modalities.TEXT.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert (
        sample[Modalities.TEXT.name].size() == torch.Size([256])
    ), f"Expected `Modalities.TEXT` to have shape {torch.Size([256])} but found {sample[Modalities.TEXT.name].size()}"
    assert (
        sample[Modalities.RGB.name].size() == torch.Size([3, 224, 224])
    ), f"Expected `Modalities.RGB` to have shape {torch.Size([3, 224, 224])} but found {sample[Modalities.RGB.name].size()}"


def test_pmc2m_sum():
    """Test PMC-2M with summarized inline references dataset."""
    root_dir = os.getenv("PMC2M_SUMM_ROOT_DIR", "")
    assert (
        root_dir is not None
    ), "Please set PMC2M-Sum root directory in `PMC2M_SUMM_ROOT_DIR` environment variable."

    # test without transform and tokenizer
    split = "test_clean_sep"
    transform = None
    tokenizer = None
    dataset = PMC2MSum(root_dir, split, transform, tokenizer)
    sample = dataset[0]
    print(f"sample: {sample}")
    assert isinstance(
        sample[Modalities.TEXT.name], str
    ), f"Expected to find `str` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert isinstance(
        sample[Modalities.RGB.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.RGB` but found {type(sample[Modalities.RGB.name])}"
    assert (
        sample[Modalities.RGB.name].size(0) == 3
    ), f"Expected `Modalities.RGB` to have 3 channels but found {sample[Modalities.RGB.name].size(0)}"

    # test with transform and tokenizer
    transform = biomedclip_vision_transform(image_crop_size=224, job_type="train")
    tokenizer = HFTokenizer(
        model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=512,
        padding="max_length",
        truncation=True,
        clean_up_tokenization_spaces=False,
    )
    dataset = PMC2MSum(root_dir, split, transform, tokenizer)
    sample = dataset[0]
    print(f"sample: {sample}")
    assert isinstance(
        sample[Modalities.TEXT.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.TEXT` but found {type(sample[Modalities.TEXT.name])}"
    assert (
        sample[Modalities.TEXT.name].size() == torch.Size([512])
    ), f"Expected `Modalities.TEXT` to have shape {torch.Size([512])} but found {sample[Modalities.TEXT.name].size()}"
    assert (
        sample[Modalities.RGB.name].size() == torch.Size([3, 224, 224])
    ), f"Expected `Modalities.RGB` to have shape {torch.Size([3, 224, 224])} but found {sample[Modalities.RGB.name].size()}"


def test_pmc2m_sum_2():
    """Test PMC-2M with summarized inline references dataset."""
    root_dir = os.getenv("PMC2M_SUMM_ROOT_DIR", "")
    assert (
        root_dir is not None
    ), "Please set PMC2M-Sum root directory in `PMC2M_SUMM_ROOT_DIR` environment variable."

    # test with transform and tokenizer and dataloader
    split = "test_clean"
    batch_size = 64
    transform = biomedclip_vision_transform(image_crop_size=224, job_type="train")
    tokenizer = HFTokenizer(
        model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=512,
        padding="max_length",
        truncation=True,
        clean_up_tokenization_spaces=False,
    )
    dataset = PMC2MSum(root_dir, split, transform, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    batch = next(iter(loader))
    assert isinstance(
        batch[Modalities.TEXT.name], torch.Tensor
    ), f"Expected to find `Tensor` in `Modalities.TEXT` but found {type(batch[Modalities.TEXT.name])}"
    assert (
        batch[Modalities.TEXT.name].size() == torch.Size([batch_size, 512])
    ), f"Expected `Modalities.TEXT` to have shape {torch.Size([batch_size, 512])} but found {batch[Modalities.TEXT.name].size()}"
    assert (
        batch[Modalities.RGB.name].size() == torch.Size([batch_size, 3, 224, 224])
    ), f"Expected `Modalities.RGB` to have shape {torch.Size([batch_size, 3, 224, 224])} but found {batch[Modalities.RGB.name].size()}"

    idx = 0
    for batch in loader:
        try:
            assert isinstance(batch["text"], torch.Tensor)
            idx += 1
        except Exception as e:
            print(f"Exception caught on batch #{idx}", e)
            print(f"batch index: {idx}")
            print(batch)
            break
    print(f"{idx} batches checked.")


if __name__ == "__main__":
    test_pmc2m_sum()
    test_pmc2m_sum_2()
    print("Passed")
