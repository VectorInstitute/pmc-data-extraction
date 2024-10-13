"""Tests for encoders added to openpmcvl project."""

import os

import torch
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

from openpmcvl.experiment.configs import biomedclip_vision_transform
from openpmcvl.experiment.modules.encoders import BiomedCLIPText, BiomedCLIPVision
from openpmcvl.experiment.modules.tokenizer import OpenClipTokenizerWrapper


def models_eq(model1, model2):
    """Return True if two pytorch models are equal.

    Equality means having the same named parameters and the same values for
    each parameter.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_model_impl():
    """Compare the model loaded via local implementation and open_clip."""
    # load the model via open_clip library
    model, _, _ = create_model_and_transforms(
        "hf-hub:microsoft/" "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # load the model via local implementation
    model_text = BiomedCLIPText(
        "microsoft/" "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=True
    )
    model_vision = BiomedCLIPVision(
        "microsoft/" "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=True
    )

    # compare
    assert models_eq(
        model_text, model.text
    ), "Text encoder is not equivalent to the official model"
    assert models_eq(
        model_vision, model.visual
    ), "Vision encoder is not equivalent to the official model"


def test_tokenizer_impl():
    """Compare the tokenizer loaded via local implementation and open_clip."""
    text = (
        "I'm a sample text used to test "
        "the implementation of the tokenizer compared to "
        "the official tokenizer loaded from open_clip library."
    )

    # load via open_clip
    tokenizer_og = get_tokenizer(
        "hf-hub:microsoft/" "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # load via wrapper in openpmcvl - config copy-pasted from biomedclip's HF
    print("Starting...")
    config = {
        "clean_up_tokenization_spaces": True,
        "cls_token": "[CLS]",
        "do_basic_tokenize": True,
        "do_lower_case": True,
        "mask_token": "[MASK]",
        "model_max_length": 1000000000000000019884624838656,
        "never_split": None,
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "strip_accents": None,
        "tokenize_chinese_chars": True,
        "tokenizer_class": "BertTokenizer",
        "unk_token": "[UNK]",
    }
    tokenizer_wr = OpenClipTokenizerWrapper(
        "hf-hub:microsoft/" "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        config=config,
    )

    # load via local implementation
    tokenizer = HFTokenizer(
        "microsoft/" "BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    # tokenize
    tokens_og = tokenizer_og([text])
    tokens_wr = tokenizer_wr([text])
    tokens = tokenizer([text])

    assert (
        tokens[Modalities.TEXT].shape == torch.Size([1, 256])
    ), f"Expected sequence length of 256 but received {tokens[Modalities.TEXT].shape[1]}"
    assert torch.equal(
        tokens_og, tokens_wr
    ), "Tokenizer doesn't match open_clip's implementation."
    assert torch.equal(
        tokens[Modalities.TEXT], tokens_og
    ), "Tokenizer doesn't match open_clip's implementation."


def test_img_transform():
    """Compare image transforms in local implementation and open_clip."""
    # load an image
    img_path = os.path.join(__file__, "../../figures/tiger.jpeg")
    with Image.open(img_path) as img:
        image = img.convert("RGB")

    # load transforms via open_clip
    _, preprocess_train, preprocess_val = create_model_and_transforms(
        "hf-hub:microsoft/" "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # load transforms via local implementation
    transform_train = biomedclip_vision_transform(image_crop_size=224, job_type="train")
    transform_val = biomedclip_vision_transform(image_crop_size=224, job_type="eval")

    if preprocess_train is not None:
        torch.manual_seed(0)
        image_train_og = preprocess_train(image)
    if transform_train is not None:
        torch.manual_seed(0)
        image_train = transform_train(image)
    if preprocess_val is not None:
        image_val_og = preprocess_val(image)
    if transform_val is not None:
        image_val = transform_val(image)

    assert torch.equal(
        image_train, image_train_og
    ), "Train image transforms don't match open_clip."
    assert torch.equal(
        image_val, image_val_og
    ), "Val image transforms don't match open_clip."


if __name__ == "__main__":
    test_tokenizer_impl()
    print("Passed")
