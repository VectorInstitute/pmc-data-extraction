"""Tests for encoders added to openpmcvl project."""

import math
import os

import torch
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from openpmcvl.experiment.configs import biomedclip_vision_transform
from openpmcvl.experiment.modules.encoders import BiomedCLIPText, BiomedCLIPVision
from openpmcvl.experiment.modules.pmc_clip import (
    ModifiedResNet,
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
        tokens[Modalities.TEXT.name].shape == torch.Size([1, 256])
    ), f"Expected sequence length of 256 but received {tokens[Modalities.TEXT.name].shape[1]}"
    assert torch.equal(
        tokens_og, tokens_wr
    ), "Tokenizer doesn't match open_clip's implementation."
    assert torch.equal(
        tokens[Modalities.TEXT.name], tokens_og
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


def get_encoder_outputs(encoder):
    """Return outputs of a given encoder inputting a dummy image-text pair.

    Parameters
    ----------
    encoder: nn.Module
        The encoder.

    Returns
    -------
    Tuple[torch.Tensor]
        The embeddings. Will be a tuple with a single element.
    """
    # load an image
    img_path = os.path.join(__file__, "../../figures/tiger.jpeg")
    with Image.open(img_path) as img:
        image = img.convert("RGB")
    # transform image
    transform_train = biomedclip_vision_transform(image_crop_size=224, job_type="train")
    image = transform_train(image)
    image = torch.stack([image, image])

    # declare and tokenize text
    text = (
        "I'm a sample text used to test "
        "the implementation of the tokenizer compared to "
        "the official tokenizer loaded from open_clip library."
    )
    tokenizer = HFTokenizer(
        "microsoft/" "BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    text = tokenizer([text, text])
    text = text[Modalities.TEXT.name]
    print(f"text input ids: {text}")
    print(f"text input ids.shape: {text.shape}")

    # create inputs dictionary
    inputs = {Modalities.RGB.name: image, Modalities.TEXT.name: text}

    return encoder(inputs)


def test_pmc_clip(pmc_clip_root=None):
    """Test local implementation of PMC-CLIP."""
    if pmc_clip_root is None:
        pmc_clip_root = os.getenv("PMC_CLIP_ROOT", "")

    # instantiate image encoder as described on PMC-CLIP repo
    image_encoder = ModifiedResNet(
        layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64
    )
    image_encoder.load_state_dict(
        torch.load(os.path.join(pmc_clip_root, "image_encoder_resnet50.pth"))
    )

    # instantiate image encoder locally
    image_encoder_ = PmcClipVision(pretrained=True, ckpt_dir=pmc_clip_root)

    # instantiate text encoder as described on PMC-CLIP repo
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    text_encoder.load_state_dict(
        torch.load(os.path.join(pmc_clip_root, "text_encoder.pth"))
    )
    text_projection_layer = torch.load(
        os.path.join(pmc_clip_root, "text_projection_layer.pth")
    )
    text_projection_layer = nn.Parameter(text_projection_layer)

    # instantiate text encoder locally
    text_encoder_ = PmcClipText(pretrained=True, ckpt_dir=pmc_clip_root)

    assert models_eq(
        image_encoder, image_encoder_
    ), "Image encoder implementations do not match."
    assert models_eq(
        text_encoder, text_encoder_.text_encoder
    ), "Text encoder implementations do not match."
    assert torch.equal(
        text_projection_layer, text_encoder_.text_projection_layer
    ), "Text projection layer implementations do not match."

    similarity = _text_pmc_clip_example(
        image_encoder, text_encoder, text_projection_layer
    )
    similarity_ = _text_pmc_clip_example(
        image_encoder_, text_encoder_, text_projection_layer=None
    )
    assert torch.equal(
        similarity, similarity_
    ), "Results of PMC-CLIP's example do not match."


def _text_pmc_clip_example(image_encoder, text_encoder, text_projection_layer=None):
    """Compute similarity with an example.

    Parameters
    ----------
    image_encoder: nn.Module
        Image encoder.
    text_encoder: nn.Module
        Text encoder.
    text_projection_layer: torch.Tensor, optional, default=None
        Text projection layer. If none is give, it is assumed that the projection layer
        is included inside `text_encoder`.
    """
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_encoder = image_encoder.to(device)
    text_encoder = text_encoder.to(device)
    if text_projection_layer is not None:
        text_projection_layer = text_projection_layer.to(device)

    # load image transform
    preprocess_val = pmc_clip_vision_transform(
        image_crop_size=224,
    )
    # load image
    pwd = os.path.dirname(os.path.realpath(__file__))
    image_path_ls = [
        os.path.join(pwd, "../figures/chest_X-ray.jpg"),
        os.path.join(pwd, "../figures/brain_MRI.jpg"),
    ]
    images = []
    image_tensor = []
    for image_path in image_path_ls:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        image_tensor.append(preprocess_val(image))
    image_tensor = torch.stack(image_tensor, dim=0).to(device)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    # load text
    bert_input = [
        "chest X-ray",
        "brain MRI",
    ]
    encoded_input = tokenizer(
        bert_input,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = encoded_input["input_ids"].to(device)

    # extract image feature
    inputs = (
        {"rgb": image_tensor, "text": input_ids}
        if text_projection_layer is None
        else image_tensor
    )
    image_feature = image_encoder(inputs)
    if isinstance(image_feature, dict):
        image_feature = image_feature["image_features"]
    if text_projection_layer is None:
        image_feature = image_feature[0]

    # extract text feature
    inputs = (
        {"rgb": image_tensor, "text": input_ids}
        if text_projection_layer is None
        else input_ids
    )
    text_feature = text_encoder(inputs)
    if text_projection_layer is None:
        text_feature = text_feature[0]
    else:
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ text_projection_layer

    # calculate similarity
    logit_scale = torch.tensor(4.4292)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    return (math.exp(logit_scale) * image_feature @ text_feature.T).softmax(dim=-1)


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute vector norm for PubmedClip prob computation.

    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    return torch.pow(sum_tensor, 0.5)


def test_pubmedclip():
    """Test local wrapper of PubmedClip."""
    # load image and text
    img_path = os.path.join(__file__, "../../figures/input.jpeg")
    image = Image.open(img_path).convert("RGB")
    text = ["Chest X-Ray", "Brain MRI", "Abdominal CT Scan"]

    # instantiate model and processor as described in original code
    processor = CLIPProcessor.from_pretrained(
        "flaviagiammarino/pubmed-clip-vit-base-patch32"
    )
    model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")

    # instantiate model and processor locally
    image_encoder = PubmedClipVision()
    text_encoder = PubmedClipText()

    # process image and text with og instantiation
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    # process image and text with local instantiation
    transform = PubmedClipTransform()
    tokenizer = PubmedClipTokenizer()
    pixel_values = transform(image).unsqueeze(0)
    tokens = tokenizer(text)
    inputs_ = {"rgb": pixel_values}
    inputs_.update(tokens)

    # compute probabilities with og model
    probs = model(**inputs).logits_per_image.softmax(dim=1).squeeze()

    # compute probabilities with local model
    image_embeds = image_encoder(inputs_)
    text_embeds = text_encoder(inputs_)
    image_embeds = image_embeds[0]
    text_embeds = text_embeds[0]
    # normalized features
    image_embeds = image_embeds / _get_vector_norm(image_embeds)
    text_embeds = text_embeds / _get_vector_norm(text_embeds)
    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()  # use the same logit scale
    logits_per_text = torch.matmul(
        text_embeds, image_embeds.t().to(text_embeds.device)
    ) * logit_scale.to(text_embeds.device)
    logits_per_image = logits_per_text.t()
    probs_ = logits_per_image.softmax(dim=1).squeeze()

    assert torch.equal(probs, probs_), "Final probabilities don't match in the example."
