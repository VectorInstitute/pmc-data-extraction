import json

import numpy as np
import open_clip
import torch
import transformers
from huggingface_hub import hf_hub_download
from mmlearn.datasets.core.modalities import Modalities
from mmlearn.datasets.processors.tokenizers import HFTokenizer
from mmlearn.hf_utils import load_huggingface_model
from mmlearn.modules.encoders.clip_encoders import \
    HFCLIPTextEncoderWithProjection
from open_clip.model import CustomTextCLIP


def load_via_open_clip():
    """Testing how to load BiomedCLIP via open_clip library.

    This is the recommended way to load the model.
    """
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # print(model)
    # print(tokenizer)

    return model, preprocess_train, preprocess_val, tokenizer


def load_checkpoint(
    model: CustomTextCLIP,
    checkpoint_path: str,
    strict: bool = True,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # if next(iter(state_dict.items()))[0].startswith('module'):
    #     state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = "text.transformer.embeddings.position_ids"
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def load_via_from_pretrained():
    """Load BiomedCLIP via huggingface's from_pretrained() method.

    This is a middle step to ultimately use mmlearn's hf_utils.
    """
    # load checkpoint file
    cached_file = hf_hub_download(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "open_clip_pytorch_model.bin",
        revision=None,
        cache_dir=None,
    )
    print("cached_file:\n", cached_file)

    # load model configs
    config_path = hf_hub_download(
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "open_clip_config.json",
        cache_dir=None,
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_cfg = config["model_cfg"]
    print("config:\n", config)

    # create model and load checkpoint
    model = CustomTextCLIP(**model_cfg)
    load_checkpoint(model, cached_file)

    return model


def models_eq(model1, model2):
    """Return True if two pytorch models are equal."""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


# This didn't work
def load_via_mmlearn_hf_utils():
    model = load_huggingface_model(
        transformers.CLIPTextModelWithProjection,
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        load_pretrained_weights=False,
    )
    print(model)


def load_via_mmlearn():
    model_kwargs = {
        "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "hf_proj_type": "mlp",
        "hf_pooler_type": "cls_last_hidden_state_pooler",
        "context_length": 256,
    }
    model_text = HFCLIPTextEncoderWithProjection(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        hf_tokenizer_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        hf_proj_type="mlp",
        hf_pooler_type="cls_last_hidden_state_pooler",
        context_length=256,
    )
    print(model_text)
    return model_text


def load_comapre_tokenizers():
    tokenizer1 = HFTokenizer(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    tokenizer2 = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    ids = "This is a very importatant random test, Ok."
    tokens1 = tokenizer1(ids)
    tokens1 = tokens1[Modalities.TEXT.name]
    tokens2 = tokenizer2(ids).reshape(-1)
    print("tokens1:\n", tokens1)
    print("tokens2:\n", tokens2)
    print("Tokens Equal:", torch.equal(tokens1, tokens2))


def load_preprocess_via_mmlearn():
    return


if __name__ == "__main__":
    model1, preprocess_train, preprocess_val, tokenizer = load_via_open_clip()
    model2 = load_via_from_pretrained()

    load_preprocess_via_mmlearn()

    # model1 = model1.visual
    # model2 = model2.visual
    # model1 = model1.text
    # model2 = model2.text

    # # validate models
    # state1 = model1.state_dict().keys()
    # state2 = model2.state_dict().keys()

    # flag = 0
    # for k1, k2 in zip(state1, state2):
    #     if k1 != k2:
    #         flag = 1
    # if flag == 1:
    #     print("Keys Equal: False")
    # else:
    #     print("Keys Equal: True")

    # print("Sanity (Must be True):", models_eq(model1, model1))
    # print("Models Equal:", models_eq(model1, model2))

    # load_via_mmlearn()

    # load_comapre_tokenizers()

# {'hf_model_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
#  'hf_tokenizer_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
#  'hf_proj_type': 'mlp',
#  'hf_pooler_type': 'cls_last_hidden_state_pooler',
#  'context_length': 256}
