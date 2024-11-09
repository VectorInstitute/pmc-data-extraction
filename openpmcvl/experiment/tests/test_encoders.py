# """Tests for encoders added to openpmcvl project."""

# import os

# import torch
# from mmlearn.datasets.core import Modalities
# from mmlearn.datasets.processors.tokenizers import HFTokenizer
# from open_clip import create_model_and_transforms, get_tokenizer
# from PIL import Image
# from transformers import AutoModelForMaskedLM

# from openpmcvl.experiment.configs import biomedclip_vision_transform
# from openpmcvl.experiment.modules.encoders import BiomedCLIPText, BiomedCLIPVision
# from openpmcvl.experiment.modules.tokenizer import OpenClipTokenizerWrapper


# def models_eq(state_dict_1, state_dict_2):
#     """Return True if two pytorch model state dicts are equal.

#     Equality means having the same named parameters and the same values for
#     each parameter.
#     """
#     for p1, p2 in zip(state_dict_1.keys(), state_dict_2.keys()):
#         if p1 != p2:
#             print(f"Names of parameters {p1} and {p2} are not equal.")
#             return False
#         if not torch.equal(state_dict_1[p1], state_dict_2[p2]):
#             print(
#                 f"Values of parameters {p1} and {p2} are not equal:",
#                 "\n",
#                 state_dict_1[p1],
#                 "\n",
#                 state_dict_2[p2],
#             )
#             return False
#     return True


# def _remove_from_keys(dictionary, string):
#     """Remove a certain string from dictionary keys."""
#     clean_dict = {}
#     for key, value in dictionary.items():
#         clean_dict[key.replace(string, "")] = value
#     return clean_dict


# def _rename_key(dictionary, string1, string2):
#     """Replace a certain string with another one in dictionary keys."""
#     clean_dict = {}
#     for key, value in dictionary.items():
#         clean_dict[key.replace(string1, string2)] = value
#     return clean_dict


# def _remove_layer_from_state(dictionary, string):
#     """Remove keys that start with a given string."""
#     clean_dict = {}
#     for key, value in dictionary.items():
#         if not key.startswith(string):
#             clean_dict[key] = value
#     return clean_dict


# def test_model_impl_1():
#     """Compare the model loaded via local implementation and open_clip."""
#     # load the model via open_clip library
#     model, _, _ = create_model_and_transforms(
#         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#     )

#     # load the model via local implementation
#     model_text = BiomedCLIPText(
#         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=True
#     )
#     model_vision = BiomedCLIPVision(
#         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=True
#     )

#     # compare
#     model_state_text = model.text.state_dict()
#     model_state_vision = model.visual.state_dict()
#     model_text_state = _remove_from_keys(model_text.state_dict(), "model.")
#     model_vision_state = _remove_from_keys(model_vision.state_dict(), "model.")
#     assert models_eq(
#         model_text_state, model_state_text
#     ), "Text encoder is not equivalent to the official model"
#     assert models_eq(
#         model_vision_state, model_state_vision
#     ), "Vision encoder is not equivalent to the official model"


# def test_model_impl_2():
#     """Compare local impl. of the text encoder vs HF implementation.

#     Loads the text encoder without pretrained weights of BiomedCLIP;
#     hence, pretrained weights of BiomedBERT are loaded instead.
#     """
#     # load biomedclip config via open_clip
#     model_biomedclip, preprocess_train, preprocess_val = create_model_and_transforms(
#         "biomedclip"
#     )

#     # load biomedclip text encoder via mmlearn implementation
#     model_text_mmlearn = BiomedCLIPText(
#         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=False
#     )

#     # Load pubmedbert directly
#     model_pubmedbert = AutoModelForMaskedLM.from_pretrained(
#         "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
#     )

#     # compare
#     state_pubmedbert = _remove_layer_from_state(
#         _rename_key(model_pubmedbert.state_dict(), "bert", "transformer"), "cls."
#     )
#     state_biomedbert = _remove_layer_from_state(
#         _remove_layer_from_state(model_biomedclip.text.state_dict(), "pooler."), "proj."
#     )
#     state_mmlearn = _remove_from_keys(model_text_mmlearn.state_dict(), "model.")
#     assert models_eq(
#         state_pubmedbert, state_biomedbert
#     ), "Text encoder loaded via open_clip is not the same as PubmedBERT."
#     assert models_eq(
#         state_pubmedbert, state_mmlearn
#     ), "Text encoder loaded via mmlearn is not the same as PubmedBERT."


# def test_model_impl_3():
#     """Compare the model loaded via local implementation and open_clip.

#     Loads the text encoder without pretrained weights of BiomedCLIP;
#     hence, pretrained weights of BiomedBERT are loaded instead.
#     """
#     # load the model via open_clip library
#     torch.manual_seed(0)
#     model, _, _ = create_model_and_transforms("biomedclip")

#     # load the model via local implementation
#     torch.manual_seed(0)
#     model_text = BiomedCLIPText(
#         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=False
#     )
#     torch.manual_seed(0)
#     model_vision = BiomedCLIPVision(
#         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", pretrained=False
#     )

#     # compare
#     model_state_text = model.text.state_dict()
#     model_text_state = _remove_from_keys(model_text.state_dict(), "model.")
#     model_state_vision = model.visual.state_dict()
#     model_vision_state = _remove_from_keys(model_vision.state_dict(), "model.")
#     assert models_eq(
#         model_text_state, model_state_text
#     ), "Text encoder is not equivalent to the official model"
#     assert models_eq(
#         model_vision_state, model_state_vision
#     ), "Vision encoder is not equivalent to the official model"


# def test_tokenizer_impl_1():
#     """Compare the tokenizer loaded via local implementation and open_clip."""
#     text = (
#         "I'm a sample text used to test "
#         "the implementation of the tokenizer compared to "
#         "the official tokenizer loaded from open_clip library."
#     )

#     # load via open_clip
#     tokenizer_og = get_tokenizer(
#         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#     )

#     # load via wrapper in openpmcvl - config copy-pasted from biomedclip's HF
#     print("Starting...")
#     config = {
#         "clean_up_tokenization_spaces": True,
#         "cls_token": "[CLS]",
#         "do_basic_tokenize": True,
#         "do_lower_case": True,
#         "mask_token": "[MASK]",
#         "model_max_length": 1000000000000000019884624838656,
#         "never_split": None,
#         "pad_token": "[PAD]",
#         "sep_token": "[SEP]",
#         "strip_accents": None,
#         "tokenize_chinese_chars": True,
#         "tokenizer_class": "BertTokenizer",
#         "unk_token": "[UNK]",
#     }
#     tokenizer_wr = OpenClipTokenizerWrapper(
#         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#         config=config,
#     )

#     # load via local implementation
#     tokenizer = HFTokenizer(
#         "microsoft/" "BiomedNLP-BiomedBERT-base-uncased-abstract",
#         max_length=256,
#         padding="max_length",
#         truncation=True,
#     )

#     # tokenize
#     tokens_og = tokenizer_og([text])
#     tokens_wr = tokenizer_wr([text])
#     tokens = tokenizer([text])

#     assert (
#         tokens[Modalities.TEXT.name].shape == torch.Size([1, 256])
#     ), f"Expected sequence length of 256 but received {tokens[Modalities.TEXT.name].shape[1]}"
#     assert torch.equal(
#         tokens_og, tokens_wr
#     ), "Tokenizer doesn't match open_clip's implementation."
#     assert torch.equal(
#         tokens[Modalities.TEXT.name], tokens_og
#     ), "Tokenizer doesn't match open_clip's implementation."


# def test_tokenizer_impl_2():
#     """Compare the tokenizer loaded via local implementation and open_clip."""
#     text = (
#         "I'm a sample text used to test "
#         "the implementation of the tokenizer compared to "
#         "the official tokenizer loaded from open_clip library."
#     )

#     # load via open_clip
#     tokenizer_og = get_tokenizer(
#         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#     )
#     tokenizer = get_tokenizer("biomedclip_tokenizer")
#     print(tokenizer_og)
#     print(tokenizer)

#     # tokenize
#     tokens_og = tokenizer_og([text])
#     tokens = tokenizer([text])

#     assert torch.equal(
#         tokens_og, tokens
#     ), "Tokenizer doesn't match in biomedclip and pubmedbert."


# def test_img_transform():
#     """Compare image transforms in local implementation and open_clip."""
#     # load an image
#     img_path = os.path.join(__file__, "../../figures/tiger.jpeg")
#     with Image.open(img_path) as img:
#         image = img.convert("RGB")

#     # load transforms via open_clip
#     _, preprocess_train, preprocess_val = create_model_and_transforms(
#         "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#     )

#     # load transforms via open_clip without pretrained weights
#     _, preprocess_train_b, preprocess_val_b = create_model_and_transforms(
#         "biomedclip_tokenizer"
#     )

#     # load transforms via local implementation
#     transform_train = biomedclip_vision_transform(image_crop_size=224, job_type="train")
#     transform_val = biomedclip_vision_transform(image_crop_size=224, job_type="eval")

#     if preprocess_train is not None:
#         torch.manual_seed(0)
#         image_train_og = preprocess_train(image)
#     if preprocess_train_b is not None:
#         torch.manual_seed(0)
#         image_train_b = preprocess_train_b(image)
#     if transform_train is not None:
#         torch.manual_seed(0)
#         image_train = transform_train(image)
#     if preprocess_val is not None:
#         image_val_og = preprocess_val(image)
#     if preprocess_val_b is not None:
#         image_val_b = preprocess_val_b(image)
#     if transform_val is not None:
#         image_val = transform_val(image)

#     assert torch.equal(
#         image_train, image_train_og
#     ), "Train image transforms don't match open_clip."
#     assert torch.equal(
#         image_train, image_train_b
#     ), "Train image transforms don't match in open_clip and biomedclip config."
#     assert torch.equal(
#         image_val, image_val_og
#     ), "Val image transforms don't match open_clip."
#     assert torch.equal(
#         image_val, image_val_b
#     ), "Val image transforms don't match in open_clip and biomedclip config."


# if __name__ == "__main__":
#     test_img_transform()
