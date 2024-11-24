"""Wrapper to load BiomedCLIP tokenizer from open_clip."""

from typing import Any, List, Union, Dict

from mmlearn.conf import external_store
from open_clip import get_tokenizer
import torch
from transformers import AutoTokenizer
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality


@external_store(group="datasets/tokenizers", provider="openpmcvl")
class OpenClipTokenizerWrapper:
    """Wrapper to load tokenizer using open_clip."""

    def __init__(
        self,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the model."""
        super().__init__()

        # load via open_clip
        self.tokenizer = get_tokenizer(model_name_or_path, **kwargs)

    def __call__(self, x: Union[str, List[str]]) -> Any:
        """Pass any input to loaded tokenizer."""
        return self.tokenizer(x)


@external_store(
    group="datasets/tokenizers",
    provider="openpmcvl",
    model_name_or_path="google/bigbird-pegasus-large-pubmed",
)
class BigBirdTokenizerWrapper:
    """Wrapper for the Big Bird tokenizer.

    Parameters
    ----------
    model_name_or_path : str
        The Hugging Face model name or a local path from which to load the tokenizer.
    max_length : int, default=512
        The maximum sequence length for the tokenizer.
    """

    def __init__(self, model_name_or_path: str = "google/bigbird-pegasus-large-pubmed", max_length: int = 512) -> None:
        """Initialize the tokenizer."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length

    def __call__(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize input texts.

        Parameters
        ----------
        texts : Union[str, List[str]]
            The input text or a list of texts to tokenize.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing tokenized inputs with keys `input_ids` and `attention_mask`.
        """
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            Modalities.TEXT.name: tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
