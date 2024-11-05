"""Wrapper to load BiomedCLIP tokenizer from open_clip."""

from typing import Any, List, Union

from mmlearn.conf import external_store
from open_clip import get_tokenizer


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
