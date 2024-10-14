"""Classify image into modalities given in [1] using retrieval.

For each image, the similarity of the image embedding with all labels
in [1] is computed and the highly similar labels are retrieved.

References
----------
[1] Garcia Seco de Herrera, A., Muller, H. & Bromuri, S.
    "Overview of the ImageCLEF 2015 medical classification task."
    In Working Notes of CLEF 2015 (Cross Language Evaluation Forum) (2015).
"""
import ast
from typing import Any, Dict, List, Union, Optional, Callable
import logging
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import json

import lightning as L  # noqa: N812
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers.wandb import WandbLogger
from transformers.utils.import_utils import is_torch_tf32_available

from mmlearn.cli._instantiators import instantiate_datasets, instantiate_loggers
from mmlearn.conf import hydra_main
from mmlearn.datasets.core import *  # noqa: F403
from mmlearn.datasets.processors import *  # noqa: F403
from mmlearn.modules.encoders import *  # noqa: F403
from mmlearn.modules.layers import *  # noqa: F403
from mmlearn.modules.losses import *  # noqa: F403
from mmlearn.modules.lr_schedulers import *  # noqa: F403
from mmlearn.modules.metrics import *  # noqa: F403
from mmlearn.tasks import *  # noqa: F403


logger = logging.getLogger(__package__)


class ModalityClassifier(nn.Module):
    """Classify the modality of an image-text pair by retrieval."""

    def __init__(self, model: L.LightningModule,
                 loader: DataLoader,
                 tokenizer: Callable[[Union[str, List[str]]], Union[torch.Tensor, Dict[str, Any]]],
                 keywords: Optional[List[str]] = None):
        """Initialize the module.

        Parameters
        ----------
        model: L.LightningModule
            Task module loaded from an mmlearn experiment.
        loader: DataLoader
            Data loader.
        keywords: List[str], optional, default=None
            List of modality keywords. If none is given, keywords in [1] are used.

        References
        ----------
        [1] Garcia Seco de Herrera, A., Muller, H. & Bromuri, S.
            "Overview of the ImageCLEF 2015 medical classification task."
            In Working Notes of CLEF 2015 (Cross Language Evaluation Forum) (2015).
        """
        super().__init__()
        self.model = model
        self.loader = loader
        self.tokenizer = tokenizer
        if keywords is not None:
            self.keywords = keywords
        else:
            self.keywords = self._default_keywords()

    def _default_keywords(self):
        """Default modality keywords."""
        return ["radiology",
                "ultrasound",
                "magnetic resonance",
                "computerized tomography",
                "x-ray",
                "angiography",
                "pet",
                "visible light photography",
                "endoscopy",
                "electroencephalography",
                "electrocardiography",
                "electromyography",
                "microscopy",
                "gene sequence",
                "chromatography",
                "chemical structure",
                "mathematical formula",
                "non-clinical photos",
                "hand-drawn sketches"]

    def encode(self) -> Dict[str, torch.Tensor]:
        """Embed image and text."""
        embeddings: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            Modalities.TEXT.embedding: [],
            Modalities.RGB.embedding: [],
        }
        assert isinstance(embeddings[Modalities.TEXT.embedding], list)
        assert isinstance(embeddings[Modalities.RGB.embedding], list)
        for idx, batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="encoding"):
            outputs = self.model(batch)
            if idx == 0:
                embeddings.update(batch["entry"])
            else:
                for key, value in batch["entry"].items():
                    embeddings[key].extend(value)
            embeddings[Modalities.TEXT.embedding].append(outputs[Modalities.TEXT.embedding].detach().cpu())
            embeddings[Modalities.RGB.embedding].append(outputs[Modalities.RGB.embedding].detach().cpu())
            # TODO: remove this
            if idx > 0:
                break
        embeddings[Modalities.TEXT.embedding] = torch.cat(embeddings[Modalities.TEXT.embedding], axis=0).cpu()  # type: ignore[call-overload]
        embeddings[Modalities.RGB.embedding] = torch.cat(embeddings[Modalities.RGB.embedding], axis=0).cpu()  # type: ignore[call-overload]
        return embeddings

    def compute(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute similarity scores between image-text pairs and modality keywords."""
        # embed keywords
        kword_embeddings = self.embed_keywords()
        # compute similarity of image to keyword embeddings
        kword_embeddings = kword_embeddings / torch.norm(kword_embeddings, dim=1, keepdim=True)
        embeddings[Modalities.RGB.embedding] = embeddings[Modalities.RGB.embedding] / torch.norm(embeddings[Modalities.RGB.embedding], dim=1, keepdim=True)
        scores = torch.matmul(embeddings[Modalities.RGB.embedding], kword_embeddings.T)
        return torch.softmax(scores, dim=1)  # num_samples x num_keywords

    def sort_labels(self, scores: torch.Tensor):
        """Sort keywords based on similarity scores."""
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True, stable=True)
        sorted_labels = [[self.keywords[idx] for idx in row] for row in indices]
        return sorted_labels, sorted_scores

    def embed_keywords(self) -> torch.Tensor:
        """
        Generate embeddings for the given classes using BiomedCLIP.

        Returns
        -------
        torch.Tensor
            Tensor of keyword embeddings.
        """
        template = "this is a photo of {}"
        inputs = self.tokenizer([template.format(word) for word in self.keywords])
        if Modalities.TEXT not in inputs and isinstance(inputs, torch.Tensor):
            inputs = {Modalities.TEXT: inputs}
        with torch.no_grad():
            text_features = self.model.encode(inputs, Modalities.TEXT)
        return text_features

    def save_entries_as_csv(self, entries: Dict[str, torch.Tensor], filename: str = "./entries.csv"):
        """Save entries as csv."""
        for key in entries.keys():
            if isinstance(entries[key], torch.Tensor):
                entries[key] = entries[key].tolist()
        entries_df = pd.DataFrame.from_dict(entries, orient="columns")
        entries_df.to_csv(filename, sep=",")

    def load_entries_from_csv(self, filename: str = "./entries.csv"):
        """Load entries from csv."""
        entries = pd.read_csv(filename, sep=",").to_dict(orient="list")
        # evaluate list-type columns
        entries["labels"] = [self._safe_eval(labels) for labels in entries["labels"]]
        entries["scores"] = [self._safe_eval(scores) for scores in entries["scores"]]
        return entries

    def _safe_eval(self, x: str) -> List[str]:
        """Safely evaluate a string as a list."""
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)  # type: ignore[no-any-return]
        except (ValueError, SyntaxError):
            return []

    def save_embeddings(
        self, embeddings: Dict[str, torch.Tensor], filename: str = "./embeddings.pt"
    ) -> None:
        """Save text and rgb embeddings on disk."""
        torch.save(embeddings, filename)
        print(f"Saved embeddings in {filename}")

    def load_embeddings(self, filename: str) -> Any:
        """Load embeddings from file."""
        return torch.load(filename, weights_only=True)

    def forward(self) -> Dict[str, torch.Tensor]:
        """Compute the similarity of image-text paris with all keywords.

        Returns
        -------
        Dict[str, Any]:
            Dictionary of entries along with their classified modalities.
        """
        # encode images and texts
        embeddings = self.encode()
        # compute similarities of images with keywords
        scores = self.compute(embeddings)
        # sort labels
        sorted_labels, sorted_scores = self.sort_labels(scores)
        # create new entrylist
        embeddings.pop(Modalities.TEXT.embedding)
        embeddings.pop(Modalities.RGB.embedding)
        embeddings.update({"labels": sorted_labels, "scores": sorted_scores})
        return embeddings


@hydra_main(version_base=None, config_path="pkg://mmlearn.conf", config_name="base_config")
def main(cfg: DictConfig):
    """Entry point for classification."""
    L.seed_everything(cfg.seed, workers=True)

    if is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if "16-mixed" in cfg.trainer.precision:
            cfg.trainer.precision = "bf16-mixed"

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # instantiate data loader
    test_dataset = instantiate_datasets(cfg.datasets.test)
    assert (
        test_dataset is not None
    ), "Test dataset (`cfg.datasets.test`) is required for evaluation."
    cfg.dataloader.test["sampler"] = None
    test_loader = hydra.utils.instantiate(
        cfg.dataloader.test, dataset=test_dataset, sampler=None
    )
    # instantiate tokenizer
    test_tokenizer = hydra.utils.instantiate(cfg.dataloader.test.collate_fn.batch_processors.text)

    # setup task module
    if cfg.task is None or "_target_" not in cfg.task:
        raise ValueError(
            "Expected a non-empty config for `cfg.task` with a `_target_` key. "
            f"But got: {cfg.task}"
        )
    logger.info(f"Instantiating task module: {cfg.task['_target_']}")
    model: L.LightningModule = hydra.utils.instantiate(cfg.task, _convert_="partial")
    assert isinstance(model, L.LightningModule), "Task must be a `LightningModule`"
    model.strict_loading = cfg.strict_loading

    # compile model
    model = torch.compile(model, **OmegaConf.to_object(cfg.torch_compile_kwargs))
    # logger.info(model)

    # load a checkpoint
    if cfg.resume_from_checkpoint is not None:
        logger.info(f"Loading model state from: {cfg.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.resume_from_checkpoint, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    # setup keywords
    keywords = None

    # instantiate classifier
    classifier = ModalityClassifier(model, test_loader, test_tokenizer, keywords)

    # classify images
    entries = classifier()
    classifier.save_entries_as_csv(entries, "openpmcvl/probe/entries.csv")

    # load entries from csv
    entries = classifier.load_entries_from_csv("openpmcvl/probe/entries.csv")






if __name__ == "__main__":
    main()