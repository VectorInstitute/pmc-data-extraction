"""Classify image into modalities given in [1] using retrieval.

For each image, the similarity of the image embedding with all labels
in [1] is computed and the highly similar labels are retrieved.

References
----------
[1] Garcia Seco de Herrera, A., Muller, H. & Bromuri, S.
    "Overview of the ImageCLEF 2015 medical classification task."
    In Working Notes of CLEF 2015 (Cross Language Evaluation Forum) (2015).
"""
from typing import Any, Dict, List, Union
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

    def __init__(self, model: L.LightningModule, loader: DataLoader):
        """Initialize the module.

        Parameters
        ----------
        model: L.LightningModule
            Task module loaded from an mmlearn experiment.
        loader: DataLoader
            Data loader.
        """
        super().__init__()
        self.model = model
        self.loader = loader

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

    def save_embeddings_as_csv(self, embeddings: Dict[str, torch.Tensor], filename: str = "./embeddings.csv"):
        """Save text and rgb embeddings along with entries as csv."""
        for mod in [Modalities.TEXT, Modalities.RGB]:
            embeddings[mod.embedding] = embeddings[mod.embedding].tolist()
        entries_df = pd.DataFrame.from_dict(embeddings, orient="columns")
        entries_df.to_csv(filename, sep=",")

    def load_embeddings_from_csv(self, filename: str):
        """Load embeddings along with entries from csv."""
        entries_df = pd.read_csv(filename)
        # print(entries_df)
        print(entries_df["text_embedding"].iloc[0])
        print(type(entries_df["text_embedding"].iloc[0]))
        l = entries_df["text_embedding"].iloc[0].tolist()
        print(type(l))
        print(l)

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
        """Compute the similarity of image-text paris with all keywords."""
        return self.encode()


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

    # instantiate classifier
    classifier = ModalityClassifier(model, test_loader)
    # logger.info(classifier)

    # encode images and texts
    embeddings = classifier.encode()
    # print(embeddings)
    print(embeddings["text_embedding"].size())
    print(embeddings["rgb_embedding"].size())
    print(len(embeddings["caption_name"]))

    # save embeddings as csv
    classifier.save_embeddings_as_csv(embeddings, "openpmcvl/probe/embeddings.csv")
    # load embeddings from csv
    classifier.load_embeddings_from_csv("openpmcvl/probe/embeddings.csv")





if __name__ == "__main__":
    main()