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

    def encode_old(self) -> Dict[str, torch.Tensor]:
        """Embed image and text."""
        embeddings: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            "text_embedding": [],
            "rgb_embedding": [],
        }
        assert isinstance(embeddings["text_embedding"], list)
        assert isinstance(embeddings["rgb_embedding"], list)
        for _, batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="encoding"):
            outputs = self.model(batch)
            embeddings["text_embedding"].append(outputs["text_embedding"].detach().cpu())
            embeddings["rgb_embedding"].append(outputs["rgb_embedding"].detach().cpu())
            # TODO: remove this
            if _ > 1:
                break
        embeddings["text_embedding"] = torch.cat(embeddings["text_embedding"], axis=0).cpu()  # type: ignore[call-overload]
        embeddings["rgb_embedding"] = torch.cat(embeddings["rgb_embedding"], axis=0).cpu()  # type: ignore[call-overload]
        assert isinstance(embeddings["text_embedding"], torch.Tensor)
        assert isinstance(embeddings["rgb_embedding"], torch.Tensor)
        return embeddings

    def encode(self) -> Dict[str, torch.Tensor]:
        """Embed image and text."""
        embeddings: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            "text_embedding": [],
            "rgb_embedding": [],
        }
        assert isinstance(embeddings["text_embedding"], list)
        assert isinstance(embeddings["rgb_embedding"], list)
        for _, batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="encoding"):
            outputs = self.model(batch)
            print(batch)
            print(f"batch.keys(): {batch.keys()}")
            print(f"outputs.keys(): {outputs.keys()}")
            print(f"batch[Modalities.TEXT].size(): {batch[Modalities.TEXT].size()}")
            print(f"outputs['text_embedding'].size(): {outputs['text_embedding'].size()}")
            print(f"Modalities.TEXT.embedding: {Modalities.TEXT.embedding}")
            entrylist_pd = pd.DataFrame.from_dict(batch["entry"].update(outputs), orient="columns")
            print(entrylist_pd)
            exit()
            embeddings["text_embedding"].append(outputs["text_embedding"].detach().cpu())
            embeddings["rgb_embedding"].append(outputs["rgb_embedding"].detach().cpu())
            # TODO: remove this
            if _ > 1:
                break
        embeddings["text_embedding"] = torch.cat(embeddings["text_embedding"], axis=0).cpu()  # type: ignore[call-overload]
        embeddings["rgb_embedding"] = torch.cat(embeddings["rgb_embedding"], axis=0).cpu()  # type: ignore[call-overload]
        assert isinstance(embeddings["text_embedding"], torch.Tensor)
        assert isinstance(embeddings["rgb_embedding"], torch.Tensor)
        return embeddings

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
    print(embeddings)





if __name__ == "__main__":
    main()