"""Zero-shot cross-modal retrieval evaluation task."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.distributed
import torch.distributed.nn
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.tasks.hooks import EvaluationHooks
from torchmetrics import Metric, MetricCollection

from openpmcvl.experiment.modules.metrics import RetrievalRecallAtKEfficient


@dataclass
class RetrievalTaskSpec:
    """Specification for a retrieval task."""

    query_modality: str
    target_modality: str
    top_k: List[int]


@external_store(group="eval_task", provider="openpmcvl")
class ZeroShotCrossModalRetrievalEfficient(EvaluationHooks):  # type: ignore [misc]
    """Zero-shot cross-modal retrieval evaluation task.

    This task evaluates the retrieval performance of a model on a set of query-target
    pairs. The model is expected to produce embeddings for both the query and target
    modalities. The task computes the retrieval recall at `k` for each pair of
    modalities.

    Parameters
    ----------
    task_specs : List[RetrievalTaskSpec]
        A list of retrieval task specifications. Each specification defines the query
        and target modalities, as well as the top-k values for which to compute the
        retrieval recall metrics.

    """

    def __init__(self, task_specs: List[RetrievalTaskSpec]):
        """Initialize the module."""
        super().__init__()

        self.task_specs = task_specs
        self.metrics: Union[Dict[str, Metric], MetricCollection] = {}

        for spec in self.task_specs:
            query_modality = spec.query_modality
            target_modality = spec.target_modality

            assert Modalities.has_modality(query_modality)
            assert Modalities.has_modality(target_modality)

            self.metrics.update(
                {
                    f"{query_modality}_to_{target_modality}_R@{k}": RetrievalRecallAtKEfficient(
                        top_k=k, aggregation="mean", reduction="none"
                    )
                    for k in spec.top_k
                }
            )
        self.metrics = MetricCollection(self.metrics)

        self.modality_pairs = [
            (key.split("_to_")[0], key.split("_to_")[1].split("_R@")[0])  # type: ignore[attr-defined]
            for key in self.metrics
        ]

    def on_evaluation_epoch_start(self, pl_module: pl.LightningModule) -> None:
        """Move the metrics to the device of the Lightning module."""
        self.metrics.to(pl_module.device)  # type: ignore [union-attr]

    def evaluation_step(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Run the forward pass and update retrieval recall metrics.

        Parameters
        ----------
        trainer : pl.Trainer
            A reference to the Lightning Trainer.
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.
        batch : Dict[str, torch.Tensor]
            A dictionary of batched input tensors.
        batch_idx : int
            The index of the batch.

        """
        if pl_module.trainer.sanity_checking:
            return

        outputs: Dict[str, Any] = pl_module(batch)
        for (query_modality, target_modality), metric in zip(
            self.modality_pairs, self.metrics.values()
        ):
            query_embeddings: torch.Tensor = outputs[
                Modalities.get_modality(query_modality).embedding
            ]
            target_embeddings: torch.Tensor = outputs[
                Modalities.get_modality(target_modality).embedding
            ]
            indexes = torch.arange(query_embeddings.size(0), device=pl_module.device)

            metric.update(query_embeddings, target_embeddings, indexes)

    def on_evaluation_epoch_end(
        self, pl_module: pl.LightningModule
    ) -> Optional[Dict[str, Any]]:
        """Compute the retrieval recall metrics.

        Parameters
        ----------
        pl_module : pl.LightningModule
            A reference to the Lightning module being evaluated.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary of evaluation results or `None` if no results are available.
        """
        if pl_module.trainer.sanity_checking:
            return None

        results: Dict[str, Any] = {}
        results.update(self.metrics.compute())  # type: ignore [union-attr]
        self.metrics.reset()  # type: ignore [union-attr]

        eval_type = "val" if pl_module.trainer.validating else "test"

        for key, value in results.items():
            pl_module.log(f"{eval_type}/{key}", value)

        return results
