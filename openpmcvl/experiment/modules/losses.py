"""Implementations of the contrastive loss and its variants."""

import itertools
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from mmlearn.conf import external_store
from mmlearn.datasets.core import find_matching_indices
from mmlearn.datasets.core.modalities import Modalities
from mmlearn.tasks.contrastive_pretraining import LossPairSpec
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchmetrics.utilities.compute import _safe_matmul


@external_store(group="modules/losses")
class ContrastiveLoss(nn.Module):
    """Contrastive Loss module.

    Parameters
    ----------
    l2_normalize : bool, default=False
        Whether to L2 normalize the features.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e. `local_features@global_features`.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    modality_alignment : bool, default=False
        Whether to include modality alignment loss. This loss considers all
        features from the same modality as positive pairs and all features
        from different modalities as negative pairs.
    cache_labels : bool, default=False
        Whether to cache the labels.

    """

    def __init__(
        self,
        l2_normalize: bool = False,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        modality_alignment: bool = False,
        cache_labels: bool = False,
    ):
        """Initialize the loss."""
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.l2_normalize = l2_normalize
        self.modality_alignment = modality_alignment

        # cache state
        self._prev_num_logits = 0
        self._labels: Dict[torch.device, torch.Tensor] = {}

    def _get_ground_truth(
        self, device: torch.device, num_logits: int, rank: int, world_size: int
    ) -> torch.Tensor:
        """Return the ground-truth labels.

        Parameters
        ----------
        device : torch.device
            Device to store the labels.
        num_logits : int
            Number of logits.
        rank : int
            Rank of the current process.
        world_size : int
            Number of processes.

        Returns
        -------
        torch.Tensor
            Ground-truth labels.
        """
        # calculate ground-truth and cache if enabled
        if self._prev_num_logits != num_logits or device not in self._labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if world_size > 1 and self.local_loss:
                labels = labels + num_logits * rank
            if self.cache_labels:
                self._labels[device] = labels
                self._prev_num_logits = num_logits
        else:
            labels = self._labels[device]
        return labels

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        example_ids: dict[str, torch.Tensor],
        logit_scale: torch.Tensor,
        modality_loss_pairs: LossPairSpec,
    ) -> torch.Tensor:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if world_size > 1 else 0

        if self.l2_normalize:
            embeddings = {k: F.normalize(v, p=2, dim=-1) for k, v in embeddings.items()}

        if world_size > 1:  # gather embeddings and example_ids across all processes
            all_embeddings = _gather_dicts(
                embeddings,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=rank,
            )

            all_example_ids = _gather_dicts(
                example_ids,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=rank,
            )
        else:
            all_embeddings = embeddings

        losses = []
        for loss_pairs in modality_loss_pairs:
            if world_size > 1:
                dist.barrier(group=dist.group.WORLD)

            modality_a = Modalities.get_modality(loss_pairs.modalities[0])
            modality_b = Modalities.get_modality(loss_pairs.modalities[1])

            if self.local_loss or world_size == 1:
                if not (
                    modality_a.embedding in embeddings
                    and modality_b.embedding in embeddings
                ):
                    continue

                indices_a, indices_b = find_matching_indices(
                    example_ids[modality_a.name], example_ids[modality_b.name]
                )
                if indices_a.numel() == 0 or indices_b.numel() == 0:
                    continue

                features_a = embeddings[modality_a.embedding][indices_a]
                features_b = embeddings[modality_b.embedding][indices_b]

            if world_size > 1:
                if not (
                    modality_a.embedding in all_embeddings
                    and modality_b.embedding in all_embeddings
                ):
                    continue

                indices_a, indices_b = find_matching_indices(
                    all_example_ids[modality_a.name],
                    all_example_ids[modality_b.name],
                )
                if indices_a.numel() == 0 or indices_b.numel() == 0:
                    continue

                all_features_a = all_embeddings[modality_a.embedding][indices_a]
                all_features_b = all_embeddings[modality_b.embedding][indices_b]

                if self.local_loss:
                    logits_per_feature_a = logit_scale * _safe_matmul(
                        features_a, all_features_b
                    )
                    logits_per_feature_b = logit_scale * _safe_matmul(
                        features_b, all_features_a
                    )
                else:
                    logits_per_feature_a = logit_scale * _safe_matmul(
                        all_features_a, all_features_b
                    )
                    logits_per_feature_b = logits_per_feature_a.T
            else:
                logits_per_feature_a = logit_scale * _safe_matmul(
                    features_a, features_b
                )
                logits_per_feature_b = logit_scale * _safe_matmul(
                    features_b, features_a
                )

            labels = torch.arange(
                logits_per_feature_a.shape[-1],
                device=logits_per_feature_a.device,
                dtype=torch.long,
            )
            if world_size > 1 and self.local_loss:
                local_size = torch.tensor(
                    logits_per_feature_a.shape[0], device=logits_per_feature_a.device
                )
                sizes = torch.stack(
                    _simple_gather_all_tensors(
                        local_size, group=dist.group.WORLD, world_size=world_size
                    )
                )
                sizes = torch.cat(
                    [torch.tensor([0], device=sizes.device), torch.cumsum(sizes, dim=0)]
                )
                labels = labels[
                    sizes[rank] : sizes[rank + 1] if rank + 1 < world_size else None
                ]

            losses.append(
                (
                    (
                        F.cross_entropy(logits_per_feature_a, labels)
                        + F.cross_entropy(logits_per_feature_b, labels)
                    )
                    / 2
                )
                * loss_pairs.weight
            )

        if self.modality_alignment:
            available_modalities = list(all_embeddings.keys())
            # TODO: support local_loss for modality_alignment
            # if world_size == 1, all_embeddings == embeddings
            all_features = torch.cat(list(all_embeddings.values()), dim=0)

            positive_indices = torch.tensor(
                [
                    (i, j)
                    if idx == 0
                    else (
                        i + all_embeddings[available_modalities[idx - 1]].size(0),
                        j + all_embeddings[available_modalities[idx - 1]].size(0),
                    )
                    for idx, k in enumerate(all_embeddings)
                    for i, j in itertools.combinations(
                        range(all_embeddings[k].size(0)), 2
                    )
                ],
                device=all_features.device,
            )
            logits = logit_scale * _safe_matmul(all_features, all_features)

            target = torch.eye(all_features.size(0), device=all_features.device)
            target[positive_indices[:, 0], positive_indices[:, 1]] = 1

            modality_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )

            target_pos = target.bool()
            target_neg = ~target_pos

            # loss_pos and loss_neg below contain non-zero values only for those
            # elements that are positive pairs and negative pairs respectively.
            loss_pos = torch.zeros(
                logits.size(0), logits.size(0), device=target.device
            ).masked_scatter(target_pos, modality_loss[target_pos])
            loss_neg = torch.zeros(
                logits.size(0), logits.size(0), device=target.device
            ).masked_scatter(target_neg, modality_loss[target_neg])

            loss_pos = loss_pos.sum(dim=1)
            loss_neg = loss_neg.sum(dim=1)
            num_pos = target.sum(dim=1)
            num_neg = logits.size(0) - num_pos

            losses.append(((loss_pos / num_pos) + (loss_neg / num_neg)).mean())

        return torch.stack(losses).sum()


def _get_dtype_max(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point():
        return torch.finfo(tensor.dtype).max
    if not tensor.is_complex():
        return torch.iinfo(tensor.dtype).max
    raise ValueError(
        f"Unsupported dtype {tensor.dtype}. Only floating point and integer types are supported."
    )


def _is_all_dtype_max(tensor: torch.Tensor) -> bool:
    dtype_max = _get_dtype_max(tensor)
    return torch.all(tensor == dtype_max)


def _gather_dicts(
    dicts: dict[str, torch.Tensor],
    local_loss: bool,
    rank: int,
    gather_with_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Gather dictionaries of tensors across all processes.

    Parameters
    ----------
    dicts : dict[str, torch.Tensor]
        Dictionary of tensors to gather.
    local_loss : bool, default=False
        Whether to calculate the loss locally i.e.
        `matmul(local_features, global_features)`. If False, this method ensures
        that the gathered features contain local features for the current rank.
    gather_with_grad : bool, default=False
        Whether to gather tensors with gradients.
    rank : int, default=0
        Rank of the current process.

    Returns
    -------
    dict[str, torch.Tensor]
        Gathered dictionary of tensors.
    """
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    current_device = next(iter(dicts.values())).device
    dist.barrier(group=group)

    # gather keys
    local_keys = list(dicts.keys())
    all_keys = [None] * world_size
    dist.all_gather_object(all_keys, local_keys, group=group)
    all_keys = sorted(set(itertools.chain.from_iterable(all_keys)))

    # gather tensors
    gathered_dict = {}
    for key in all_keys:
        if key not in dicts:  # use dummy tensor for missing key in current process
            placeholder_tensor = dicts[local_keys[0]]
            tensor = torch.full_like(
                placeholder_tensor,
                fill_value=_get_dtype_max(placeholder_tensor),
                device=current_device,
                memory_format=torch.contiguous_format,
                requires_grad=gather_with_grad
                and placeholder_tensor.is_floating_point(),  # only floating point tensors can have gradients
            )
        else:
            tensor = dicts[key].contiguous()

        gathered_tensors: list[torch.Tensor] = _gather_all_tensors(
            tensor,
            world_size=world_size,
            group=group,
            gather_with_grad=gather_with_grad,
        )

        if not gather_with_grad and not local_loss:
            gathered_tensors[rank] = tensor

        # filter out placeholder tensors
        gathered_tensors = [t for t in gathered_tensors if not _is_all_dtype_max(t)]

        gathered_dict[key] = torch.cat(gathered_tensors, dim=0)

    return gathered_dict


def _simple_gather_all_tensors(
    result: torch.Tensor, group: Any, world_size: int, gather_with_grad: bool = False
) -> list[torch.Tensor]:
    if gather_with_grad:
        return list(dist_nn.all_gather(result, group))

    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    dist.all_gather(gathered_result, result, group)
    return gathered_result


def _gather_all_tensors(
    result: torch.Tensor,
    world_size: Optional[int] = None,
    group: Optional[Any] = None,
    gather_with_grad: bool = False,
) -> list[torch.Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcasted to all processes."""
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    if world_size is None:
        world_size = dist.get_world_size(group)
        dist.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size, gather_with_grad)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size, gather_with_grad)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    if gather_with_grad:
        gathered_result = list(dist_nn.all_gather(result_padded, group))
    else:
        gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
        dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result
