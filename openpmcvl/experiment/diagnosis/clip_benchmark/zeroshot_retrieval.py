"""Retrieval recall calculation as implemented by clip_benchmark library [1].

References
----------
[1] https://github.com/LAION-AI/CLIP_benchmark
"""

import json
from typing import Any, Callable, Dict, List

import torch
from tqdm import tqdm


def evaluate(
    texts_emb: torch.Tensor,
    images_emb: torch.Tensor,
    batch_size: int,
    recall_k_list: List[int],
    device: torch.device,
    amp: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the model on the given dataset.

    Parameters
    ----------
    text_embedding: torch Tensor

    rgb_embedding: torch Tensor

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use

    Returns
    -------
    dict of retrieval metrics
    """
    # for each text, we collect the corresponding image index,
    # as each image can have multiple corresponding texts
    texts_image_index = torch.arange(len(texts_emb))

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells
    # whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e.
        # nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,
        # the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching
        # the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in
        # CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1
        # if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking
        # whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute
        # the recall for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )
        metrics[f"text_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    recall_at_k,
                    scores.T,
                    positive_pairs.T,
                    batch_size,
                    device,
                    k=recall_k,
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )

    # print results on the terminal
    print(json.dumps(metrics, indent=4))
    return metrics


def recall_at_k(
    scores: torch.Tensor, positive_pairs: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Compute the recall at k for each sample.

    :param scores: compatibility score between  text and image embeddings
    (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs
    (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    return nb_true_positive / nb_positive


def batchify(
    func: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    xx: torch.Tensor,
    yy: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Process data in batches given the function."""
    results = []
    for start in tqdm(range(0, len(xx), batch_size), desc="recall@k computation"):
        end = start + batch_size
        x = xx[start:end].to(device)
        y = yy[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)

    return torch.cat(results)


if __name__ == "__main__":
    """Run zeroshot retrieval on saved embeddings."""
    # set params
    batch_size = 16
    recall_k_list = [10, 50, 200]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True

    # load embeddings
    embeddings = torch.load(
        "openpmcvl/experiment/diagnosis/embeddings_neurips.pt", weights_only=True
    )

    # compute recall@k
    metrics = evaluate(
        embeddings["text_embedding"],
        embeddings["rgb_embedding"],
        batch_size=batch_size,
        recall_k_list=recall_k_list,
        device=device,
        amp=amp,
    )
