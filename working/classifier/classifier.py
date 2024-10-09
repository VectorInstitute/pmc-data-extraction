"""Modality classifier based on retrieval and a given taxonomy of image modalities.

Classifier is an auxiliary task used with `ContrastivePretraining` in `mmlearn` package.
Training is similar to the training pipeline of `ContrastivePretraining`; the only difference is the dataset.
Evaluation is classification instead of retrieval; classification can be done in two ways:
 1. Retrieval-like: using the similarity of image embeddings with the textual modality labels, retrieve highly probable labels.
 2. Linear probe: adding a linear classification head to deduct label probabilities from image embeddings.
Zero-shot is only a name; in reality, we *can* finetune the encoders on rule-based extracted image-label pairs.
Linear proble requires trianing for the linear head which means `ContrastivePretraining` needs to be modified as well.
"""
