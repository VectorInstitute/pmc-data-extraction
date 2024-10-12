"""Classify image into modalities given in [1] using retrieval.

For each image, the similarity of the image embedding with all labels
in [1] is computed and the highly similar labels are retrieved.

References
----------
[1] Garcia Seco de Herrera, A., Muller, H. & Bromuri, S.
    "Overview of the ImageCLEF 2015 medical classification task."
    In Working Notes of CLEF 2015 (Cross Language Evaluation Forum) (2015).
"""
import hydra
from omegaconf import OmegaConf, DictConfig

import torch.nn as nn

from mmlearn.conf import JobType, MMLearnConf, hydra_main
from mmlearn.datasets.core import *  # noqa: F403
from mmlearn.datasets.processors import *  # noqa: F403
from mmlearn.modules.encoders import *  # noqa: F403
from mmlearn.modules.layers import *  # noqa: F403
from mmlearn.modules.losses import *  # noqa: F403
from mmlearn.modules.lr_schedulers import *  # noqa: F403
from mmlearn.modules.metrics import *  # noqa: F403
from mmlearn.tasks import *  # noqa: F403


# class ModalityClassifier(nn.Module):
#     """Classify the modality of an image-text pair by retrieval."""

#     def __init__(self, config_name):
#         """Initialize the module.

#         Parameters
#         ----------
#         config_name: str
#             Path to a yaml file containinng experiment config.
#             This config is expected to contain an instance of
#             `ContrastivePretraining` as "task".
#         """
#         super().__init__()
#         # load config
#         config = OmegaConf.load(config_name)
#         print(config)


#         # load model
#         task = self._instantiate_task(task_name="biomedclip", checkpoint=None)

#         # load dataset

#         # load taxonomy
    

#     def _instantiate_task(self, task_name="biomedclip", checkpoint=None):
#         """Instantiate ContrastivePretraining task."""
#         if task_name == "biomedclip":
#             return self._instantiate_biomedclip(checkpoint=checkpoint)
    

#     def _instantiate_biomedclip(self, checkpoint=None):
#         """Instantiate BiomedCLIP encoders and post-processors."""
#         encoders = {
#         "text": BiomedCLIPText(
#             model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#             pretrained=True,
#             use_all_token_embeddings=False,
#             freeze_layers=False,
#             freeze_layer_norm=True,
#             model_config_kwargs=None,
#         ),
#         "rgb": BiomedCLIPVision(
#             model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#             pretrained=True,
#             use_all_token_embeddings=False,
#             freeze_layers=False,
#             freeze_layer_norm=True,
#             model_config_kwargs=None,
#         ),
#     }
#     heads = None
#     postprocessors = {
#         "norm_and_logit_scale": {
#             "norm": L2Norm(dim=-1),
#             "logit_scale": LearnableLogitScaling(
#                 logit_scale_init=14.285714285714285,
#                 learnable=True,
#                 max_logit_scale=100.0,
#             ),
#         }
#     }
#     modality_module_mapping = {
#         "text": ModuleKeySpec(
#             encoder_key=None, head_key=None, postprocessor_key="norm_and_logit_scale"
#         ),
#         "rgb": ModuleKeySpec(
#             encoder_key=None, head_key=None, postprocessor_key="norm_and_logit_scale"
#         ),
#     }
#     optimizer = None
#     lr_scheduler = None
#     loss = CLIPLoss(
#         l2_normalize=False, local_loss=True, gather_with_grad=True, cache_labels=False
#     )
#     modality_loss_pairs = None
#     auxiliary_tasks = None
#     log_auxiliary_tasks_loss = False
#     compute_validation_loss = True
#     compute_test_loss = True
#     evaluation_tasks = {
#         "retrieval": EvaluationSpec(
#             task=ZeroShotCrossModalRetrievalEfficient(
#                 task_specs=[
#                     RetrievalTaskSpec(
#                         query_modality="text",
#                         target_modality="rgb",
#                         top_k=[10, 50, 200],
#                     ),
#                     RetrievalTaskSpec(
#                         query_modality="rgb",
#                         target_modality="text",
#                         top_k=[10, 50, 200],
#                     ),
#                 ]
#             ),
#             run_on_validation=False,
#             run_on_test=True,
#         )
#     }

#     tasl = ContrastivePretraining(
#         encoders=encoders,
#         heads=heads,
#         postprocessors=postprocessors,
#         modality_module_mapping=modality_module_mapping,
#         optimizer=optimizer,
#         lr_scheduler=lr_scheduler,
#         loss=loss,
#         modality_loss_pairs=modality_loss_pairs,
#         auxiliary_tasks=auxiliary_tasks,
#         log_auxiliary_tasks_loss=log_auxiliary_tasks_loss,
#         compute_validation_loss=compute_validation_loss,
#         compute_test_loss=compute_test_loss,
#         evaluation_tasks=evaluation_tasks,
#     )

#     # load checkpoint





#     def forward(self, x):
#         """Compute the similarity of image-text paris with all keywords."""


@hydra_main(version_base=None, config_path="pkg://mmlearn.conf", config_name="base_config")
def main(cfg: DictConfig):
    # classifier = ModalityClassifier(config_name="openpmcvl/experiment/configs/experiment/biomedclip_matched.yaml")
    print(OmegaConf.to_yaml(cfg))





if __name__ == "__main__":
    # do something
    main()