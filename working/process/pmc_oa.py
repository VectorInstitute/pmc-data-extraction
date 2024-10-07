"""PMC-OA Dataset"""

import os

import datasets
import jsonlines

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{lin2023pmc,
  title={PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents},
  author={Lin, Weixiong and Zhao, Ziheng and Zhang, Xiaoman and Wu, Chaoyi and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  journal={arXiv preprint arXiv:2303.07240},
  year={2023}
}
"""

_DESCRIPTION = """\
Foundation models trained on large-scale dataset gain a recent surge in CV and NLP. In contrast, development in biomedical domain lags far behind due to data scarcity.
To address this issue, we build and release PMC-OA, a biomedical dataset with 1.6M image-caption pairs collected from PubMedCentral's OpenAccess subset, which is 8 times larger than before.
PMC-OA covers diverse modalities or diseases, with majority of the image-caption samples aligned at finer-grained level, i.e., subfigure and subcaption.
While pretraining a CLIP-style model on PMC-OA, our model named PMC-CLIP achieves state-of-the-art results on various downstream tasks,
including image-text retrieval on ROCO, MedMNIST image classification, Medical VQA, i.e. +8.1% R@10 on image-text retrieval, +3.9% accuracy on image classification.
"""

_HOMEPAGE = "https://weixionglin.github.io/PMC-CLIP/"

_URLs = {
    "images": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/images.zip",
    "pmc_oa_beta": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/pmc_oa_beta.jsonl",
    "pmc_oa": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/pmc_oa.jsonl",
}


class PMC_OA_Config(datasets.BuilderConfig):
    """BuilderConfig for PMC_OA"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(PMC_OA_Config, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class PMC_OA(datasets.GeneratorBasedBuilder):
    """PMC_OA Dataset"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        PMC_OA_Config(
            name="pmc_oa_beta",
            description="<subfigure, caption> pairs. Subfigures detected by a DETR model.",
        ),
        PMC_OA_Config(
            name="pmc_oa",
            description="<subfigure, subcaption> pairs. Subfigures detected by a DETR model. Subcaptions detected by ChatGPT and aligned with subfigures.",
        ),
    ]

    def _info(self):
        if self.config.name == "pmc_oa_beta":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "image": datasets.Value("string"),
                        "caption": datasets.Value("string"),
                    }
                ),
                supervised_keys=None,
                citation=_CITATION,
                homepage=_HOMEPAGE,
            )
        elif self.config.name == "pmc_oa":
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "image": datasets.Value("string"),
                        "caption": datasets.Value("string"),
                        "alignment_type": datasets.Value("string"),
                        "alignment_score": datasets.Value("float"),
                    }
                ),
                supervised_keys=None,
                citation=_CITATION,
                homepage=_HOMEPAGE,
            )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLs)
        if self.config.name == "pmc_oa_beta":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["pmc_oa_beta"], "image_dir": downloaded_files['images']}
                )
            ]
        elif self.config.name == "pmc_oa":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["pmc_oa"], "image_dir": downloaded_files['images']}
                )
            ]

    def _generate_examples(self, filepath, image_dir):
        """Yields examples."""
        logger.info("generating examples from = %s", filepath)
         
        with jsonlines.open(filepath) as reader:
            for _id, obj in enumerate(reader):
                if self.config.name == "pmc_oa_beta":
                    relative_image_path = obj['image']
                    image_path = os.path.join(image_dir, "caption_T060_filtered_top4_sep_v0_subfigures", relative_image_path)
                    caption = obj['caption']
                    yield _id, {
                        "image": {
                            "path": image_path,
                            "bytes": open(image_path, "rb").read(),
                        },
                        "caption": caption,
                    }
                elif self.config.name == "pmc_oa":
                    relative_image_path = obj['image']
                    image_path = os.path.join(image_dir, "caption_T060_filtered_top4_sep_v0_subfigures", relative_image_path)
                    caption = obj['caption']
                    alignment_type = obj['alignment_type']
                    alignment_score = obj['alignment_score']
                    yield _id, {
                        "image": {
                            "path": image_path,
                            "bytes": open(image_path, "rb").read(),
                        },
                        "caption": caption,
                        "alignment_type": alignment_type,
                        "alignment_score": alignment_score,
                    }
