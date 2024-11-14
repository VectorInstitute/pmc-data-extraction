"""PMCPatients Dataset."""

import json
import os
from typing import Callable, Dict, Literal, Optional, Union

import pandas as pd
import torch
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from pandas import DataFrame
from torch.utils.data import Dataset


# TODO: make this dataset work for PAR as well as PPR.
@external_store(group="datasets", root_dir=os.getenv("PMCPATIENTS_ROOT_DIR", MISSING))
class PMCPatients(Dataset[Example]):
    """PMCPatients dataset.

    Patient case descriptions extracted from PubMed open-access articles,
    prepared for patient-to-patient and patient-to-article retrieval benchmarks.
    This dataset (PMC-Patients-ReCDS) is a part of a group of three datasets,
    a summary of each follows.

    1. PMC-Patients[1]: patient summary containing patient ID, article ID, case
    description, relevant patients and articles, and their relevancy score.

    2. PMC-Patients-ReCDS[2]: benchmark data for patient-to-patient and
    patient-to-article retrieval tasks. Data format resembles the BEIR[3]
    benchmark, meaning that three concepts are defined: queries, corpus, qrels.
    Both PPR and PAR share the same query patient sets, described by patient-
    ID and case description in `PMC-Patients-ReCDS/queries`. The corpus of PPR
    contains 155.2K reference patients and the corpus of PAR contains 11.7M
    PubMed articles. qrels are TREC-style retrieval annotation files in `tsv`
    format. A qrels file contains three tab-separated columns, i.e. the query
    identifier, corpus identifier, and score in this order. The scores
    (2 or 1) indicate the relevance level in ReCDS-PAR or similarity level in
    ReCDS-PPR. Note that the qrels may not be the same as `relevant_articles`
    and `similar_patients` in PMC-Patients due to dataset split.

    3. PMC-Patients-MetaData[4]: Meta data for PMC-Patients that might
    facilitate reproduction or usage of the dataset, most of which can be
    derived from the main datasets above.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing `queries`, `PPR` and `PAR` folders.
    task: {"ppr", "par"}, default="ppr"
        Intended task of the loaded data.
        `ppr` stands for patient-to-patient retrieval,
        and `par` stands for patient-to-article retrieval.
    split : {"train", "dev", "test"}, default="train"
        Dataset split.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.

    Notes
    -----
    [1] https://huggingface.co/datasets/zhengyun21/PMC-Patients
    [2] https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS
    [3] https://github.com/beir-cellar/beir
    [4] https://huggingface.co/datasets/zhengyun21/PMC-Patients-MetaData
    """

    def __init__(
        self,
        root_dir: str,
        task: Literal["ppr", "par"] = "ppr",
        split: Literal["train", "dev", "test"] = "train",
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        # load queries
        self.queries: DataFrame = self._load_jsonl_data(
            os.path.join(root_dir, "queries", f"{split}_queries.jsonl")
        )
        # load corpus
        self.corpus: DataFrame = self._load_jsonl_data(
            os.path.join(
                root_dir,
                task.upper(),
                "corpus.jsonl" if task == "ppr" else "PAR_PMIDs.json",
            )
        )
        # load qrels
        self.qrels: DataFrame = pd.read_csv(
            os.path.join(root_dir, task.upper(), f"qrels_{split}.tsv"), sep="\t"
        )

        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        try:
            query_id = self.qrels.iloc[idx].loc["query-id"]
            corpus_id = self.qrels.iloc[idx].loc["corpus-id"]
            query_text = (
                self.queries["text"].loc[self.queries["_id"] == query_id].values[0]
            )
            target_text = (
                self.corpus["text"].loc[self.corpus["_id"] == corpus_id].values[0]
            )
        except Exception as e:
            print(f"Error loading data for entry {idx}: {e}")
            idx = (idx + 1) % len(self)
            return self.__getitem__(idx)

        query_tokens = (
            self.tokenizer(query_text) if self.tokenizer is not None else None
        )
        target_tokens = (
            self.tokenizer(target_text) if self.tokenizer is not None else None
        )

        example = Example(
            {
                Modalities.PATIENT_Q.name: query_text,
                Modalities.PATIENT_T.name: target_text,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if query_tokens is not None:
            if isinstance(query_tokens, dict) and isinstance(
                target_tokens, dict
            ):  # output of HFTokenizer
                assert (
                    Modalities.TEXT.name in query_tokens
                ), f"Missing key `{Modalities.TEXT.name}` in query tokens."
                assert (
                    Modalities.TEXT.name in target_tokens
                ), f"Missing key `{Modalities.TEXT.name}` in target tokens."
                example[Modalities.PATIENT_Q.name] = query_tokens[Modalities.TEXT.name]
                example[Modalities.PATIENT_T.name] = target_tokens[Modalities.TEXT.name]
                if (
                    "attention_mask" in query_tokens
                    and "attention_mask" in target_tokens
                ):
                    example["attention_mask_q"] = query_tokens["attention_mask"]
                    example["attention_mask_t"] = target_tokens["attention_mask"]
            else:
                example[Modalities.PATIENT_Q.name] = query_tokens
                example[Modalities.PATIENT_T.name] = target_tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.qrels.index)

    def _load_jsonl_data(self, filename: str) -> DataFrame:
        """Load a jsonl dataset and convert to pandas Dataframe."""
        with open(filename, encoding="utf-8") as file:
            data = [json.loads(line) for line in file.readlines()]
        return pd.DataFrame.from_records(data)
