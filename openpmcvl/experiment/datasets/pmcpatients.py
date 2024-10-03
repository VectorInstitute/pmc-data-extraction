"""PMC-VL Dataset."""

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


@external_store(group="datasets", root_dir=os.getenv("PMCPATIENTS_ROOT_DIR", MISSING))
class PMCPatients(Dataset[Example]):
    """PMC-VL dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "dev", "test"}
        Dataset split.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"] = "train",
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        # load test queries
        queries_file = os.path.join(
            root_dir, "PMC-Patients-ReCDS/queries/test_queries.jsonl"
        )
        with open(queries_file, encoding="utf-8") as file:
            queries = [json.loads(line) for line in file.readlines()]
        queries = pd.DataFrame.from_records(queries)
        self.queries: DataFrame = queries
        # load ppr corpus
        corpus_file = os.path.join(root_dir, "PMC-Patients-ReCDS/PPR/corpus.jsonl")
        with open(corpus_file, encoding="utf-8") as file:
            corpus = [json.loads(line) for line in file.readlines()]
        corpus = pd.DataFrame.from_records(corpus)
        self.corpus: DataFrame = corpus
        # load ppr test qrels
        qrels_file = os.path.join(root_dir, "PMC-Patients-ReCDS/PPR/qrels_test.tsv")
        qrels = pd.read_csv(qrels_file, sep="\t")
        self.qrels: DataFrame = qrels

        self.root_dir = root_dir

        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        try:
            query_id = self.qrels.iloc[idx].loc["query-id"]
            corpus_id = self.qrels.iloc[idx].loc["corpus-id"]
            query_text = (
                self.queries.loc["text"].loc[self.queries["_id"] == query_id].values[0]
            )
            target_text = (
                self.corpus.loc["text"].loc[self.corpus["_id"] == corpus_id].values[0]
            )
        except Exception:
            print(f"Error loading image or caption for entry {idx}")
            idx = (idx + 1) % len(self.qrels.index)
            return self.__getitem__(idx)

        query_tokens = (
            self.tokenizer(query_text) if self.tokenizer is not None else None
        )
        target_tokens = (
            self.tokenizer(target_text) if self.tokenizer is not None else None
        )

        example = Example(
            {
                Modalities.PATIENT_Q: query_text,
                Modalities.PATIENT_T: target_text,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if query_tokens is not None and target_tokens is not None:
            if isinstance(query_tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT in query_tokens
                ), f"Missing key `{Modalities.TEXT}` in query tokens."
                assert (
                    Modalities.TEXT in target_tokens
                ), f"Missing key `{Modalities.TEXT}` in target tokens."
                example[Modalities.PATIENT_Q] = query_tokens[Modalities.TEXT]
                example[Modalities.PATIENT_T] = target_tokens[Modalities.TEXT]
            else:
                example[Modalities.PATIENT_Q] = query_tokens
                example[Modalities.PATIENT_T] = target_tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.qrels.index)
