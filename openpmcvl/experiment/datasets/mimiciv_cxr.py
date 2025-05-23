"""MIMIC-IV-CXR Dataset."""

import json
import logging
import os
from typing import Callable, Literal, Optional, get_args

import numpy as np
import pandas as pd
import torch
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm


logger = logging.getLogger(__name__)


@external_store(
    group="datasets",
    root_dir=os.getenv("MIMICIVCXR_ROOT_DIR", MISSING),
    split="train",
    labeler="double_image",
)
class MIMICIVCXR(Dataset):  # type: ignore[type-arg]
    """Module to load image-text pairs from MIMIC-IV-CXR dataset.

    Parameters
    ----------
    root_dir : str
        Path to the directory containing json files which describe data entries.
    split : {"train", "validate", "test"}
        Dataset split.
    labeler : {"chexpert", "negbio", "double_image", "single_image"}
        Model which was used to generate labels from raw-text reports.
    transform :  Optional[Callable]
        Custom transform applied to images.
    tokenizer : Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        A function that tokenizes the raw text reports.
    include_report : bool, default=False
        Whether or not to include the raw text reports in the data example.

    Notes
    -----
    Some datapoints have not been processed by the labelers; hence, they have no
    assigned label. Note that this is different from all labels being set to `NaN`.
    These datapoints are removed from the dataset if `include_report` is False.
    Otherwise, the datapoints are included with an empty list ([]) for `label`.

    # Pre-processing
    This module requires access to json files which contain data entries for each
    labeler-split pair (i.e., 6 json files can be generated with the available 2
    labelers and 3 splits). These json files are generated by extracting relevant
    information from existing csv files in the original dataset, and the path to where
    the dataset is stored locally. Please refer to `CreateJSONFiles` class for more
    information.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "validate", "test"],
        labeler: Literal["chexpert", "negbio", "double_image", "single_image"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[Callable[[str], torch.Tensor]] = None,
        include_report: bool = False,
    ) -> None:
        """Initialize the dataset."""
        # input validation
        all_splits = ["train", "validate", "test"]
        if split not in all_splits:
            raise ValueError(
                f"Split {split} is not available. Valid splits are {all_splits}."
            )

        all_labelers = ["chexpert", "negbio", "double_image", "single_image"]
        if labeler not in all_labelers:
            raise ValueError(
                f"Labeler {labeler} is not available. Valid splits are {all_labelers}."
            )

        if transform is not None and not callable(transform):
            raise ValueError("`transform` is not callable.")

        self.image_root = root_dir
        data_path = os.path.join(
            root_dir,
            "mimic_cxr_"
            + labeler
            + "_"
            + split
            + (".json" if labeler in ["chexpert", "negbio"] else ".csv"),
        )
        if not os.path.exists(data_path):
            raise RuntimeError(f"Entries file is not accessible: {data_path}.")
        self._labeler = labeler

        if self._labeler in ["double_image", "single_image"]:
            df = pd.read_csv(data_path)
            df = df.dropna(subset=["caption"])  # some captions are missing
            self.entries = df.to_dict("records")
        else:
            with open(data_path, "rb") as file:
                entries = json.load(file)

            # remove entries with no label if reports are not requested either
            old_num = len(entries)
            entries_df = pd.DataFrame(entries)
            entries_df = entries_df[entries_df["label"].apply(len) > 0]
            self.entries = entries_df.to_dict("records")
            logger.info(
                f"{old_num - len(entries)} datapoints removed due to lack of a label."
            )

        if transform is not None:
            self.transform = transform
        else:
            self.transform = ToTensor()
        self.tokenizer = tokenizer
        self.include_report = include_report

    def __getitem__(self, idx: int) -> Example:
        """Return all the images and the label vector of the idx'th study."""
        entry = self.entries[idx]
        img_path = entry["image_path"]

        with Image.open(
            img_path
            if self._labeler in ["negbio", "chexpert"]
            else os.path.join(self.image_root, img_path)
        ) as img:
            image = self.transform(img.convert("RGB"))

        example = Example({Modalities.RGB.name: image, EXAMPLE_INDEX_KEY: idx})

        if self._labeler in ["negbio", "chexpert"]:
            example["subject_id"] = entry["subject_id"]
            example["study_id"] = entry["study_id"]
            example["qid"] = entry["qid"]
            if self.include_report:
                with open(entry["report_path"], encoding="utf-8") as file:
                    report = file.read()
                tokenized_report = (
                    self.tokenizer(report) if self.tokenizer is not None else None
                )
                example[Modalities.TEXT.name] = report
                if tokenized_report is not None:
                    example[Modalities.TEXT.name] = tokenized_report
        else:
            example[Modalities.TEXT.name] = entry["caption"]
            tokens = (
                self.tokenizer(entry["caption"]) if self.tokenizer is not None else None
            )
            if tokens is not None:
                if isinstance(tokens, dict):  # output of HFTokenizer
                    assert (
                        Modalities.TEXT in tokens
                    ), f"Missing key `{Modalities.TEXT.name}` in tokens."
                    example.update(tokens)
                else:
                    example[Modalities.TEXT.name] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)


class CreateJSONFiles(object):
    """Required pre-processing to use MIMICIVCXR module.

    `MIMICIVCXR` requires access to json files containing concise information about all
    data entries assigned to a specific split and labeler. `CreateJSONFiles` creates
    these json files from existing csv files in the dataset, and information about
    where the dataset is stored on the system. Each json file contains a list of
    entries; each entry is a dictionary with the following keys:
        `image_path`: str
            Absolute path to the image.
        `report_path`: str
            Absolute path to the textual report.
        `label`: List[int | NaN]
            List of labels assigned to the data sample by the desired labeler.
        `qid`: int
            Query ID. This ID enables tracking the sample in the original dataset.
        `subject_id`: int
            ID of the patient.
        `study_id`: int
            ID of this specific X-ray study of the patient.
        `dicom_id`: int
            ID of the original X-ray DICOM image before it was converted to JPG.

    Parameters
    ----------
    data_root : str
        Path to the root directory of JPG image dataset;
        e.g., `mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0`.
    report_root : str
        Path to the root directory of DICOM image dataset where the textual reports are
        also stored; e.g., `mimic-cxr/physionet.org/files/mimic-cxr/2.0.0`.
    json_root : str
        Directory where the resulting json files will be stored.
    skip_all_nan : bool, default=False
        Skip datapoints with all labels assigned NaN.
    skip_no_label : bool, default=False
        Skip datapoints with no assigned labels; i.e. no entry in the labeler's csv
        file.
    nan_val : float, default=float('nan')
        Value to replace with NaN labels.
    """

    def __init__(
        self,
        data_root: str,
        report_root: str,
        json_root: str,
        skip_all_nan: bool = False,
        skip_no_label: bool = False,
        nan_val: float = float("nan"),
    ) -> None:
        """Initialize the module."""
        for path in [data_root, report_root, json_root]:
            if not os.path.exists(path):
                raise RuntimeError(f"Directory is not accessible: {path}.")

        self.data_root = data_root
        self.report_root = report_root
        self.json_root = json_root
        self.skip_all_nan = skip_all_nan
        self.skip_no_label = skip_no_label
        self.nan_val = nan_val

    def make(
        self,
        labeler: Literal["chexpert", "negbio"],
        split: Literal["train", "validate", "test"],
    ) -> None:
        """Make json file for the given labeler and split.

        Parameters
        ----------
        labeler : {"chexpert", "negbio"}
            Model which was used to generate labels from raw-text reports.
        split : {"train", "validate", "test"}
            Dataset split.
        """
        # input validation
        all_splits = ["train", "validate", "test"]
        if split not in all_splits:
            raise ValueError(
                f"Split {split} is not available. Valid splits are {all_splits}."
            )

        all_labelers = ["chexpert", "negbio"]
        if labeler not in all_labelers:
            raise ValueError(
                f"Labeler {labeler} is not available. Valid splits are {all_labelers}."
            )

        # load the splits file
        split_df = pd.read_csv(
            os.path.join(self.data_root, "mimic-cxr-2.0.0-split.csv.gz"),
            compression="gzip",
        )
        split_df = split_df.loc[split_df["split"] == split]

        # read the labels file
        label_path = os.path.join(
            self.data_root, "mimic-cxr-2.0.0-" + labeler + ".csv.gz"
        )
        label_df = pd.read_csv(label_path, compression="gzip")

        entries = []
        for index, row in tqdm(split_df.iterrows(), total=len(split_df.index)):
            subject_id = "p" + str(int(row["subject_id"]))
            study_id = "s" + str(int(row["study_id"]))
            image_name = row["dicom_id"] + ".jpg"
            image_path = os.path.join(
                self.data_root,
                "files",
                subject_id[:3],
                subject_id,
                study_id,
                image_name,
            )
            report_path = os.path.join(
                self.report_root, "files", subject_id[:3], subject_id, study_id + ".txt"
            )

            if not os.path.exists(image_path):
                print(f"File does not exit {image_path}.")
                continue

            label_row = label_df.query(
                "subject_id==@row['subject_id'] and study_id==@row['study_id']"
            )
            if len(label_row) < 1:
                print(
                    f"No entry with subject_id#{subject_id}, study_id#{study_id} in file {label_path}."
                )
                # drop rows that have no label entry;
                # note: this is different from having all NaN labels
                if self.skip_no_label:
                    continue
                label = np.array([])
            else:
                label = label_row.iloc[0][2:].to_numpy()

            # skip rows that have no assigned labels
            if self.skip_all_nan and pd.isna(label).all() and len(label) > 0:
                continue

            # convert all NaN values to a given value, default is NaN itself
            label[pd.isna(label)] = self.nan_val

            entry = {
                "dicom_id": row["dicom_id"],
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "qid": index,
                "image_path": image_path,
                "report_path": report_path,
                "label": label.tolist(),
            }
            entries.append(entry)

        # dump to file
        output_file = f"{labeler}_{split}.json"
        with open(os.path.join(self.json_root, output_file), "w") as file:
            json.dump(entries, file)

        print(f"Data dumped to {output_file} ({len(entries)} entries).")

    def make_all(self) -> None:
        """Make json files for all labelers and splits.

        MIMIC-IV-CXR contains two labelers ("negbio" and "chexpert") and three splits
        ("train", "validate" and "test"); hence, six json files will be created by this
        method.
        """
        all_labelers = Literal["chexpert", "negbio"]
        all_splits = Literal["validate", "test", "train"]
        for labeler in get_args(all_labelers):
            for split in get_args(all_splits):
                print(
                    f"Creating json file for labeler '{labeler}' and split '{split}'."
                )
                self.make(labeler, split)
