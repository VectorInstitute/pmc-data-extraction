# **Granular Pipeline**
Our goal is to create a finegrained dataset of biomedical subfigure-subcaption pairs from the raw dataset of PMC figure-caption pairs. We assume that a dataset of PMC figure-caption pairs, e.g. PMC-17M, is already downloaded, formatted as a directory of JSONL files and a directory of image .jpg files. Note that all .sh files require you to pass in the JSONL numbers from the PMC dataset as arguments.

Sample command:
```bash
sbatch openpmcvl/granular/pipeline/preprocess.sh 0 1 2 3 4 5 6 7 8 9 10 11
```


## **1. Preprocess**
> **Code:**  `preprocess.py & preprocess.sh` <br>
> **Input:** Directory of figures and PMC metadata in JSONL format <br>
> **Output:** Filtered figure-caption pairs in JSONL format (`${num}_meta.jsonl`) <br>

- Filter out figure-caption pairs that are not .jpg images, missing, or corrupted.
- Filter for figure-caption pairs that contain target biomedical keywords.

Each datapoint contains the following fields:
- `id`: A unique identifier for the figure-caption pair.
- `PMC_ID`: The PMC ID of the article.
- `caption`: The caption of the figure.
- `image_path`: The path to the image file.
- `width`: The width of the image in pixels.
- `height`: The height of the image in pixels.
- `media_id`: The ID of the media file.
- `media_url`: The URL of the media file.
- `media_name`: The name of the media file.
- `keywords`: The keywords found in the caption.
- `is_medical`: Whether the caption contains any target biomedical keywords.
<br><br>

This script saves the output both as a directory of processed JSONL files and a merged JSONL file. The former is used in the next step of the pipeline.
<br><br>


## **2. Subfigure Extraction**
> **Code:**  `subfigure.py & subfigure.sh` <br>
> **Input:**  Filtered figure-caption pairs in JSONL format (`${num}_meta.jsonl`) <br>
> **Output:**  Directory of subfigure jpg files, and subfigure metadata in JSONL format (`${num}_subfigures.jsonl`) <br>

- Breakdown compound figures into subfigures.
- Keep original figure for non-compound figures or if an exception occurs.

Each datapoint contains the following fields:

When a subfigure is successfully detected and separated:
- `id`: Unique identifier for the subfigure (format: {source_figure_id}_{subfigure_number}.jpg)
- `source_fig_id`: ID of the original compound figure
- `PMC_ID`: PMC ID of the source article
- `media_name`: Original filename of the compound figure
- `position`: Coordinates of subfigure bounding box [(x1,y1), (x2,y2)]
- `score`: Detection confidence score
- `subfig_path`: Path to saved subfigure image

When subfigure extraction fails:
- `id`: Generated ID that would have been used
- `source_fig_id`: ID of the original figure
- `PMC_ID`: PMC ID of the source article
- `media_name`: Original filename

This script saves extracted subfigures as .jpg files in the target directory. Metadata for each subfigure is stored in separate JSONL files, with unique IDs that link back to the original figure-caption pairs in the source JSONL files.
<br><br>


## **3. Subcaption Extraction**
> **Code:**  `subcaption.ipynb | subcaption.py & subcaption.sh` <br>
> **Input:**  PMC metadata in JSONL format <br>
> **Output:**  PMC metadata in JSONL format with subcaptions <br>

- Extract subcaptions from captions.
- Keep original caption if the caption cannot be split into subcaptions.

While this pipeline works, its slow as it goes through API calls one by one. There is a notebook (`subcaption.ipynb`) using batch API calls to speed it up. It's highly recommended to use the notebook instead of this script.
<br><br>


## **4. Classification**
> **Code:**  `classify.py & classify.sh` <br>
> **Input:**  Subfigure metadata in JSONL format (`${num}_subfigures.jsonl`) <br>
> **Output:**  Subfigure metadata in JSONL format (`${num}_subfigures_classified.jsonl`) <br>

- Classify subfigures and include metadata about their class.

The following fields are added to each datapoint:
- `is_medical_subfigure`: Whether the subfigure is a medical subfigure.
- `medical_class_rank`: The model's confidence in the medical classification.

This script preserves all subfigures and adds an `is_medical_subfigure` boolean flag to identify medical subfigures. It also includes a `medical_class_rank` field indicating the model's confidence in the medical classification.
<br><br>


## **5. Alignment**
> **Code:**  `align.py & align.sh` <br>
> **Input:**  Subfigure metadata in JSONL format (`${num}_subfigures_classified.jsonl`) <br>
> **Output:**  Aligned subfigure metadata in JSONL format (`${num}_aligned.jsonl`) <br>

- Find the label associated with each subfigure.
- If no label is found, it means either:
  - The image is a standalone figure (not part of a compound figure)
  - The OCR model failed to detect the subfigure label (e.g. "A", "B", etc.)

The non biomedical subfigures will be removed. The following fields are added to each datapoint:
- `label`: The label associated with the subfigure. (e.g. "Subfigure-A")
- `label_position`: The position of the label in the subfigure.


The outputs from steps 3 and 5 contain labeled subcaptions and labeled subfigures respectively. By matching these labels (e.g. "Subfigure-A"), we can create the final subfigure-subcaption pairs. Any cases where labels are missing or captions couldn't be split will be handled in subsequent steps. Refer to notebook for more details.
<br><br>
