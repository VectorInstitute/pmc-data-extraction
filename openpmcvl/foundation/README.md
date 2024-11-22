# Foundation Package

This package contains tools to download and extract image-text pairs from PubMedCentral Open-access articles.
Moreover, you can create metadata and train-val-test splits with this package.

The code in this package is a modified version of [Build-PMC-OA](https://github.com/WeixiongLin/Build-PMC-OA).
Please refer to the original repository for more information, and cite their paper[[1]](#cite) if you use this package.

- [Foundation Package](#foundation-package)
  - [Installation](#installation)
  - [Limitation](#limitation)
  - [Cite](#cite)

## Installation

1. Setup virtual environment.
After installing the general virtual environment for `openpmcvl` (instructions in the main `README.md`), 
install the specific requirements for this package with
```bash
source path/to/venv/bin/activate
pip install -r openpmcvl/foundation/requirements.txt
```

2. Run the script.

```bash
source path/to/venv/bin/activate  # activate venv
cd openpmcvl/foundation
python src/fetch_oa.py --volumes 0 1 2  # Choose volumes of 0,1,2 only
nohup python -u src/fetch_oa.py --extraction-dir path/to/output/dir --volumes 0 > output.txt  # write output to a file
```


## Limitation
1. Some of the paper are only presented in pdf formart, the figures in those would not be obtained by this pipeline
2. We do not provide the capability to downlaod media files other than images, such as suffix mp4, avi; however, the original repository provides this capability.


## Cite
```bash
@article{lin2023pmc,
  title={PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents},
  author={Lin, Weixiong and Zhao, Ziheng and Zhang, Xiaoman and Wu, Chaoyi and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  journal={arXiv preprint arXiv:2303.07240},
  year={2023}
}
```

