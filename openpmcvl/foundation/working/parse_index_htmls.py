"""Extract a list of all article packages and their size on disk from `index.html` files.

`index.html` files can be downloaded from PubMedCentral's FTP server through the following command:
```bash
wget -c -r -np https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/
```

The following command will download all article packages as well as `index.html` files. 
Note: You must have about 33.3TB of free space to download all articles.
```bash
wget -c -r -np -e robots=off https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/
```
"""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the `oa_package` directory.",
                        default="/projects/multimodal/datasets/pmc_articles/ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/")
    args = parser.parse_args()
    print(args)
