"""Test article parser."""
import pathlib
from ..src.parser.parse_oa import get_volume_info
from ..src.utils.io import write_jsonl
from ..src.data.data_oa import OA_LINKS


def test_extract_volume_info():
    """Extract <img, caption> pairs from a given volume."""
    print("\033[32mParse PMC documents\033[0m")

    volume_info = get_volume_info(volumes=[0], extraction_dir=pathlib.Path("./PMC_OA"))
    print(f"Num of figs in volumes: {len(volume_info)}")
    write_jsonl(data_list=volume_info, save_path="./volume0.jsonl")


def print_oa_links():
    """Print FTP addresses from where articles are downloaded."""
    print(OA_LINKS)


if __name__ == "__main__":
    test_extract_volume_info()
    print_oa_links()
    print("Finished. Please check results for correctness.")
