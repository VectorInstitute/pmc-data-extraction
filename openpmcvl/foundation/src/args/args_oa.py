"""Parse arguments for PMC OA."""

import argparse
import os


def parse_args_oa() -> argparse.Namespace:
    """Declare commandline arguments for the module."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--extraction-dir",
        help="path to the directory where downloaded archives and "
        + "images are extracted before being moved to the data subdirectory",
        default=os.path.join("./", "PMC_OA"),
    )
    parser.add_argument(
        "-r",
        "--num-retries",
        help="number of retries for failed downloads before giving up",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--volumes",
        help="determine the volumes to fetch from PMC OA",
        nargs="+",
        default=[0],
        type=int,
    )
    parser.add_argument(
        "--license-type",
        help="type of license of the downloaded papers. options = [comm, noncomm, other]",
        default="comm",
    )

    return parser.parse_args()
