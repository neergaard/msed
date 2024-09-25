import sys
from argparse import ArgumentParser
from pathlib import Path

from msed.utils.logger import get_logger

logger = get_logger()


def check_and_return_args():
    parser = ArgumentParser("MSED: multimodal sleep event detection using deep learning")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to directory containing data files. Supply this or the `data-file` argument.",
    )
    parser.add_argument("--match-pattern", type=str, help="Pattern to match in filenames (optional)")
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory to save predictions")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Device to use for training and inference",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to model directory",
        default="models/splitstream",
    )
    args = parser.parse_args()
    print_args(args)
    args.target_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Saving predictions to "{args.target_dir}"')
    return args


def print_args(args):
    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == len(vars(args)) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")
