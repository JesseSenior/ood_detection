from argparse import ArgumentParser, Namespace

import torchvision
from torch.backends import cudnn

torchvision.disable_beta_transforms_warning()

from ood_detection.train import train
from ood_detection.test import test
from ood_detection import dataset
from ood_detection import model


def parse_args() -> Namespace:
    """Handling command-line input."""
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["download", "train", "eval"],
        required=True,
        help='Mode should be either "download", "train" or "eval"',
    )

    parser.add_argument(
        "-e",
        "--epochs",
        action="store",
        default=128,
        type=int,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        action="store",
        default=128,
        type=int,
        help="Size of mini batch.",
    )
    parser.add_argument(
        "--threshold",
        action="store",
        default=0.8,
        type=float,
        help="Threshold for OOD.",
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        action="store",
        default="Adam",
        type=str,
        choices=["Adam", "SGD"],
        help="Optimizer used to train the model.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        action="store",
        default=2e-3,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        action="store",
        default=0,
        type=int,
        help="Random Seed.",
    )

    # Path
    parser.add_argument(
        "--log-path", type=str, default="./log.csv", help="Path to log file."
    )

    parser.add_argument(
        "--model-dir", type=str, default="./weights", help="Model save dir."
    )

    parser.add_argument(
        "--dataset", type=str, default="pathmnist", help="Dataset name."
    )
    # ODD keys
    parser.add_argument(
        "--val-ood-keys",
        nargs="+",
        default=[],
        help="List of keys treated as OOD on validation. Note that both OOD \
keys will not being trained.",
    )
    parser.add_argument(
        "--test-ood-keys",
        nargs="+",
        default=[],
        help="List of keys treated as OOD on test. Note that both OOD keys will\
 not being trained.",
    )
    return parser.parse_args()


def download(args):
    model.download()
    dataset.download(args.dataset)


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    assert (
        len(args.val_ood_keys) > 0
    ), "At least one OOD key for validation is required."
    assert (
        len(args.test_ood_keys) > 0
    ), "At least one OOD key for test is required."
    if args.mode == "download":
        download(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        test(args)
