from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.manifold import TSNE
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ood_detection import accelerator
from ood_detection.dataset import prepare_dataset, prepare_dataloader
from ood_detection.model import prepare_model

plt.rcParams["axes.grid"] = True
plt.rcParams["figure.dpi"] = 600
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["font.family"] = "serif"
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.bbox"] = "tight"


def plot_log(filename):
    print("Plotting the training curve")
    data = pd.read_csv(filename)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.grid(True)  # Add grid to ax1

    sns.lineplot(
        data=data,
        x="epoch",
        y="train_loss",
        label="Train Loss",
        ax=ax1,
        color="c",
    )
    ax2 = ax1.twinx()

    # ax2.grid(True)  # Add grid to ax2

    sns.lineplot(
        data=data,
        x="epoch",
        y="train_acc",
        label="Train Accuracy",
        ax=ax2,
        color="g",
    )
    sns.lineplot(
        data=data,
        x="epoch",
        y="val_acc",
        label="Validation Accuracy",
        ax=ax2,
        color="y",
    )
    sns.lineplot(
        data=data,
        x="epoch",
        y="ood_acc",
        label="OoD Accuracy",
        ax=ax2,
        color="r",
    )

    # ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
    # ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
    ax1.set_yticks(np.linspace(0, 2, 9))
    ax2.set_yticks(np.linspace(0, 1, 9))

    ax1.set_ylim((0, 2.2))
    ax2.set_ylim((0, 1.1))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2.set_ylabel("Accuracy")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend([], [], frameon=False)
    ax2.legend(lines + lines2, labels + labels2)
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1.1, 1))

    plt.title("Training Progress")
    plt.savefig(str(Path(filename).parent / "training_progress.pdf"))

    # Plot learning rate separately
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="epoch", y="learning_rate", label="Learning Rate")
    plt.title("Learning Rate Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)  # Add grid to this plot
    plt.savefig(str(Path(filename).parent / "learning_rate.pdf"))
    print(f"Figures saved to {str(Path(filename).parent)}")


@torch.no_grad()
def plot_graphs(model, dataloader, save_dir):
    print("Plotting the testing results.")
    model_state_key = (
        model.module if isinstance(model, DDP) else model
    ).label_dict
    inv_model_state_key = {
        v: k
        for k, v in (
            model.module if isinstance(model, DDP) else model
        ).label_dict.items()
    }  # class_name(str) -> class_id(int)
    pred_true = {
        v: [list(), list()]
        for v in dataloader.dataset.label_dict.values()
        if v in inv_model_state_key.keys()
    }  # class_name(str) -> (y_pred(nparrays),y_true(nparrays))
    assert "OOD" not in pred_true.keys(), "?!"
    pred_true["OOD"] = [list(), list()]
    features = []
    labels = []

    for source, targets in tqdm(
        dataloader,
        disable=(not accelerator.is_local_main_process),
        desc="[Test]",
    ):
        output: torch.Tensor = model(source)
        features.append(output)
        output = F.softmax(output, dim=1)
        conf, pred = torch.max(output, dim=1)
        for key, value in dataloader.dataset.label_dict.items():
            targets[targets == key] = (
                -1
                if value not in inv_model_state_key.keys()
                else inv_model_state_key[value]
            )
        labels.append(targets)
        for v in dataloader.dataset.label_dict.values():
            if v in inv_model_state_key.keys():
                y_true = targets == inv_model_state_key[v]
                y_pred = conf.where(pred == inv_model_state_key[v], 0)
            else:
                v = "OOD"
                y_true = targets == -1
                y_pred = 1 - conf
            pred_true[v][0].append(y_pred)
            pred_true[v][1].append(y_true)
    features = torch.cat(features).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    for key in pred_true.keys():
        pred_true[key][0] = torch.cat(pred_true[key][0]).cpu().numpy()
        pred_true[key][1] = torch.cat(pred_true[key][1]).cpu().numpy()

    plt.figure()
    for key in pred_true.keys():
        y_pred, y_true = pred_true[key]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        sns.lineplot(
            x=fpr,
            y=tpr,
            label=f"{key} (area = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(str(Path(save_dir) / "ROC.pdf"))

    plt.figure()
    for key in pred_true.keys():
        y_pred, y_true = pred_true[key]
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        sns.lineplot(
            x=recall,
            y=precision,
            label=f"{key}",
        )

    plt.plot([0, 1], [1, 0], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(str(Path(save_dir) / "P-R.pdf"))

    tsne = TSNE(random_state=0).fit_transform(features)
    plt.figure()
    max_points = min(tsne.shape[0], 512)
    sns.scatterplot(
        x=tsne[:max_points, 0],
        y=tsne[:max_points, 1],
        hue=labels[:max_points] + 1,
        palette="Spectral",
    )
    legend = plt.gca().legend_
    labels = [t.get_text() for t in legend.texts]
    plt.legend(
        loc="lower right",
        labels=["OOD"] + [model_state_key[int(x) - 1] for x in labels[1:]],
    )
    plt.title("t-SNE Plot")
    plt.savefig(str(Path(save_dir) / "t-SNE.pdf"))
    print(f"Figures saved to {str(Path(save_dir))}")


def test(args: Namespace):
    """评估的主函数"""
    set_seed(args.seed)
    train_dataset = prepare_dataset(
        data_flag=args.dataset,
        split="train",
        ood_keys=args.val_ood_keys + args.test_ood_keys,
    )
    test_loader = prepare_dataloader(
        prepare_dataset(
            data_flag=args.dataset,
            split="test",
            ood_keys=args.val_ood_keys,
        ),
        batch_size=args.batch_size,
    )

    model = prepare_model(
        train_dataset.num_classes,
        state_dict_path=Path(args.model_dir) / "best" / "model.safetensors",
    )
    model.eval()
    setattr(
        model,
        "label_dict",
        train_dataset.label_dict,
    )

    plot_log(args.log_path)
    plot_graphs(model, test_loader, str(Path(args.log_path).parent))
