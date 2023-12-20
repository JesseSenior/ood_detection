from typing import Callable

import torch
import torch.nn.functional as F
import safetensors.torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    resnet18,
    ResNet18_Weights,
)
from tqdm import tqdm

from ood_detection import accelerator


def download():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)


def prepare_model(
    num_classes: int = 1000, method="resnet18", state_dict_path: str = None
) -> nn.Module:
    with accelerator.local_main_process_first():
        if method == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif method == "vit":
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            model.heads = nn.Linear(
                in_features=model.heads[0].in_features, out_features=num_classes
            )
    if state_dict_path is not None:
        safetensors.torch.load_model(model, state_dict_path)
    return accelerator.prepare(model)


@torch.enable_grad()
def train_epoch(
    model: nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> None:
    """训练一轮"""
    model.train()
    for source, targets in tqdm(
        dataloader,
        disable=(not accelerator.is_local_main_process),
        desc="<Train>",
        leave=False,
    ):
        optimizer.zero_grad()
        output: torch.Tensor = model(source)
        loss = loss_func(output, targets)
        accelerator.backward(loss)
        optimizer.step()


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
) -> tuple[float, float]:
    """在指定测试集上测试模型的损失和准确率"""
    model.eval()
    correct_sum, loss_sum, cnt_samples = 0, 0.0, 0
    for source, targets in tqdm(
        dataloader,
        disable=(not accelerator.is_local_main_process),
        desc="<Eval>",
        leave=False,
    ):
        output: torch.Tensor = model(source)
        loss = loss_func(output, targets)
        prediction: torch.Tensor = accelerator.gather_for_metrics(
            output.argmax(dim=1) == targets
        )  # type: ignore
        correct_sum += prediction.sum().item()
        loss_sum += loss.item()
        cnt_samples += len(prediction)
    return loss_sum / len(dataloader), correct_sum / cnt_samples


@torch.no_grad()
def eval_epoch_with_ood(
    model: nn.Module,
    threshold: float,
    dataloader: DataLoader,
) -> tuple[float, float]:
    """在指定测试集上测试模型的损失和准确率（考虑分布外标签）"""
    model.eval()
    correct_sum, ood_sum, cnt_samples, cnt_ood = 0, 0, 0, 0
    inv_model_state_key = {
        v: k
        for k, v in (
            model.module if isinstance(model, DDP) else model
        ).label_dict.items()
    }
    for source, targets in tqdm(
        dataloader,
        disable=(not accelerator.is_local_main_process),
        desc="<Eval (with OOD)>",
        leave=False,
    ):
        output: torch.Tensor = model(source)
        output = F.softmax(output, dim=1)
        conf, pred = torch.max(output, dim=1)
        pred[conf < threshold] = -1
        for key, value in dataloader.dataset.label_dict.items():
            targets[targets == key] = (
                -1
                if value not in inv_model_state_key.keys()
                else inv_model_state_key[value]
            )
        prediction: torch.Tensor = accelerator.gather_for_metrics(
            pred == targets
        )  # type: ignore
        targets: torch.Tensor = accelerator.gather_for_metrics(targets)
        correct_sum += prediction.sum().item()
        ood_sum += prediction[targets == -1].sum().item()
        cnt_samples += len(prediction)
        cnt_ood += (targets == -1).count_nonzero().item()
    return correct_sum / cnt_samples, ood_sum / cnt_ood
