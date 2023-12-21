from argparse import Namespace
from pathlib import Path

import torch
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm, trange

from ood_detection import accelerator
from ood_detection.dataset import prepare_dataset, prepare_dataloader
from ood_detection.model import (
    prepare_model,
    train_epoch,
    eval_epoch,
    eval_epoch_with_ood,
)


def train(args: Namespace):
    set_seed(args.seed)

    train_loader = prepare_dataloader(
        prepare_dataset(
            data_flag=args.dataset,
            split="train",
            ood_keys=args.val_ood_keys + args.test_ood_keys,
        ),
        batch_size=args.batch_size,
    )
    val_loader = prepare_dataloader(
        prepare_dataset(
            data_flag=args.dataset,
            split="val",
            ood_keys=args.test_ood_keys,
        ),
        batch_size=args.batch_size,
    )

    model = prepare_model(train_loader.dataset.num_classes)
    setattr(model, "label_dict", train_loader.dataset.label_dict)

    optimizer: torch.optim.Optimizer = (
        torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=0.90,
            weight_decay=2e-2,
        )
        if args.optimizer != "SGD"
        else torch.optim.Adam(model.parameters(), args.learning_rate)
    )
    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler: CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts(
        optimizer, 8, 2
    )
    optimizer, loss_func, scheduler = accelerator.prepare(
        optimizer, loss_func, scheduler
    )

    best_acc = 0

    log_file = open(args.log_path, "wt")
    if accelerator.is_local_main_process:
        print(
            "epoch,train_loss,train_acc,val_acc,ood_acc,learning_rate",
            file=log_file,
        )
        log_file.flush()

    for epoch in trange(
        args.epochs + 1,
        disable=(not accelerator.is_local_main_process),
        desc="[Train]",
    ):
        log_str = f"[Train]: lr: {optimizer.param_groups[-1]['lr']:.3f} "

        if epoch != 0:
            train_epoch(model, loss_func, train_loader, optimizer)
        accelerator.wait_for_everyone()

        train_loss, train_acc = eval_epoch(model, loss_func, train_loader)
        log_str += (
            f"train: (acc: {train_acc * 100:.2f}% loss: {train_loss:.4f}) "
        )

        val_acc, ood_acc = eval_epoch_with_ood(
            model, args.threshold, val_loader
        )
        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            print(
                epoch,
                train_loss,
                train_acc,
                val_acc,
                ood_acc,
                optimizer.param_groups[-1]["lr"],
                sep=",",
                file=log_file,
            )
            log_file.flush()
            accelerator.save_model(model, Path(args.model_dir) / "last")
            if val_acc > best_acc:
                best_acc = val_acc
                accelerator.save_model(model, Path(args.model_dir) / "best")
        accelerator.wait_for_everyone()
        log_str += f"eval: (acc: {val_acc * 100:.2f}% best: \
{best_acc * 100:.2f}% ood: {ood_acc * 100:.2f}%)"

        if accelerator.is_local_main_process:
            tqdm.write(log_str)

        if epoch != 0:
            scheduler.step()

    log_file.close()
