import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets

from loguru import logger
from utils import get_dataloader, train, validate, get_device


def get_idx_to_class(class_to_idx:dict):
    idx_to_class = {}
    for key, val in class_to_idx.items():
        idx_to_class[val] = key
    return idx_to_class


def make_dir(args):
    if not os.path.exists(args.dataset_root):
        os.mkdir(args.dataset_root)
    if not os.path.exists(args.ckpts_dir):
        os.mkdir(args.ckpts_dir)
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)
    if not os.path.exists(args.logs_dir):
        os.mkdir(args.logs_dir)


def getMNISTloader(
    root,
    download,
    batch_size,
    test_batch_size,
    valid_split,
):
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        valid_split=valid_split,
    )
    return train_loader, val_loader, test_loader



def getFashionMNISTloader(
    root,
    download,
    batch_size,
    test_batch_size,
    valid_split,
):
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )
    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        valid_split=valid_split,
    )
    return train_loader, val_loader, test_loader


def fit(
    model: nn.Module,
    crit,
    epochs,
    init_lr,
    ckpt_file,
    train_loader,
    val_loader,
):
    device = get_device()
    model = model.to(device)
    train_loss, train_acc, val_loss, val_acc = [[] for _ in range(4)]
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    for epoch in range(1, epochs+1):
        # training
        train_loss_, train_acc_ = train(
            model=model,
            device=device,
            criterion=crit,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        train_loss.append(train_loss_)
        train_acc.append(train_acc_)

        # validation
        val_loss_, val_acc_ = validate(
            model=model,
            device=device,
            criterion=crit,
            val_loader=val_loader,
        )
        val_loss.append(val_loss_)
        val_acc.append(val_acc_)

        logger.info(f"Train Epoch: {epoch} / {epochs} LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Train Loss: {train_loss_:.5f}\tTrain Accuracy: {train_acc_:.5f}")
        logger.info(f"Valid Loss: {val_loss_:.5f}\tValid Accuracy: {val_acc_:.5f}\n")

    logger.info(f"Saving model to {ckpt_file}\n")
    torch.save(model.state_dict(), ckpt_file)
    history = {
        "train_history": {"train_accuracy": train_acc, "train_loss": train_loss},
        "val_history": {"val_accuracy": val_acc, "val_loss": val_loss},
    }
    return history


