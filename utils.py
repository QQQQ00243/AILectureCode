import torch
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


def get_device(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Using {}\n", device)
    return device


def train(
    model,
    device,
    criterion,
    optimizer,
    train_loader,
):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)

        acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
        train_acc += acc

        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def validate(
    model,
    device,
    criterion,
    val_loader
):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            val_acc += pred.eq(target.view_as(pred)).sum().item() / len(data)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


def get_dataloader(
    batch_size,
    test_batch_size,
    train_dataset,
    test_dataset,
    num_workers,
    valid_split,
):
    num_train = len(train_dataset)
    idx = list(range(num_train))
    split = int(valid_split * num_train)
    '''
    train_idx, valid_idx = idx[split:], idx[:split]
    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(train_dataset, valid_idx)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    '''
    np.random.shuffle(idx)
    split = int(np.floor(valid_split * num_train))
    train_idx, valid_idx = idx[split:], idx[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    # load validation data in batches
    valid_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader
