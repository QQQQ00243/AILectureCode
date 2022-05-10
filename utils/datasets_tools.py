import abc

import matplotlib.pyplot as plt
import numpy as np

from utils.plot_tools import plot_samples
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data.sampler import SubsetRandomSampler


class DatasetsLoader:
    def __init__(
            self,
            datasets_dir,
            batch_size,
            test_batch_size,
            datasets_name,
            augmentation,
            download=False,
            valid_split=0.2,
            num_workers=4,
            idx_to_class=None
    ):
        self.datasets_dir = datasets_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.datasets_name = datasets_name
        self.augmentation = augmentation
        self.download = download
        self.valid_split = valid_split
        self.num_workers = num_workers
        self.idx_to_class = idx_to_class
        self.train_loader, self.val_loader, self.test_loader = self._get_dataloader()

    def _get_idx_to_class(self):
        dataset = getattr(
            datasets,
            self.datasets_name
        )(root=self.datasets_dir, download=self.download)

        class_to_idx = dataset.class_to_idx
        classes = list(class_to_idx.keys())
        indices = list(class_to_idx.values())

        self.idx_to_class = dict(zip(indices, classes))
        return self.idx_to_class

    @abc.abstractmethod
    def _get_augmenter(self):
        pass

    def _get_transform(self):
        if not self.augmentation:
            return T.ToTensor()
        else:
            print('[INFO]Using augmentation.')
            return T.Compose([self._get_augmenter(), T.ToTensor()])

    def _get_dataloader(self):
        # choose the training and test datasets_tools
        train_data = getattr(datasets, self.datasets_name)(
            root=self.datasets_dir,
            train=True,
            download=self.download,
            transform=self._get_transform(),
        )
        test_data = getattr(datasets, self.datasets_name)(
            root=self.datasets_dir,
            train=False,
            download=False,
            transform=T.ToTensor(),
        )

        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_split * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # load training data in batches
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=0
        )
        # load validation data in batches
        valid_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
        )
        # load test data in batches
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
        return train_loader, valid_loader, test_loader

    def plot_samples(
            self,
            title,
            img_path,
            num_per_row,
            num_instances,
            title_size=25,
            subtitle_size=15,
    ):
        dataloader = self.test_loader
        plot_samples(
            path=img_path,
            title=title,
            dataloader=dataloader,
            num_per_row=num_per_row,
            num_instances=num_instances,
            title_size=title_size,
            subtitle_size=subtitle_size,
            idx_to_class=self._get_idx_to_class(),
        )

    def show_augmentaion(
            self,
            path,
            title,
            num_instances,
            num_augmentations,
            title_size=30,
            subtitle_size=25,
    ):
        dataset = getattr(datasets, self.datasets_name)(
            root=self.datasets_dir,
            train=True,
            download=self.download,
        )
        augmenter = self._get_augmenter()

        plt.rcParams["savefig.bbox"] = 'tight'
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["figure.figsize"] = [10, 10]
        nrows = num_instances
        ncols = num_augmentations + 1
        plt.subplots(nrows=nrows, ncols=ncols)
        plt.figtext(0.5, 1.08, title, fontsize=title_size, ha='center')
        plt.figtext(0.03, 1, 'original', fontsize=subtitle_size)
        plt.figtext(0.4, 1, 'after augmentation', fontsize=subtitle_size)
        for i in range(nrows):
            plt.axis('off')
            img = dataset[i][0]
            plt.subplot(nrows, ncols, i*ncols+1)
            plt.imshow(img)
            for j in range(1, ncols):
                plt.axis('off')
                plt.subplot(nrows, ncols, i*ncols+j+1)
                plt.imshow(augmenter(img))
        plt.savefig(path)
        plt.show()
