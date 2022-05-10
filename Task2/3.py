import os
import argparse

from torch import optim
from torchvision import transforms as T
from basic_models.cnn import LeNet_3Channel as LeNet
from utils.analyze_utils import Analyzer
from utils.training_utils import Trainer
from utils.datasets_tools import DatasetsLoader
from utils.checkpoints import EarlyStop
from utils.name_tools import get_img_path


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR - LeNet')
    parser.add_argument('--ExpNo', default=3,
                        help='Number of Experiment')
    # --------------------configurations of datasets_tools-------------------------------
    parser.add_argument('--datasets_name', default='CIFAR10',
                        help='name of the dataset')
    parser.add_argument('--download', default=False,
                        help='enable download datasets')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--valid-split', type=float, default=0.2,
                        help='split ratio of validation set')
    parser.add_argument('--augmentation', type=bool, default=False,
                        help='enable augmentation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers when creating data loader')
    parser.add_argument('--classes', type=list, default=10,
                        help='list containing classes')
    parser.add_argument('--idx-to-class',
                        help="dictionary {idx: name of the idx}")

    # ---------------------configurations of training------------------------------
    parser.add_argument('--training', default=True,
                        help='enable training')
    parser.add_argument('--case', default=1,
                        help='number of case')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--log-interval', type=int, default=128, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--criterion_name', default='CrossEntropyLoss',
                        help='name of loss function')
    parser.add_argument('--optimizer-name', type=str, default='Adam',
                        help='name of the optimizer')

    # ---------------------configurations of learning rate scheduler------------------------
    parser.add_argument('--lrScheduler', default=True,
                        help='store learing rate scheduler')
    parser.add_argument('--init-lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='gamma parameter of lr scheduler')

    # ---------------------configurations of early stopping------------------------------
    parser.add_argument('--EarlyStop', default=True,
                        help='enable early stop')
    parser.add_argument('--patience', default=5,
                        help='patience of early stop')
    parser.add_argument('--min-delta', default=0,
                        help='minimum delta to best performace')

    # ---------------------configurations of analyzing------------------------------
    parser.add_argument('--plot-samples', default=True,
                        help='enable plot samples')
    parser.add_argument('--analyzing', default=False,
                        help='enable analyzing')

    # ---------------------configurations of saving------------------------------
    parser.add_argument('--datasets-dir', type=str, default='../datasets',
                        help='directory storing datasets_tools')
    parser.add_argument('--models-dir', type=str, default='./models',
                        help='directory to save models')
    parser.add_argument('--imgs-dir', type=str, default='./imgs',
                        help='directory to save images')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.ExpNo = 3
    args.case = 3

    model = LeNet()
    args.training = True
    args.analyzing = True
    args.plot_samples = False
    args.augmentation = True

    # -------------- Initialize Dataloader --------------
    class CIFARLoader(DatasetsLoader):
        # rewrite augmentation
        def _get_augmenter(self):
            policy = T.AutoAugmentPolicy.CIFAR10
            return T.AutoAugment(policy)

    cifarloader = CIFARLoader(
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        datasets_name=args.datasets_name,
        augmentation=args.augmentation,
        download=args.download,
        valid_split=args.valid_split,
        num_workers=args.num_workers,
        idx_to_class=args.idx_to_class,
    )

    # ------------ plot samples of dataset -------------
    if args.plot_samples:
        title = '{} Examples of {}'.format(args.ExpNo, args.datasets_name)
        cifarloader.plot_samples(
            title=title,
            num_per_row=8,
            num_instances=64,
            title_size=25,
            subtitle_size=15,
            img_path=get_img_path(args.imgs_dir, title),
        )
        title = '{} Illustration of augmentation'.format(args.ExpNo)
        cifarloader.show_augmentaion(
            title=title,
            num_instances=5,
            num_augmentations=4,
            path=get_img_path(args.imgs_dir, title),
        )

    # --------------- set paths --------------------
    model_path = os.path.join(
        args.models_dir,
        '{}_{}.pt'.format(args.ExpNo, args.case)
    )
    log_path = '{}_{}.pkl'.format(args.ExpNo, args.case)
    cm_title = '{}_{} Confusion Matrix'.format(args.ExpNo, args.case)
    mistakes_title = '{}_{} Mistaken Instances True-(False)'.format(args.ExpNo, args.case)
    history_title = '{}_{} Training History of LeNet on CIFAR10'.format(args.ExpNo, args.case)

    # ------------ [CASE:1] early stopping only -------------
    if args.case == 1:
        args.lrScheduler = None
        args.augmentation = False

    # ------------ [CASE:2] learning rate scheduler only -------------
    elif args.case == 2:
        args.lrScheduler = optim.lr_scheduler.ExponentialLR
        args.augmentation = False

    # ------------ [CASE:3] learning rate scheduler + Image Augmentation --------
    elif args.case == 3:
        args.lrScheduler = optim.lr_scheduler.ExponentialLR
        args.augmentation = True

    # ------------ START -------------
    if args.training:
        print('[INFO]Training...\n')
        train_loader, val_loader, test_loader = cifarloader.get_dataloader()
        earlystop = EarlyStop(patience=args.patience, path=model_path)
        trainer = Trainer(
            model=model,
            log_path=log_path,
            model_path=model_path,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        trainer.fit()
        trainer.eval()
        trainer.plot_history(
            title=history_title,
            img_path=get_img_path(args.imgs_dir, history_title)
        )
    if args.analyzing:
        print('Analyzing...\n')
        _, _, test_loader = cifarloader.get_dataloader()
        analyzer = Analyzer(
            args=args,
            model=model,
            model_path=model_path,
            dataloader=test_loader
        )
        analyzer.plot_mistakes(
            num_per_row=6,
            num_instances=36,
            title=mistakes_title,
            path=get_img_path(args.imgs_dir, mistakes_title)
        )
        analyzer.plot_confusion_matrix(
            title=cm_title,
            path=get_img_path(args.imgs_dir, cm_title)
        )

    print('Finished!\n')


if __name__ == "__main__":
    main()
