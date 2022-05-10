import os
import argparse

from torch import nn
from experiments.Exp import Exp
from utils.plot_tools import plot_samples
from datasets_tools.data_loader import datasets_info, create_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='FashionMNIST')
    parser.add_argument('--ExpNo', default=4,
                        help='Number of Experiment')
    # --------------------configurations of datasets_tools-------------------------------
    parser.add_argument('--datasets_tools', default='FashionMNIST',
                        help='name of the dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--classes', type=list, default=10,
                        help='list containing classes')
    parser.add_argument('--idx-to-class',
                        help="dictionary {idx: name of the idx}")

    # ---------------------configurations of training------------------------------
    parser.add_argument('--training', default=True,
                        help='enable training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--log-interval', type=int, default=128, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--criterion', default=nn.CrossEntropyLoss(),
                        help='name of loss function')
    parser.add_argument('--optimizer-name', type=str, default='Adam',
                        help='name of the optimizer')
    parser.add_argument('--init-lr', type=float, default=0.001,
                        help='initial learning rate')

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
    parser.add_argument('--datasets_tools-dir', type=str, default='D:/datasets_tools',
                        help='directory storing datasets_tools')
    parser.add_argument('--models-dir', type=str, default='./models',
                        help='directory to save models')
    parser.add_argument('--imgs-dir', type=str, default='./imgs',
                        help='directory to save images')

    args = parser.parse_args(args=[])
    return args


def main():
    args = get_args()

    # --------- get basic information of dataset ------------
    args.classes, args.idx_to_class = datasets_info(
        datasets_name=args.datasets, dir=args.datasets_dir,
    )

    # ------------ plot samples of dataset -------------
    args.plot_samples = False
    if args.plot_samples:
        dataloader, _, _ = create_dataloader(
            datasets_name=args.datasets,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
        )
        plot_samples(
            num_instances=64, num_per_row=8,
            idx_to_class=args.idx_to_class,
            dataloader=dataloader, title='{}_Examples'.format(args.ExpNo),
            path=os.path.join(args.imgs_dir, '{}_Examples.svg'.format(args.ExpNo))
        )

    # ------ missions --------
    mission = 3
    args.training = False
    in_shape = [28, 28]
    # ------ simple fully-connected network with Sigmoid as activation function--------
    if mission == 1:
        args.EarlyStop = False
        args.epochs = 20
        activations = ['Sigmoid']
        regularizations = ['None']
        width_hiddens = [64]
        args.analyzing = False
    # --------- deep fully-connected network with more than two layers ----------------
    # --------- and Sigmoid as activation function ------------------------------------
    if mission == 2:
        args.EarlyStop = False
        args.epochs = 20
        activations = ['Sigmoid']
        regularizations = ['None']
        width_hiddens = [64, 64, 64]
        args.analyzing = False
    # --------- deep fully-connected network with more than two layers ----------------
    # ------ try different activation functions and regularizations tricks ------------
    if mission == 3:
        args.EarlyStop = True
        args.epochs = 30
        activations = ['Sigmoid', 'Tanh', 'ReLU']
        regularizations = ['bn', 'dropout_0.25', 'dropout_0.5', 'None']
        width_hiddens = [64, 64, 64]
        args.analyzing = True

    exp = Exp(
        args=args, num_classes=len(args.classes),
        in_shape=in_shape, width_hiddens=width_hiddens,
        prefix='{}_{}'.format(args.ExpNo, mission),
        log_path='./{}_{}.txt'.format(args.ExpNo, mission),
        activations=activations, regularizations=regularizations
    )
    if args.training:
        exp.run()
    if args.analyzing:
        exp.plot_res(
            title='Activations Functions and Regularizations',
            path=os.path.join(
                args.imgs_dir,
                '{}_Results.svg'.format(args.ExpNo)
            )
        )

    print('Finished!\n')


if __name__ == "__main__":
    main()
