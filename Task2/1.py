import os
import argparse

from torch import nn
from basic_models.cnn import LeNet
from utils.analyze_utils import Analyzer
from utils.training_utils import Trainer
from datasets_tools.data_loader import datasets_info


def get_args():
    parser = argparse.ArgumentParser(description='LeNet - MNIST')
    parser.add_argument('--ExpNo', default=1,
                        help='Number of Experiment')
    # --------------------configurations of datasets_tools-------------------------------
    parser.add_argument('--datasets_tools', default='MNIST',
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
    args.ExpNo = 1

    model = LeNet()
    args.training = True
    args.analyzing = True
    model_path = os.path.join(
            args.models_dir,
            '{}_LeNetMNIST.pt'.format(args.ExpNo)
        )

    # --------- get basic information of dataset ------------
    args.classes, args.idx_to_class = datasets_info(
        datasets_name=args.datasets, dir=args.datasets_dir,
    )

    trainer = Trainer(model=model, args=args, path=model_path)

    if args.training:
        print('Training...\n')
        trainer.fit()
        trainer.eval()
        trainer.plot_history(
            title='Training History of LeNet on MNIST',
            path=os.path.join(
                args.imgs_dir, '{}_TrainingHistoryOfLeNetonMNIST.svg'.format(args.ExpNo)
            )
        )

    if args.analyzing:
        print('Analyzing...\n')
        analyzer = Analyzer(
            args=args, model=model,
            model_path=model_path, dataloader=trainer.test_loader)
        analyzer.plot_mistakes(
            num_per_row=6, num_instances=36,
            title='{}_Mistaken Instances True-(False)'.format(args.ExpNo),
            path=os.path.join(
                args.imgs_dir,
                '{}_MistakenInstances.svg'.format(args.ExpNo)
            )
        )
        analyzer.plot_confusion_matrix(
            title='{}_Confusion Matrix'.format(args.ExpNo),
            path=os.path.join(
                args.imgs_dir,
                '{}_ConfusionMatrix.svg'.format(args.ExpNo)
            )
        )


if __name__ == "__main__":
    main()
