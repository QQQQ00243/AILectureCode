import os
import argparse

from torch import nn
from utils.analyze_utils import Analyzer
from utils.training_utils import Trainer
from utils.plot_tools import plot_samples


def get_args():
    parser = argparse.ArgumentParser(description='MNIST -- FC')
    # --------------------configurations of datasets_tools-------------------------------
    parser.add_argument('--dataset', default='MNIST',
                        help='name of the dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1024)')

    # ---------------------configurations of training------------------------------
    parser.add_argument('--training', default=True,
                        help='enable training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--log-interval', type=int, default=128, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--criterion', default=nn.CrossEntropyLoss(),
                        help='name of loss function')
    parser.add_argument('--optimizer-name', type=str, default='Adam',
                        help='name of the optimizer')
    parser.add_argument('--init-lr', type=float, default=0.001,
                        help='initial learning rate')

    # ---------------------configurations of analyzing------------------------------
    parser.add_argument('--plot-samples', default=True,
                        help='enable plot samples')
    parser.add_argument('--analyzing', default=False,
                        help='enable analyzing')

    # ---------------------configurations of saving------------------------------
    parser.add_argument('--models-dir', type=str, default='./models',
                        help='directory to save models')
    parser.add_argument('--model-path', type=str, default='1.pt',
                        help='path tp save model')
    parser.add_argument('--imgs-dir', type=str, default='./imgs',
                        help='directory to save images')

    args = parser.parse_args(args=[])
    return args


def get_model(in_shape, width_hiddens, classes, act_name):
    in_features = 1
    for i in in_shape:
        in_features *= i

    act_layer = getattr(nn, act_name)()

    layers = [nn.Flatten()]
    for out_features in width_hiddens:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(act_layer)
        in_features = out_features
    layers.append(nn.Linear(in_features, classes))
    return nn.Sequential(*layers)


def main():
    args = get_args()

    classes = 10
    model = get_model(
        in_shape=[28, 28], width_hiddens=[64, 64],
        classes=classes, act_name='Sigmoid'
    )
    args.plot_samples = False
    args.training = False
    args.analyzing = True
    trainer = Trainer(model=model, args=args)

    if args.plot_samples:
        plot_samples(
            num_instances=64, num_per_row=8, dataloader=trainer.train_loader,
            title='examples', path=os.path.join(args.imgs_dir, '1_examples.svg')
        )

    if args.training:
        print('Training...\n')
        trainer.fit()
        trainer.eval()
        trainer.plot_history(
            title='1_training_history',
            PATH=os.path.join(args.imgs_dir, '1_training_history.svg'),
        )

    if args.analyzing:
        print('Analyzing...\n')
        analyzer = Analyzer(
            args=args, classes=classes, model=model,
            dataloader=trainer.test_loader,
            model_path=os.path.join(args.models_dir, args.model_path)
        )
        analyzer.plot_mistakes(
            num_per_row=6, num_instances=36,
            title='Mistaken Instances True-(False)',
            path=os.path.join(args.imgs_dir, '1_MistakenInstances.svg')
        )
        analyzer.plot_confusion_matrix(
            title='Confusion Matrix',
            path=os.path.join(args.imgs_dir, '1_ConfusionMatrix.svg')
        )


if __name__ == "__main__":
    main()
