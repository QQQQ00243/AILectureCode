import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from utils.name_tools import join
from utils.analyze_utils import Analyzer
from utils.training_utils import Trainer


def get_model(
        in_shape, width_hiddens, num_classes,
        act_name, batch_norm=False, dropout=None,
):
    if batch_norm and dropout is not None:
        print('Generally BatchNorm and dropout can not be used at the same time')

    in_features = 1
    for i in in_shape:
        in_features *= i

    layers = [nn.Flatten()]
    act_layer = getattr(nn, act_name)()
    for out_features in width_hiddens:
        layers.append(nn.Linear(in_features, out_features))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(act_layer)
        in_features = out_features
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


class Exp:
    def __init__(
            self, args, log_path, prefix,
            activations, regularizations,
            in_shape, width_hiddens, num_classes
    ):
        self.args = args
        self.log_path = log_path
        self.prefix = prefix
        self.activations = activations
        self.regularizations = regularizations
        self.in_shape = in_shape
        self.width_hiddens = width_hiddens
        self.num_classes = num_classes

    def experiment(self, act_name, regularization):
        assert act_name in ['Sigmoid', 'Tanh', 'ReLU']
        assert regularization in ['bn', 'dropout_0.25', 'dropout_0.5', 'None']
        print(
            'Starting experiment with activation {},'
            ' regularization{}\n'.format(act_name, regularization)
        )

        batch_norm, dropout = False, None
        if 'dropout' in regularization:
            dropout = float(regularization.split('_')[1])
            batch_norm = False
        elif regularization == 'bn':
            batch_norm = True

        model = get_model(
            act_name=act_name, num_classes=self.num_classes,
            batch_norm=batch_norm, dropout=dropout,
            in_shape=self.in_shape, width_hiddens=self.width_hiddens,
        )

        prefix = join(self.prefix, '{}_{}'.format(act_name, regularization))

        model_path = os.path.join(self.args.models_dir, prefix + '.pt')

        trainer = Trainer(model=model, args=self.args, path=model_path)
        trainer.fit()
        test_loss, test_acc = trainer.eval()

        trainer.plot_history(
            title=join(prefix, 'TrainingHistory'),
            path=os.path.join(
                self.args.imgs_dir,
                join(prefix, 'TrainingHistory.svg')
            )
        )

        analyzer = Analyzer(
            args=self.args, model=model,
            dataloader=trainer.test_loader, model_path=model_path
        )
        analyzer.plot_mistakes(
            num_per_row=6, num_instances=36,
            title=join(prefix, 'Mistaken Instances True-(False)'),
            path=os.path.join(
                self.args.imgs_dir,
                join(prefix, 'MistakenInstances.svg')
            )
        )
        analyzer.plot_confusion_matrix(
            title=join(prefix, 'Confusion Matrix'),
            path=os.path.join(
                self.args.imgs_dir,
                join(prefix, 'ConfusionMatrix.svg')
            )
        )
        res = {
            'activation': act_name, 'regularization': regularization,
            'test_loss': test_loss, 'test_acc': test_acc,
        }
        return res

    def run(self):
        output_file = open(self.log_path, 'w')
        num_exps = len(self.activations) * len(self.regularizations)
        print(
            '{} activations, {} regularizations, {} experiments in total\n'.format(
                len(self.activations), len(self.regularizations), num_exps)
        )
        i, keys = 1, []
        for regularization in self.regularizations:
            for act_name in self.activations:
                res = self.experiment(act_name=act_name, regularization=regularization)
                if i == 1:
                    for key, _ in res.items():
                        output_file.write(key + ',')
                        keys.append(key)
                    output_file.write('\n')
                for key in keys:
                    output_file.write(str(res[key]) + ',')
                output_file.write('\n')
                print(
                    'Experiments {} of {} with activation ({activation}) and '
                    'regularization ({regularization}) : Test accuracy: {test_acc},'
                    ' Test loss: {test_loss}\n'.format(i, num_exps, **res)
                )
                i += 1
        output_file.close()
        print('Finished!\n')

    def plot_res(self, title, path):
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="darkgrid")
        res = pd.read_csv(self.log_path, usecols=[0, 1, 2, 3])

        plt.subplot(121)
        plt.title('Accuracy', fontsize=15)
        lp = sns.lineplot(x='activation', y='test_acc', data=res, hue='regularization')
        lp.set_xlabel('Activation Function', fontsize=15)
        lp.set_ylabel('Test Accuracy', fontsize=15)

        plt.subplot(122)
        plt.title('Loss', fontsize=15)
        lp = sns.lineplot(x='activation', y='test_loss', data=res, hue='regularization')
        lp.set_xlabel('Activation Function', fontsize=15)
        lp.set_ylabel('Test Loss', fontsize=15)

        plt.suptitle(title, fontsize=20)
        plt.savefig(path)
        plt.show()
