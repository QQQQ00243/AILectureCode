import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_tools import imshow, plot_confusion_matrix


class Analyzer:
    def __init__(self, args, model, model_path, dataloader):
        self.args = args
        model.load_state_dict(torch.load(model_path))
        self.model = model
        self.dataloader = dataloader

    def _get_device(self):
        args = self.args
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def _get_mistakes(self):
        model = self.model
        model.eval()
        device = self._get_device()
        dataloader = self.dataloader
        mistakes = []
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                preds = output.argmax(dim=1)
                for i, (pred, label) in enumerate(zip(preds, labels)):
                    if pred != label.view_as(pred):
                        mistake = {
                            'img': imgs[i],
                            'label': label.item(),
                            'pred': pred.item()
                        }
                        mistakes.append(mistake)
        return mistakes

    def plot_mistakes(
            self, num_per_row, title, path,
            num_instances=None,
            subtitle_size=20, title_size=25
    ):
        mistakes = self._get_mistakes()
        if num_instances is None or num_instances > len(mistakes):
            num_instances = len(mistakes)
        nrows = -(-num_instances // num_per_row)
        ncols = num_per_row
        fig, _ = plt.subplots(nrows=nrows, ncols=ncols)
        plt.rcParams["figure.autolayout"] = True
        fig.set_size_inches(10, 10)
        plt.subplots_adjust(hspace=0.5)

        for i in range(num_instances):
            mistake = mistakes[i]
            plt.subplot(nrows, ncols, i + 1)
            plt.axis('off')
            plt.text(8, -0.5, '{}-'.format(mistake['label']),
                     fontsize=subtitle_size)
            plt.text(15, -0.5, '{}'.format(mistake['pred']),
                     fontsize=subtitle_size, color='r')
            imshow(mistake['img'])
        plt.suptitle(title, fontsize=title_size)
        plt.savefig(path)
        plt.show()

    def _get_confusion_matrix(self):
        num_classes = len(self.args.classes)
        model = self.model
        model.eval()
        device = self._get_device()
        cm = np.zeros((num_classes, num_classes), dtype=int)
        dataloader = self.dataloader
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                for label, pred in zip(labels.view(-1), preds.view(-1)):
                    cm[label.item(), pred.item()] += 1
        return cm

    def plot_confusion_matrix(self, title, path):
        cm = self._get_confusion_matrix()
        plot_confusion_matrix(
            cm=cm, title=title, path=path,
            classes=self.args.idx_to_class.values(),
        )
