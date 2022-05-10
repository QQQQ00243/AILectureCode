import torch
import pickle
import matplotlib.pyplot as plt

from torch import nn, optim


def train(
        model,
        device,
        criterion,
        optimizer,
        train_loader,
):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
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
            val_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            val_acc += pred.eq(target.view_as(pred)).sum().item() / len(data)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


class Trainer:
    def __init__(
            self,
            model,
            epochs,
            log_path,
            model_path,
            dataloader,
            init_lr,
            optimizer_name,
            criterion_name,
            no_cuda=False,
            earlystop=None,
            lr_scheduler=None,
    ):
        self.model = model
        self.epochs = epochs
        self.log_path = log_path
        self.model_path = model_path
        self.dataloader = dataloader
        self.init_lr = init_lr
        self.lr_scheduler = lr_scheduler
        self.optimizer_name = optimizer_name
        self.criterion_name = criterion_name
        self.no_cuda = no_cuda
        self.earlystop = earlystop

    def _get_device(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('[INFO]Using', device)
        return device

    def _get_optimizer(self, model):
        print(f'Using {self.optimizer_name} as optimizer')
        return getattr(optim, self.optimizer_name)(model.parameters(), lr=self.init_lr)

    def _get_criterion(self):
        print(f'Using {self.criterion_name} as criterion')
        return getattr(nn, self.criterion_name)()

    def fit(self):
        device = self._get_device()
        model = self.model.to(device)
        train_loader = self.dataloader.train_loader
        val_loader = self.dataloader.val_loader
        criterion = self._get_criterion()
        optimizer = self._get_optimizer(model)
        train_loss, train_acc, val_loss, val_acc = [[] for _ in range(4)]
        for epoch in range(1, self.epochs+1):
            # train
            train_loss_, train_acc_ = train(
                model=model,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
            )
            train_loss.append(train_loss_)
            train_acc.append(train_acc_)

            # validation
            val_loss_, val_acc_ = validate(
                model=model,
                device=device,
                criterion=criterion,
                val_loader=val_loader,
            )
            val_loss.append(val_loss_)
            val_acc.append(val_acc_)

            # print info
            info_prefix = f'[INFO]Train Epoch: {epoch} / {self.epochs}'
            info_train = f'\tTrain Loss: {train_loss_:.6f}\tTrain Accuracy: {train_acc_:.6f}'
            info_val = f'\tValidation Loss: {val_loss:.6f}\tValidation Accuracy: {val_acc:.6f}\n'
            print(info_prefix + info_train + info_val)

            # early stop
            if self.earlystop is not None:
                self.earlystop(val_loss=val_loss_, model=model)
                if self.earlystop.stop:
                    break

            # learning rate scheduler
            if self.lr_scheduler is not None:
                scheduler.step()
        if args.EarlyStop:
            model.load_state_dict(torch.load(self.model_path))
        self.model = model
        print('Saving model to ', self.model_path)
        torch.save(model.state_dict(), self.model_path)
        self.history = {
            'train_history': {'train_accuracy': train_acc, 'train_loss': train_loss},
            'val_history': {'val_accuracy': val_acc, 'val_loss': val_loss},
        }

    def eval(self):
        device = self._get_device()
        test_loader = self.test_loader
        test_loss, test_acc = test(self.args, self.model, device, test_loader)
        test_history = {'test_loss': test_loss, 'test_acc': test_acc}
        self.history['test_history'] = test_history
        with open(self.log_path, 'wb') as f:
            print('Saving history to ', self.log_path)
            pickle.dump(self.history, f)
        return test_loss, test_acc

    def plot_history(self, title, img_path):
        with open(self.log_path, 'rb') as f:
            history = pickle.load(f)
        train_history, val_history, test_history = history.values()
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.suptitle(title, fontsize=25)

        epochs = [i + 1 for i in range(len(train_history['train_loss']))]

        plt.subplot(121)
        plt.title('Loss', fontsize=20)
        plt.plot(epochs, train_history['train_loss'], color='blue')
        plt.plot(epochs, val_history['val_loss'], color='red')
        plt.legend(['Train Loss', 'Validation Loss'], loc='upper right', fontsize=15)
        plt.xticks(epochs[::2])
        plt.xlabel('epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)

        plt.subplot(122)
        plt.title('Accuracy', fontsize=20)
        plt.plot(epochs, train_history['train_accuracy'], color='blue')
        plt.plot(epochs, val_history['val_accuracy'], color='red')
        plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='best', fontsize=15)
        plt.xlabel('epochs', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.xticks(epochs[::2])
        plt.subplots_adjust(bottom=0.4)
        plt.figtext(0.5, -0.05, 'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}'.format(**test_history),
                    ha="center",
                    fontsize=20, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

        plt.savefig(img_path, bbox_inches='tight')
        plt.show()
