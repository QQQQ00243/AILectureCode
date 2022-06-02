import torch


class EarlyStop():
    def __init__(
        self,
        mode,
        monitor,
        patience,
        ckpt_file,
    ):
        self.mode = mode
        self.monitor = monitor
        self.patience = patience
        self.counter = 0
        self.best = None
        self.stop = False
        self.ckpt_file = ckpt_file

    def __call__(self, val, model):
        if self.best == None:
            self.best = val
        else:
            if (self.mode == "max" and self.best < val) \
                or (self.mode == "min" and self.best > val):
                   self.best = val
                   self.counter = 0
            else:
                print(f'[INFO]Early stopping counter {self.counter+1} of {self.patience}')
                if self.counter == 0:
                    print(f'[INFO]Saving Model to {self.ckpt_file}\n')
                    torch.save(model.state_dict(), self.ckpt_file)
                self.counter += 1
                if self.counter >= self.patience:
                    print('[INFO]Early stopping\n')
                    self.stop = True
