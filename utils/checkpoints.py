import torch


class EarlyStop():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience, path, min_delta):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif (self.best_loss - val_loss) < self.min_delta:
            print(f'INFO: Early stopping counter {self.counter+1} of {self.patience}\n')
            if self.counter == 0:
                print('Saving Model to ' + self.path + '\n')
                torch.save(model.state_dict(), self.path)
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.stop = True


