import itertools
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=None)


def plot_samples(
        path,
        title,
        dataloader,
        num_per_row,
        num_instances,
        title_size=25,
        subtitle_size=15,
        idx_to_class=None,
):
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [10, 10]
    nrows = -(-num_instances // num_per_row)
    ncols = num_per_row
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(hspace=0.5)

    dataiter = iter(dataloader)
    imgs, labels = dataiter.next()
    batch_size = len(labels)
    for i in range(num_instances):
        j = i % batch_size
        if j == 0 and i != 0:
            dataiter.next()
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        if idx_to_class is None:
            plt.title('{}'.format(labels[j]), fontsize=subtitle_size)
        else:
            plt.title('{}'.format(idx_to_class[labels[j].item()]), fontsize=subtitle_size)
        imshow(imgs[j])
    plt.suptitle(title, fontsize=title_size)
    plt.savefig(path)
    plt.show()


def plot_confusion_matrix(
        cm,
        title,
        classes,
        path,
        normalize=True,
        cmap=plt.cm.Blues,
        label_size=12, title_size=15
):
    plt.rcParams["figure.autolayout"] = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black"
                 )

    plt.tight_layout()
    plt.title(title, fontsize=title_size)
    plt.ylabel('True label', fontsize=label_size)
    plt.xlabel('Predicted label', fontsize=label_size)
    plt.savefig(path)
    plt.show()
