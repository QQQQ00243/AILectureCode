import os
import argparse
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

from Task1.utils import get_idx_to_class


def make_args():
    parser = argparse.ArgumentParser(description="Show FashionMNIST")
    parser.add_argument("--dataset-root", default="./data", 
                        type=str, help="root of dataset")
    parser.add_argument("--download", default=False,
                        help="enable download")
    parser.add_argument("--imgs-dir", default="./imgs",
                        type=str, help="directory to images")
    return parser


def make_dir(args):
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)
    if not os.path.exists(args.dataset_root):
        os.mkdir(args.dataset_root)


def show_FashionMNIST(
    root,
    title,
    img_file,
    num_per_row,
    num_instances,
    idx_to_class,
    title_size=25,
    subtitle_size=15,
):
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [10, 10]
    nrows = -(-num_instances // num_per_row)
    ncols = num_per_row
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    fig.set_size_inches(10, 10)
    dataset = datasets.FashionMNIST(root=root, train=True)
    for i in range(num_instances):
        img, label = dataset[i]
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        class_ = idx_to_class[label]
        plt.title('{}'.format(class_), fontsize=subtitle_size)
        plt.imshow(img, cmap="gray")
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title, fontsize=title_size)
    plt.savefig(img_file)
    plt.show()



def main():
    args = make_args().parse_args()
    dataset = datasets.FashionMNIST(root=args.dataset_root, train=True)
    idx_to_class = get_idx_to_class(dataset.class_to_idx)

    img_file=os.path.join(args.imgs_dir, "FashionMNIST_Examples.svg"),
    print(f"Saving FashionMNIST examples to {img_file}.\n")
    show_FashionMNIST(
        root=args.dataset_root,
        title="FashionMNIST Examples",
        img_file=os.path.join(args.imgs_dir, "FashionMNIST_Examples.svg"),
        num_per_row=8,
        num_instances=64,
        idx_to_class=idx_to_class,
    )


if __name__ == "__main__":
    main()
