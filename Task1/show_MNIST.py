import os
import argparse
import matplotlib.pyplot as plt
import torchvision.datasets as datasets


def make_args():
    parser = argparse.ArgumentParser(description="Show MNIST")
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


def show_MNIST(
    root,
    title,
    img_file,
    num_per_row,
    num_instances,
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
    dataset = datasets.MNIST(root=root, train=True)
    for i in range(num_instances):
        img, label = dataset[i]
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        plt.title('{}'.format(label), fontsize=subtitle_size)
        plt.imshow(img, cmap="gray")
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title, fontsize=title_size)
    plt.savefig(img_file)
    plt.show()


def main():
    args = make_args().parse_args()  
    img_file=os.path.join(args.imgs_dir, "MNIST_Examples.svg"),
    print(f"Saving MNIST examples to {img_file}.\n")
    show_MNIST(
        root=args.dataset_root,
        title="MNIST Examples",
        img_file=os.path.join(args.imgs_dir, "MNIST_Examples.svg"),
        num_per_row=8,
        num_instances=64,
    )


if __name__ == "__main__":
    main()
