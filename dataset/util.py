from torch.utils.data import DataLoader, random_split
from dataset.mnist_m import MNISTM
import torch
import torchvision
import torchvision.transforms as transforms


def get_train_dataset(name, img_size, root):
    train_set, val_set = get_dataset(name, True, img_size, root)
    return train_set, val_set


def get_test_dataset(name, img_size, root):
    dataset, _ = get_dataset(name, False, img_size, root)
    return dataset


def get_dataset(name, train, img_size, root):
    if name == "mnist":
        dataset = get_mnist(train, img_size, root)
    if name == "svhn":
        dataset = get_svhn(train, img_size, root)
    if name == "mnist_m":
        dataset = get_mnist_m(train, img_size, root)
    return dataset


def get_mnist(train, img_size, root):
    if img_size > 28:
        transform_mnist_train = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ChanDup(),
        ])
    else:
        transform_mnist_train = transforms.Compose([
            transforms.CenterCrop(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            #             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ChanDup(),
        ])

    dataset = torchvision.datasets.MNIST(
        root=root, train=train, download=True, transform=transform_mnist_train)
    if train:
        train_set, val_set = random_split(
            dataset, [50000, 10000])
        return train_set, val_set
    else:
        return dataset, None


def get_svhn(train, img_size, root):
    if img_size > 32:
        transform_svhn_train = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        transform_svhn_train = transforms.Compose([
            transforms.CenterCrop(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    if train:
        split = "train"
        dataset = torchvision.datasets.SVHN(
            root=root, split=split, download=True, transform=transform_svhn_train)
        train_set, val_set = random_split(
            dataset, [63257, 10000])
        return train_set, val_set
    else:
        split = "test"
        dataset = torchvision.datasets.SVHN(
            root=root, split=split, download=True, transform=transform_svhn_train)
        return dataset, None


def get_mnist_m(train, img_size, root):
    if img_size > 28:
        transform_mnist_train = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        transform_mnist_train = transforms.Compose([
            transforms.CenterCrop(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    dataset = MNISTM(root=root, train=train, download=True,
                   transform=transform_mnist_train)
    if train:
        train_set, val_set = random_split(
            dataset, [50000, 10000])
        return train_set, val_set
    else:
        return dataset, None


class ChanDup:
  def __call__(self, img):
    return img.repeat(3, 1, 1)