import copy
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataset, Subset
import numpy as np


class Dataset(dataset.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def create_imbalanced_test_set(self, source_label, target_label,
                                   source_count=5000, target_count=200, other_count=600):
        """
        创建不平衡的测试集用于评估标签翻转攻击

        Args:
            source_label: 源标签（攻击源）
            target_label: 目标标签（攻击目标）
            source_count: 源类样本数量
            target_count: 目标类样本数量
            other_count: 其他类每类样本数量
        """
        new_data = []
        new_targets = []
        targets_array = np.array(self.targets)

        # 获取每个类别的原始样本数
        n_classes = targets_array.max() + 1
        original_counts = [np.sum(targets_array == i) for i in range(n_classes)]

        for class_idx in range(n_classes):
            class_indices = np.where(targets_array == class_idx)[0]
            original_class_count = len(class_indices)

            if class_idx == source_label:
                target_count_for_class = source_count
            elif class_idx == target_label:
                target_count_for_class = target_count
            else:
                target_count_for_class = other_count

            # 如果需要的样本数超过原有样本数，进行重复采样
            if target_count_for_class > original_class_count:
                # 计算需要重复多少轮
                repeat_times = (target_count_for_class // original_class_count) + 1
                extended_indices = np.tile(class_indices, repeat_times)
                selected_indices = extended_indices[:target_count_for_class]
            else:
                # 随机选择指定数量的样本
                selected_indices = np.random.choice(class_indices, target_count_for_class, replace=False)

            # 添加选中的样本
            for idx in selected_indices:
                new_data.append(self.data[idx])
                new_targets.append(self.targets[idx])

        print(f"Created imbalanced test set:")
        print(f"  Source class {source_label}: {source_count} samples")
        print(f"  Target class {target_label}: {target_count} samples")
        print(f"  Other classes: {other_count} samples each")

        return Dataset(new_data, new_targets)

    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        np.random.seed(114514)
        targets = np.array(self.targets)
        n_classes = targets.max() + 1
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        class_idcs = [np.argwhere(targets == y).flatten() for y in range(n_classes)]

        client_idcs = [[] for _ in range(n_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs

    def uniform_split_iid(self, n_clients):
        np.random.seed(19260817)
        targets = np.array(self.targets)
        n_classes = targets.max() + 1
        label_distribution = np.array([[1.0 / n_clients] * n_clients] * n_classes)
        class_idcs = [np.argwhere(targets == y).flatten() for y in range(n_classes)]

        client_idcs = [[] for _ in range(n_clients)]
        for k_idcs, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs


class Dataset_CIFAR10(Dataset):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, data=[], targets=[]):
        super(Dataset_CIFAR10, self).__init__(data, targets)

    def load(self, path='./data/CIFAR10', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]) if train == True else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        dataset = torchvision.datasets.CIFAR10(root=path, train=train, download=True, transform=transform)
        data = []
        targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_CIFAR10, self).__init__(data, targets)

    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)

    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_CIFAR10, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []
            targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_CIFAR10(data, targets))
        return datasets

    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_CIFAR10, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []
            targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_CIFAR10(data, targets))
        return datasets

    def _get_statistics(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        return mean, std


class Dataset_FMNIST(Dataset):
    classes = (
        "0 - t-shirt",
        "1 - trouser",
        "2 - pullover",
        "3 - dress",
        "4 - coat",
        "5 - sandal",
        "6 - shirt",
        "7 - sneaker",
        "8 - bag",
        "9 - ankle boot",
    )

    def __init__(self, data=[], targets=[]):
        super(Dataset_FMNIST, self).__init__(data, targets)

    def load(self, path='./data/FMNIST', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]) if train == True else transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        dataset = torchvision.datasets.FashionMNIST(root=path, train=train, download=True, transform=transform)
        data = []
        targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_FMNIST, self).__init__(data, targets)

    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)

    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_FMNIST, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []
            targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_FMNIST(data, targets))
        return datasets

    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_FMNIST, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []
            targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_FMNIST(data, targets))
        return datasets

    def _get_statistics(self):
        mean = (0.2860, 0.2860, 0.2860)
        std = (0.3530, 0.3530, 0.3530)
        return mean, std


class Dataset_MNIST(Dataset):
    classes = (
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    )

    def __init__(self, data=[], targets=[]):
        super(Dataset_MNIST, self).__init__(data, targets)

    def load(self, path='./data/MNIST', train=True):
        mean, std = self._get_statistics()

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]) if train == True else transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        dataset = torchvision.datasets.MNIST(root=path, train=train, download=True, transform=transform)
        data = []
        targets = []
        for i in range(len(dataset)):
            img, target = dataset[i]
            data.append(img)
            targets.append(target)

        super(Dataset_MNIST, self).__init__(data, targets)

    def split(self, n, iid=False):
        if iid == False:
            return self.dirichlet_split_noniid(n)
        return self.uniform_split_iid(n)

    def dirichlet_split_noniid(self, n_clients, alpha=1.0):
        client_idcs = super(Dataset_MNIST, self).dirichlet_split_noniid(n_clients, alpha)
        datasets = []
        for i in range(n_clients):
            data = []
            targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_MNIST(data, targets))
        return datasets

    def uniform_split_iid(self, n_clients):
        client_idcs = super(Dataset_MNIST, self).uniform_split_iid(n_clients)
        datasets = []
        for i in range(n_clients):
            data = []
            targets = []
            for idx in client_idcs[i]:
                data.append(self.data[idx])
                targets.append(self.targets[idx])
            datasets.append(Dataset_MNIST(data, targets))
        return datasets

    def _get_statistics(self):
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        return mean, std


if __name__ == '__main__':
    from collections import Counter

    # dataset = Dataset_CIFAR10()
    # dataset = Dataset_FMNIST()
    dataset = Dataset_MNIST()
    dataset.load()

    n_clients = 5

    client_datasets = dataset.split(n=n_clients, iid=False)  # 改为 True 以测试 IID 分割

    for i, client_data in enumerate(client_datasets):
        label_counts = Counter(client_data.targets)
        print(f"Client {i} label distribution:")
        for label in sorted(label_counts):
            print(f"  Label {label}: {label_counts[label]}")
        print("-" * 40)