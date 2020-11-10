from __future__ import print_function

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn

# from models.vgg import vgg16, vgg19
# from models.resnet import resnet50
from torchvision.models import resnet50, vgg16, densenet161, resnet101, densenet121


# https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
class GenHelper(data.Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, train_split=0.9, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return:
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    train_size = int(dslen * train_split)
    valid_size = dslen - train_size
    np.random.shuffle(indices)
    train_mapping = indices[0:train_size]
    valid_mapping = indices[train_size:]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid


class Model_Builder:
    def __init__(self, model_type, dataset, model_path, args):
        self.args = args
        self.model_type = model_type
        self.dataset = dataset
        self.model_path = model_path
        self.data_dir = args['data_dir']

        self.min_value = np.Inf
        self.max_value = -np.Inf

        if 'cifar10' in dataset:
            output = 10
        elif 'unmask' in dataset:
            if 'cs3' in args['class_set']:
                output = 3
            elif 'cs5' in args['class_set']:
                output = 5

        if model_type == 'vgg16':
            self.model = vgg16(pretrained=True)
            self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output))

        elif model_type == 'densenet121':
            self.model = densenet121(pretrained=True)
            self.model.classifier = nn.Linear(1024, output)

        elif model_type == 'resnet50':
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, output)

        elif model_type == 'resnet101':
            self.model = resnet101(pretrained=True)
            self.model.fc = nn.Linear(2048, output)

        self.model.cuda()
        device = args['device']
        self.model = self.model.to(device)

        if dataset == 'cifar10':
            self.train_loader, self.val_loader, self.test_loader = self.get_cifar10()

        elif dataset == 'cifar100':
            self.train_loader, self.test_loader = self.get_cifar100()

        elif dataset == 'unmask':
            self.train_loader, self.val_loader, self.test_loader = self.get_unmask()

    def get_cifar10(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10/train', train=True, download=True, transform=transform)
        trainset, validset = train_valid_split(trainset)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args['batch_size'], shuffle=True, num_workers=2)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=self.args['batch_size'], shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10/test', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.args['batch_size'], shuffle=False, num_workers=2)

        return train_loader, valid_loader, test_loader

    def get_cifar100(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])

        cifar100_training = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=True, download=True, transform=transform_train)
        cifar100_training_loader = data.DataLoader(cifar100_training, shuffle=True, num_workers=2, batch_size=self.args['batch_size'])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])

        cifar100_test = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=False, download=True, transform=transform_test)
        cifar100_test_loader = data.DataLoader(cifar100_test, shuffle=True, num_workers=2, batch_size=self.args['batch_size'])

        return cifar100_training_loader, cifar100_test_loader

    def get_unmask(self):
        transform = transforms.Compose(
            [
             transforms.Resize(150),
             transforms.CenterCrop(150),
             transforms.ToTensor(),
            ])

        # torch.min(next(iter(data.DataLoader(trainset, batch_size=len(train_loader.dataset.targets), shuffle=True, num_workers=2)))[0])
        trainset = torchvision.datasets.ImageFolder(root='./data/unmask/{}/train/'.format(self.args['class_set']), transform=transform)
        train_loader = data.DataLoader(trainset, batch_size=self.args['batch_size'], shuffle=True, num_workers=0)

        valset = torchvision.datasets.ImageFolder(root='./data/unmask/{}/val/'.format(self.args['class_set']), transform=transform)
        val_loader = data.DataLoader(valset, batch_size=self.args['batch_size'], shuffle=True, num_workers=0)

        testset = torchvision.datasets.ImageFolder(root='./data/unmask/{}/test'.format(self.args['class_set']), transform=transform)
        test_loader = data.DataLoader(testset, batch_size=self.args['batch_size'], shuffle=True, num_workers=0)

        return train_loader, val_loader, test_loader

    def get_bounds(self):
        return self.min_value, self.max_value

    def get_model(self):
        return self.model

    def refresh_model_builder(self):
        self.__init__(self.model_type, self.dataset, self.args)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

