import copy
import numpy as np
import os
import random
import math
from sklearn.metrics import classification_report, confusion_matrix
import csv

import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init(module):
    if isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight,a=math.sqrt(5))
        torch.nn.init.constant_(module.bias, 0)
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        weight_shape = module.weight.shape
        out_channels, in_channels, kernel_size = weight_shape[0], weight_shape[1], weight_shape[2:]

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)


# set all seeds for reproducability
def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate_cifar10(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/2 and 3/4 epochs"""
    if epoch in [40, 60]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']>0: print("changed learning rate to {}".format(lr))


def adjust_learning_rate_mnist(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if epoch in [int(args['epochs']/3), int(2 * args['epochs'] / 3)]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print("changed learning rate to {}".format(lr))


def adjust_learning_rate_physionet(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if epoch in [int(args['epochs']/3), int(2 * args['epochs'] / 3)]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print("changed learning rate to {}".format(lr))


def adjust_learning_rate_shhs(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1/3 and 2/3 epochs"""
    if epoch in [int(args['epochs']/3), int(2 * args['epochs'] / 3)]:
        lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print("changed learning rate to {}".format(lr))


def get_layers(network, all_layers=[]):
    '''
    gets all layers of a network
    '''
    for layer in network.children():
        if type(layer) == nn.Sequential:
            get_layers(layer, all_layers)
        if list(layer.children()) == []:
            all_layers.append(layer)
    return all_layers


def make_idx_dict(model, ctr, ary, d):
    for m_idx, m_k in enumerate(model._modules.keys()):
        n_ary = copy.deepcopy(ary)
        if len(model._modules[m_k]._modules.keys()):
            n_ary.append(m_k)
            ctr, d = make_idx_dict(model._modules[m_k], ctr, n_ary, d)
        else:
            n_ary.append(m_k)
            ctr = ctr+1
            d[ctr] = n_ary
    return ctr, d


def get_layer_from_idx(model, idx_ds, idx):
    if len(idx_ds[idx]) == 1:
        return model._modules[idx_ds[idx][0]]
    m_idx = idx_ds[idx].pop(0)
    return get_layer_from_idx(model._modules[m_idx],idx_ds,idx)


def set_layer_to_idx(model, idx_ds, idx, layer):
    if len(idx_ds[idx]) == 1:
        model._modules[idx_ds[idx][0]] = layer
    else:
        m_idx = idx_ds[idx].pop(0)
        set_layer_to_idx(model._modules[m_idx], idx_ds, idx, layer)


def _lr_rate_schedule(args, optimizer, epoch):

    if (epoch * 3 == args.epochs) or (epoch * 3 == 2 * args.epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr = param_group['lr']

        if args['verbose']: print('reduce lr to {}'.format(lr))


def print_results(args, train_acc=None, val_acc=None, test_acc=None, adv_train_acc=None, adv_val_acc=None, adv_test_acc=None,
                  avg_val_acc=None, epoch='N/A'):

    if train_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TRAIN ACC (benign): {}'.format(epoch, train_acc))

    if val_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VAL ACC (benign): {}'.format(epoch, val_acc))

    if test_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TEST ACC (benign): {}'.format(epoch, test_acc))

    if adv_train_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TRAIN ACC (adversarial): {}'.format(epoch, adv_train_acc))

    if adv_val_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VAL ACC (adversarial): {}'.format(epoch, adv_val_acc))

    if adv_test_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - TEST ACC (adversarial): {}'.format(epoch, adv_test_acc))

    if avg_val_acc is not None:
        if args['verbose'] > 0: print('\t\tEpoch {} - VAL ACC (average): {}'.format(epoch, avg_val_acc))


def load_model(model_path):
    model = torch.load(model_path)
    model = model.eval()
    return model


def save_model(model, model_path, args):
    model_path = model_path[:-3]  # remove '.pt'
    torch.save(model, model_path + '.pt', pickle_protocol=4)


def convert_labels_to_categorical(labels):
    new_labels = []

    for label in labels:
        label_index = np.argmax(label)
        new_labels.append(label_index)

    return new_labels