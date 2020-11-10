import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.metrics import (roc_auc_score, roc_curve)

from advertorch.attacks import LinfPGDAttack, LinfMomentumIterativeAttack, L2MomentumIterativeAttack, L2BasicIterativeAttack, L2PGDAttack, GradientSignAttack
from advertorch.context import ctx_noparamgrad_and_eval

from config import object_dict
from extract_features import Extract
from utils import save_model, print_results
from utils import adjust_learning_rate_cifar10, adjust_learning_rate_mnist, adjust_learning_rate_physionet, adjust_learning_rate_shhs


class Train:
    def __init__(self, args, spectral_args):
        self.args = args
        self.spectral_args = spectral_args

        self.train_loader = spectral_args['train_loader']
        self.val_loader = spectral_args['val_loader']
        self.test_loader = spectral_args['test_loader']

        self.robust_training = spectral_args['robust_training']

    def init_pgd(self, model, test=False):
        epsilon = self.spectral_args['test_epsilon'] if test else self.spectral_args['train_epsilon']
        step_size = self.spectral_args['test_eps_iter'] if test else self.spectral_args['eps_iter']
        nb_iter = self.spectral_args['test_nb_iter'] if test else self.spectral_args['nb_iter']

        return LinfPGDAttack(
                            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                            eps=epsilon / 255.0, nb_iter=nb_iter,
                            eps_iter=step_size, rand_init=True, clip_min=self.spectral_args['clip_min'],
                            clip_max=self.spectral_args['clip_max'], targeted=False)

    def init_pgdl2(self, model):
        return L2PGDAttack(
                            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                            eps=self.args['test_epsilon'] / 255.0, nb_iter=self.spectral_args["test_nb_iter"],
                            eps_iter=self.spectral_args["test_eps_iter"], rand_init=True, clip_min=self.spectral_args['clip_min'],
                            clip_max=self.spectral_args['clip_max'], targeted=False)

    def init_mialinf(self, model):
        return LinfMomentumIterativeAttack(
                                            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=self.args['test_epsilon'] / 255.0, nb_iter=self.spectral_args["test_nb_iter"],
                                            eps_iter=self.spectral_args["test_eps_iter"],  clip_min=self.spectral_args['clip_min'],
                                            clip_max=self.spectral_args['clip_max'], targeted=False, decay_factor=1.0)

    def init_mial2(self, model):
        return L2MomentumIterativeAttack(
                                            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                            eps=self.args['test_epsilon'] / 255.0, nb_iter=self.spectral_args["test_nb_iter"],
                                            eps_iter=self.spectral_args["test_eps_iter"],  clip_min=self.spectral_args['clip_min'],
                                            clip_max=self.spectral_args['clip_max'], targeted=False, decay_factor=1.0)

    # def init_bia(self, model):
    #     return LinfBasicIterativeAttack(
    #                             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    #                             eps=self.args['test_epsilon'] / 255.0, nb_iter=self.spectral_args["test_nb_iter"],
    #                             eps_iter=self.spectral_args["test_eps_iter"], clip_min=self.spectral_args['clip_min'],
    #                             clip_max=self.spectral_args['clip_max'],
    #     )

    # def init_fgsm(self, model):
    #     return GradientSignAttack(
    #                             model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    #                             eps=self.args['test_epsilon'] / 255.0, clip_min=self.spectral_args['clip_min'],
    #                             clip_max=self.spectral_args['clip_max']
    #     )
    #
    # def init_cwl2(self, model):
    #     num_classes = 3 if 'cs3' in self.args['class_set'] else 5
    #
    #     return CarliniWagnerL2Attack(
    #                             model, num_classes=num_classes, confidence=0,
    #                             binary_search_steps=9, learning_rate=0.01, initial_const=0.001, max_iterations=1000,
    #                             clip_min=self.spectral_args['clip_min'], clip_max=self.spectral_args['clip_max'],
    #     )

    def filter_data(self, data_benign, data_adv, output_benign, output_adv, target):
        indices_to_keep = []
        for index, label in enumerate(output_adv):
            if label != target[index]:
                indices_to_keep.append(index)

        return [data_benign[i] for i in indices_to_keep], [data_adv[i] for i in indices_to_keep], [output_benign[i] for i in indices_to_keep], [output_adv[i] for i in indices_to_keep]

    def run_attack(self, model, data, target, attack_type='pgd_linf'):
        correct = 0

        if attack_type == 'pgd_linf':
            adversary = self.adversary_pgd_linf
        elif attack_type == 'pgd_l2':
            adversary = self.adversary_pgd_l2
        elif attack_type == 'mia_linf':
            adversary = self.adversary_mia_linf
        elif attack_type == 'mia_l2':
            adversary = self.adversary_mia_l2

        data_perturbed = adversary.perturb(data, target)
        output = model(data_perturbed)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        # data_perturbed, output, target = self.filter_data(data_perturbed, output, target)

        return data_perturbed, np.argmax(output.detach().cpu().numpy(), 1).tolist(), correct

    def test_unmask_defense(self, model, data_loader, attack_type='pgd_linf'):
        self.adversary_pgd_linf = self.init_pgd(model, test=True)
        self.adversary_pgd_l2 = self.init_pgdl2(model)
        self.adversary_mia_linf = self.init_mialinf(model)
        self.adversary_mia_l2 = self.init_mial2(model)

        num_images = len(data_loader.dataset.targets)

        experiment_dir = self.args['log_dir'] + 'unmask_results_rt_{}_attack_{}_strength_{}'.format(self.args['robust_training'], attack_type,  self.args['test_epsilon'])
        os.makedirs(experiment_dir, exist_ok=True)

        model_path = os.getcwd() + '/Mask_RCNN/logs/model_k/mask_rcnn_parts_00{}.h5'.format(40)  # or None
        objects_to_consider = object_dict[self.args['class_set']]

        label_map = data_loader.dataset.class_to_idx
        label_map_reverse = {v: k for k, v in label_map.items()}

        correct_benign, correct_benign_unmask, correct_adv, correct_adv_unmask = 0, 0, 0, 0
        features_benign_total, features_adv_total, output_benign_total,  output_adv_total = [], [], [], []

        model.eval()

        t = tqdm(iter(data_loader), leave=False, total=len(data_loader), disable=not self.args['verbose'] > 1)
        for batch_idx, (data, target) in (enumerate(t)):

            data, target = data.to(self.args['device']), target.to(self.args['device'])
            target_numpy = target.cpu().numpy().flatten()

            output_benign = model(data)
            pred = output_benign.max(1, keepdim=True)[1]
            correct_benign += pred.eq(target.view_as(pred)).sum().item()

            data_adv, output_adv, correct_adv_batch = self.run_attack(model, data, target, attack_type=attack_type)
            correct_adv += correct_adv_batch

            # Measure UnMask defense
            model_k = Extract(model_path=model_path, objects_to_consider=objects_to_consider)

            features_benign, predictions_benign = model_k.extract(data.cpu(), label_map)
            correct_benign_unmask += np.count_nonzero(target_numpy == predictions_benign)

            features_adv, predictions_adv = model_k.extract(data_adv.cpu(), label_map)
            correct_adv_unmask += np.count_nonzero(target_numpy == predictions_adv)

            model_k.reset_keras()  # reset to prevent memory error

            features_benign_filt, features_adv_filt, output_benign_filt, output_adv_filt = self.filter_data(
                features_benign, features_adv, np.argmax(output_benign.detach().cpu().numpy(), 1).tolist(), output_adv, target)

            features_benign_total = features_benign_total + features_benign_filt
            features_adv_total = features_adv_total + features_adv_filt
            output_benign_total += output_benign_filt
            output_adv_total += output_adv_filt

        ben_acc = round(100. * correct_benign / num_images, 2)
        adv_acc = round(100. * correct_adv / num_images, 2)
        ben_acc_unmask = round(100. * correct_benign_unmask / num_images, 2)
        adv_acc_unmask = round(100. * correct_adv_unmask / num_images, 2)

        tpr, fpr, thresholds = self.test_unmask_detection(model_path, objects_to_consider, attack_type, experiment_dir, label_map_reverse,
                                                          features_benign_total, features_adv_total, output_benign_total, output_adv_total)

        detection_results_path = experiment_dir + "/results.txt"
        with open(detection_results_path, 'w') as f:
            f.write("Number of defense images: {}\n".format(num_images))
            f.write("Number of filtered benign images: {}\n".format(len(features_benign_total)))
            f.write("Number of filtered {} images: {}\n".format(attack_type, len(features_adv_total)))
            f.write("Benign Accuracy UnMask: {}\n".format(ben_acc_unmask))
            f.write("{} Accuracy UnMask: {}\n".format(attack_type, adv_acc_unmask))
            f.write("Benign Accuracy: {}\n".format(ben_acc))
            f.write("{} Accuracy: {}\n".format(attack_type, adv_acc))
            f.write("True positive rate values: {}\n".format(tpr))
            f.write("False positive rate values: {}\n".format(fpr))
            f.write("Thresholds: {}".format(thresholds))

        return ben_acc, adv_acc, ben_acc_unmask, adv_acc_unmask

    def test_unmask_detection(self, model_path, objects_to_consider, attack_type, experiment_dir, label_map_reverse, features_benign_total,
                              features_adv_total, output_benign_total, output_adv_total):

        model_k = Extract(model_path=model_path, objects_to_consider=objects_to_consider)
        similarity_scores = model_k.decision_function(features_adv_total + features_benign_total, output_adv_total + output_benign_total, label_map_reverse)

        binary_labels = [1] * len(features_adv_total) + [0] * len(features_benign_total)
        auc_score = roc_auc_score(y_true=binary_labels, y_score=similarity_scores)

        fig = plt.figure()
        fig.suptitle("Attack type: {}".format(attack_type, fontweight="bold"))
        ax = fig.add_subplot(111)
        fpr, tpr, thresholds = roc_curve(binary_labels, similarity_scores)
        plt.plot(fpr, tpr, color='blue', label='AUC={}'.format(auc_score))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        plt.title('ROC Curve')
        legend = plt.legend(loc=4)

        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        figure_path = os.path.join(experiment_dir, "roc_curve.pdf")
        plt.savefig(figure_path, bbox_inches='tight', format='pdf', dpi=300)

        return tpr, fpr, thresholds

    def test_robustness(self, model, data_loader, attack_type='pgd_linf', attack_strength='test'):
        model.eval()
        # set_random_seed(self.args['seed'])
        data_len = len(data_loader.dataset)
        correct, correct_adv = 0, 0

        if attack_strength == 'train':
            adversary_pgd_linf = self.init_pgd(model)
        else:
            adversary_pgd_linf = self.init_pgd(model, test=True)

        adversary_pgd_l2 = self.init_pgdl2(model)
        adversary_mia_linf = self.init_mialinf(model)
        adversary_mia_l2 = self.init_mial2(model)

        t = tqdm(iter(data_loader), leave=False, total=len(data_loader), disable=not self.args['verbose'] > 1)
        for batch_idx, (data, target) in (enumerate(t)):
            data, target = data.to(self.args['device']), target.to(self.args['device'])

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if attack_type == 'pgd_linf':
                data_adversary = adversary_pgd_linf.perturb(data, target)
            elif attack_type == 'pgd_l2':
                data_adversary = adversary_pgd_l2.perturb(data, target)
            elif attack_type == 'mia_linf':
                data_adversary = adversary_mia_linf.perturb(data, target)
            elif attack_type == 'mia_l2':
                data_adversary = adversary_mia_l2.perturb(data, target)

            output = model(data_adversary)
            pred = output.max(1, keepdim=True)[1]
            correct_adv += pred.eq(target.view_as(pred)).sum().item()

        ben_acc = 100. * correct / data_len
        adv_acc = 100. * correct_adv / data_len

        return ben_acc, adv_acc

    def train(self, model):
        if self.args['enable_logging']:
            writer = SummaryWriter(self.args['log_dir'])

        if self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args['lr'])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'],
                                  momentum=self.args['momentum'], nesterov=self.args['nesterov'])

        adversary = self.init_pgd(model)

        best_model = copy.deepcopy(model)
        best_val_acc = 0
        for epoch in range(1, self.args['epochs']+1):
            model.train()

            train_loss = 0
            correct = 0
            true_labels = []
            pred_labels = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.args['device']), target.to(self.args['device'])

                if self.robust_training:
                    # when performing attack, the model needs to be in eval mode
                    # also the parameters should be accumulating gradients
                    with ctx_noparamgrad_and_eval(model):
                        data = adversary.perturb(data, target)

                optimizer.zero_grad()
                output = model(data)

                loss = F.cross_entropy(output, target, reduction='mean')
                loss.backward()

                optimizer.step()
                train_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                true_labels.extend(target.data.cpu().numpy().flatten().tolist())
                pred_labels.extend(pred.data.cpu().numpy().flatten().tolist())

                if batch_idx % 10 == 0:
                    if self.args['verbose'] > 1: print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))

            train_loss /= len(self.train_loader.dataset)
            acc_train = 100. * correct / len(self.train_loader.dataset)

            ben_val_acc, adv_val_acc = self.test_robustness(model, self.val_loader)
            ben_test_acc, adv_test_acc = self.test_robustness(model, self.test_loader)
            avg_val_acc = (ben_val_acc + adv_val_acc) / 2.0

            log_dict = {'bv_acc': ben_val_acc, 'av_acc': adv_val_acc, 'bt_acc': ben_test_acc, 'at_acc': adv_test_acc, 'avg_val': avg_val_acc}

            model_improved = False
            if self.robust_training:
                print_results(self.args, adv_train_acc=acc_train, val_acc=ben_val_acc, test_acc=ben_test_acc, adv_val_acc=adv_val_acc, adv_test_acc=adv_test_acc, avg_val_acc=avg_val_acc, epoch=epoch)

                if avg_val_acc > best_val_acc:
                    if self.args['verbose'] > 0: print('Model improved average acc {} -> {} '.format(best_val_acc, avg_val_acc))
                    best_val_acc = avg_val_acc
                    model_improved = True
            else:
                print_results(self.args, train_acc=acc_train, val_acc=ben_val_acc, test_acc=ben_test_acc, adv_val_acc=adv_val_acc, adv_test_acc=adv_test_acc, avg_val_acc=avg_val_acc, epoch=epoch)

                if ben_val_acc > best_val_acc:
                    if self.args['verbose'] > 0: print('Model improved average acc {} -> {} '.format(best_val_acc, ben_val_acc))
                    best_val_acc = ben_val_acc
                    model_improved = True

            # log stats
            if self.args['enable_logging']:
                writer.add_scalars('{}_large_{}_trainepsilon_{}'.format(self.args['logging_comment'],
                                                                        self.args['run'],
                                                                        self.args['train_epsilon']),
                                                                        log_dict, epoch)

            # save best model
            if self.args['enable_saving'] and model_improved:
                best_model = copy.deepcopy(model)
                if self.args['verbose'] > 0: print('Saving to file ...\n')
                save_model(model, self.args['full_model_path'], self.args)

            # reduce learning rate depending on dataset
            if self.args['optimizer'] == 'sgd':
                if self.args['dataset'] == 'cifar10':
                    adjust_learning_rate_cifar10(self.args, optimizer, epoch)
                elif self.args['dataset'] == 'mnist':
                    adjust_learning_rate_mnist(self.args, optimizer, epoch)
                elif self.args['dataset'] == 'physionet':
                    adjust_learning_rate_physionet(self.args, optimizer, epoch)
                elif self.args['dataset'] == 'shhs':
                    adjust_learning_rate_shhs(self.args, optimizer, epoch)
                # elif self.args['dataset'] == 'unmask':
                #     adjust_learning_rate_cifar10(self.args, optimizer, epoch)
            elif self.args['optimizer'] == 'adam':
                scheduler.step()

        # ben_test_acc, adv_test_acc = self.test_robustness(best_model, self.test_loader)
        # print_results(self.args, test_acc=ben_test_acc, adv_test_acc=adv_test_acc, epoch=self.args['epochs'])

        # return best_model
