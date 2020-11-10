import os
from collections import defaultdict
import torch
from tensorboardX import SummaryWriter

import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not sys.warnoptions:
    warnings.simplefilter("ignore")


from utils import set_random_seed, load_model, weights_init
from models.model_builder import Model_Builder
from train import Train
from config import *


class AdversarialCompression:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_device']
        use_cuda = not args['no_cuda'] and torch.cuda.is_available()
        args['device'] = torch.device("cuda" if use_cuda else "cpu")

        # get current working dir
        args['dir'] = os.getcwd()

        # set logging params
        args['run'] = args['logging_run']
        args['log_dir'] = args['dir'] + '/logs{}/{}/{}/{}/'.format(args['seed'], args['dataset'], args['model'], args['class_set'])

        # define paths for model persistence
        args['chkpnt_dir'] = args['dir'] + '/checkpoints{}/{}/{}/{}/'.format(args['seed'], args['dataset'], args['model'], args['class_set'])

        # decide folder structure
        params = 'rt_{}/eps_{}/num_steps_{}'.format(args['robust_training'], args['train_epsilon'], args['nb_iter'])

        os.makedirs(args['chkpnt_dir']+'/'+'/'.join(params.split('/')[:-1])+'/', exist_ok=True)
        args['full_model_path'] = args['chkpnt_dir'] + params + 'spec_large.pt'
        args['small_model_path'] = args['chkpnt_dir'] + params + 'spec_small.pt'

        # paths for model def and data
        args['model_dir'] = args['dir'] + '/models/'
        args['data_dir'] = args['dir'] + '/data'

        self.args = args

    def set_params(self, model_builder):
        train_loader, val_loader, test_loader = model_builder.get_loaders()
        clip_min, clip_max = 0, 1

        self.spectral_params = {
                                    # adv robustness params
                                    'robust_training': self.args['robust_training'],
                                    'attack': 'pgdInf',
                                    'train_epsilon': self.args['train_epsilon'],
                                    'test_epsilon': self.args['test_epsilon'],
                                    'nb_iter': self.args['nb_iter'],
                                    'eps_iter': self.args['eps_iter'],
                                    'clip_min': clip_min,
                                    'clip_max': clip_max,
                                    'test_nb_iter': self.args['test_nb_iter'],
                                    'test_eps_iter': self.args['test_eps_iter'],
                                    ####
                                    'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader,
                                }

    def get_full_model(self, index=0):
        set_random_seed(self.args['seed'])  # set seed for reproducability

        if os.path.exists(self.args['full_model_path']):
            full_model = load_model(self.args['full_model_path'])
        else:
            model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args)
            self.set_params(model_builder)
            full_model = model_builder.get_model()

            spc = Train(self.args, self.spectral_params)

            # log args
            if self.args['enable_logging'] and (not index):
                args_log = {k: v for (k, v) in self.args.items() if
                            not (isinstance(v, str) or isinstance(v, list) or isinstance(v, torch.device))}
                writer = SummaryWriter(self.args['log_dir'])
                writer.add_scalars('{}_large_{}/args'.format(self.args['logging_comment'], self.args['run']), args_log, 1)

            if self.args['verbose'] > 0: print('\ttraining large model')
            spc.train(full_model)
            full_model = load_model(self.args['full_model_path'])

        return full_model

    def reinit(self, model):
        model.apply(weights_init)
        if self.args['device'] == torch.device('cuda'):
            model = model.cuda()
        return model

    def print_metrics(self, model, loader='test_loader', unmask_metrics=False):
        set_random_seed(self.args['seed'])

        if self.args['device'] == torch.device('cuda'):
            model = model.cuda()

        epsilons = {'pgd_linf': [8.0, 16.0],
                    'pgd_l2': [300.0, 600.0],
                    'mia_linf': [8.0, 16.0],
                    'mia_l2': [300.0, 600.0]}

        attack_types = ['pgd_linf', 'pgd_l2', 'mia_linf', 'mia_l2']  #  'bia'

        no_defense_results = defaultdict(lambda: defaultdict(float))
        unmask_results = defaultdict(lambda: defaultdict(float))

        for attack_type in attack_types:
            for epsilon in epsilons[attack_type]:
                print("Testing model (robust training: {}) with attack {}, epsilon {}, steps {} on {}".format(
                    self.args['robust_training'], attack_type, epsilon, self.args['test_nb_iter'], self.args['class_set']))

                self.args['test_epsilon'] = epsilon
                model_builder = Model_Builder(self.args['model'], self.args['dataset'], self.args['full_model_path'], self.args)
                self.set_params(model_builder)

                spc = Train(self.args, self.spectral_params)
                if unmask_metrics:
                    ben_acc, adv_acc, ben_acc_unmask, adv_acc_unmask = spc.test_unmask_defense(model, self.spectral_params['test_loader'], attack_type=attack_type)

                    unmask_results[attack_type][epsilon] = adv_acc_unmask
                    unmask_results['benign'][epsilon] = ben_acc_unmask
                else:
                    ben_acc, adv_acc = spc.test_robustness(model, self.spectral_params['test_loader'], attack_type=attack_type, attack_strength='test')

                no_defense_results[attack_type][epsilon] = adv_acc
                no_defense_results['benign'][epsilon] = ben_acc

        result_str = '\n\nModel accuracy with {}\n'.format(loader)
        result_str_unmask = '\n\nUnMask Accuracy with {}\n'.format(loader)

        for index in range(2):
            epsilon = epsilons['pgd_linf'][index]
            l2_eps = epsilons['pgd_l2'][index]

            result_str += '\ttest_epsilon {:0.2f}, steps {} : {:0.2f} (BEN ACC), {:0.2f} (PGD-LINF ACC), {:0.2f} (PGD-L2 ACC eps {}), ' \
                          '{:0.2f} (MIA-LINF ACC), {:0.2f} (MIA-L2 ACC)\n'.format(
                epsilon, self.args['test_nb_iter'], no_defense_results['benign'][epsilon], no_defense_results['pgd_linf'][epsilon],
                no_defense_results['pgd_l2'][l2_eps], l2_eps, no_defense_results['mia_linf'][epsilon], no_defense_results['mia_l2'][l2_eps])

            result_str_unmask += '\ttest_epsilon {:0.2f}, steps {} : {:0.2f} (BEN ACC), {:0.2f} (PGD-LINF ACC), {:0.2f} (PGD-L2 ACC eps {}), ' \
                                 '{:0.2f} (MIA-LINF ACC), {:0.2f} (MIA-L2 ACC) \n'.format(
                epsilon, self.args['test_nb_iter'], unmask_results['benign'][epsilon], unmask_results['pgd_linf'][epsilon],
                unmask_results['pgd_l2'][l2_eps], l2_eps, unmask_results['mia_linf'][epsilon], unmask_results['mia_l2'][l2_eps])

        if self.args['verbose'] > 0: print(result_str)
        if self.args['verbose'] > 0 and unmask_metrics: print(result_str_unmask)

        if self.args['enable_logging']:
            file_path = '{}/{}_{}/result_info_train_epsilon_{}_metrics.txt'.format(self.args['log_dir'],
                                                                           self.args['logging_comment'],
                                                                           self.args['run'],
                                                                           self.args['train_epsilon'])

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write(result_str)
                if unmask_metrics: file.write(result_str_unmask)


def main():
    args = args_unmask
    ac = AdversarialCompression(args)
    model = ac.get_full_model(0)
    ac.print_metrics(model, loader='test_loader', unmask_metrics=True)


if __name__ == '__main__':
    main()
