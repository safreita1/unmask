from joblib import Parallel, delayed
from tqdm import tqdm
import os
import time
import itertools

from main import AdversarialCompression
from config import *


class Experiments:
    def __init__(self, args):
        self.args = args

    def line_search_epsilon(self):
        self.args['model'] = 'resnet50'

        epsilons = [1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
        class_sets = ['cs3-1']

        parameter_combinations = list(itertools.product(epsilons, class_sets))

        Parallel(n_jobs=6)(
            delayed(self.run_epsilon_experiment)(index, epsilon, class_set)
            for (index, (epsilon, class_set)) in tqdm(enumerate(parameter_combinations)))

    def run_epsilon_experiment(self, index, epsilon, class_set):
        GPU_IDS = [0, 1, 2, 3, 4, 5]

        # to avoid all args being logged at the same time
        time.sleep(int(index / len(GPU_IDS)) * len(GPU_IDS) + GPU_IDS[index % len(GPU_IDS)])

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        DEVICE_ID = GPU_IDS[index % len(GPU_IDS)]
        self.args['cuda_device'] = str(DEVICE_ID)

        self.args['class_set'] = class_set
        self.args['train_epsilon'] = epsilon
        self.args['logging_comment'] = 'Adversarial training epsilon grid search ({}) {} ({}), train epsilon={}'.format(self.args['robust_training'], self.args['model'], class_set, epsilon)

        self.at(loader='val_loader')

    def run_experiment(self, index, model, class_set, robust_train):
        GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

        # to avoid all args being logged at the same time
        time.sleep(int(index / len(GPU_IDS)) * len(GPU_IDS) + GPU_IDS[index % len(GPU_IDS)])


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        DEVICE_ID = GPU_IDS[index % len(GPU_IDS)]
        self.args['cuda_device'] = str(DEVICE_ID)

        self.args['model'] = model
        self.args['class_set'] = class_set
        self.args['robust_training'] = robust_train
        self.args['logging_comment'] = 'Adversarial training ({}) {} ({})'.format(self.args['robust_training'], self.args['model'], self.args['class_set'])

        self.train(loader='test_loader')

    def run_all_experiments(self):
        models = ['vgg16', 'resnet50', 'densenet121']
        class_sets = ['cs3-1', 'cs3-2', 'cs5-1', 'cs5-2']
        robust_training = [True, False]

        # get every combination of ortho lambdas and gamma lambdas
        parameter_combinations = list(itertools.product(models, class_sets, robust_training))

        num_processes = len(parameter_combinations)

        Parallel(n_jobs=num_processes)(
            delayed(self.run_experiment)(index, model, class_set, robust_train)
            for (index, (model, class_set, robust_train)) in tqdm(enumerate(parameter_combinations)))

    def train(self, loader='test_loader'):
        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model, loader=loader, unmask_metrics=False)

    def bt(self, loader='test_loader'):
        print("Benign training")
        self.args['robust_training'] = False

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model, loader=loader, unmask_metrics=True)

    def at(self, loader='test_loader'):
        self.args['robust_training'] = True

        ac = AdversarialCompression(self.args)
        model = ac.get_full_model()
        ac.print_metrics(model, loader=loader, unmask_metrics=False)


def main():
    args = args_unmask

    args['enable_logging'] = True
    args['enable_saving'] = True
    args['seed'] = 6

    ex = Experiments(args)
    # ex.run_all_experiments()
    ex.line_search_epsilon()


if __name__ == '__main__':
    main()


