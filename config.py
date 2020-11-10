args_unmask = {
    'no_cuda': False,
    'cuda_device': '0',
    'seed': 5,
    'model': 'densenet121',  # vgg16, resnet50, resnet101, densenet121
    'dataset': 'unmask',
    'class_set': 'cs5-2',
    # training params
    'val_ratio': 0.1,
    'batch_size': 32,
    'epochs': 30,
    'optimizer': 'sgd',  # can be sgd or Adam
    'lr': 0.001,
    'weight_decay': 0.0002,
    'momentum': 0.9,
    'nesterov': False,
    #######
    # adv defense params
    'robust_training': False,
    'train_epsilon': 4.0,
    'eps_iter': 2.0,
    'nb_iter': 7,

    'test_epsilon': 8.0,
    'test_eps_iter': 2.0,
    'test_nb_iter': 20,
    #######
    # logging params
    'enable_logging': True,
    'enable_saving': True,
    'logging_comment': 'No adversarial training DenseNet121 (cs5-2) train epsilon=4',
    'logging_run': 1,
    #######
    'verbose': 2
}

object_dict = {
    'cs3-1': ['car', 'person', 'train'],
    'cs3-2': ['person', 'dog', 'bird'],
    'cs5-1': ['dog', 'car', 'bottle', 'train', 'person'],
    'cs5-2': ['dog', 'car', 'bird', 'train', 'person'],
}