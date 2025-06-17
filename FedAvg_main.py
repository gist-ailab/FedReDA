import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import sys
import numpy as np
import copy
import random

from collections import OrderedDict, Counter
from util import util
from util.util import *
from torch.utils.data import DataLoader
from util import dataset
from util.options import args_parser
from util.util import get_prob, create_data, balance_data, combine_data, load_data, transform_target, wrap_as_local_dataset

from rein import *
import update
import model
import model_trans
import model_dino
import dino_variant
from update import train, average_weights, average_weights_weighted, evaluate, train_forward
from ensemble import compute_var, compute_mean_sq

args = args_parser()
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True

# result path
save_dir = args.result_dir + '/' + args.dataset + '/iid_' + str(args.iid)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(args)
    # model path
    model_dir = save_dir + str(args.seed) + '_rate_' + str(args.noise_rate) + '_threshold_' + str(args.tau)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load and balance dataset
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    val_dataset = wrap_as_local_dataset(val_dataset, tag='val', dataset_type='ham10000')
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    print('total original data counter')
    print(Counter(np.array(train_dataset.local_clean_labels)))
    print(Counter(np.array(val_dataset.local_clean_labels)))
    print(Counter(np.array(test_dataset.local_clean_labels)))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    # used for validation
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    # clients data split
    if args.iid:
        train_prob = (1.0 / args.num_classes) * np.ones((args.num_clients, args.num_classes))
    else:
        if not os.path.exists(model_dir + '/' + 'train_prob.npy'):
            train_prob = get_prob(args.num_clients, p=1.0)
            np.save(model_dir + '/' + 'train_prob.npy', np.array(train_prob))
        else:
            train_prob = np.load(model_dir + '/' + 'train_prob.npy')
    clients_train_loader_list = []
    clients_train_loader_batch_list = []
    clients_test_loader_list = []

    # if args.iid:
    #     clients_train_dataset_list = create_data(train_prob, len(train_dataset.local_data) / args.num_clients,
    #                                              train_dataset, args.num_classes, args.dataset, args.seed)
    #     clients_test_dataset_list = create_data(train_prob, int(len(test_dataset.local_data) / args.num_clients),
    #                                             test_dataset, args.num_classes, args.dataset, args.seed)
    #     clients_test_dataset_combination = combine_data(clients_test_dataset_list, args.dataset)
    #     clients_train_dataset_combination = combine_data(clients_train_dataset_list, args.dataset)
    # else:
    #     # may fail due to incorrect prob
    #     clients_train_dataset_list = create_data(train_prob,
    #                                              int(len(train_dataset.local_data) * 0.5 / args.num_clients),
    #                                              train_dataset, args.num_classes, args.dataset, args.seed)
    #     clients_test_dataset_list = create_data(train_prob, int(len(test_dataset.local_data) * 0.5 / args.num_clients),
    #                                             test_dataset, args.num_classes, args.dataset, args.seed)
    #     clients_test_dataset_combination = combine_data(clients_test_dataset_list, args.dataset)
    #     clients_train_dataset_combination = combine_data(clients_train_dataset_list, args.dataset)

    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)

    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)
    clients_test_dataset_combination = combine_data(clients_test_dataset_list, args.dataset)
    clients_train_dataset_combination = combine_data(clients_train_dataset_list, args.dataset)

    train_loader = torch.utils.data.DataLoader(dataset=clients_train_dataset_combination,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=False)
    print('total test data counter')
    print(Counter(clients_test_dataset_combination.local_clean_labels))
    # used for test (i.e. only have clean labels)
    test_loader = torch.utils.data.DataLoader(dataset=clients_test_dataset_combination,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    for i in range(args.num_clients):
        print('Client [%d] train and test data counter' % (i + 1))
        print(Counter(clients_train_dataset_list[i].local_clean_labels))
        print(Counter(clients_test_dataset_list[i].local_clean_labels))
        print('Client [%d] train noisy data counter' % (i + 1))
        print(Counter(clients_train_dataset_list[i].local_noisy_labels))
        local_train_loader = torch.utils.data.DataLoader(dataset=clients_train_dataset_list[i],
                                                         batch_size=args.batch_size,
                                                         num_workers=args.num_workers,
                                                         drop_last=False,
                                                         shuffle=True)
        # for distilling
        local_train_loader_batch = torch.utils.data.DataLoader(dataset=clients_train_dataset_list[i],
                                                               batch_size=args.batch_size,
                                                               num_workers=args.num_workers,
                                                               drop_last=False,
                                                               shuffle=False)

        local_test_loader = torch.utils.data.DataLoader(dataset=clients_test_dataset_list[i],
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        drop_last=False,
                                                        shuffle=True)
        clients_train_loader_list.append(local_train_loader)
        clients_train_loader_batch_list.append(local_train_loader_batch)
        clients_test_loader_list.append(local_test_loader)

    # construct model
    print('constructing model...')
    if args.dataset == 'svhn':
        classifier = model.ResNet18(10).cuda()
    if args.dataset == 'cifar10':
        classifier = model.ResNet34(10).cuda()
    if args.dataset == 'ham10000':
        # classifier = model.ResNet34(10).cuda()
        # DINO classifier
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
        pretrained = torch.hub.load('facebookresearch/dinov2', model_load)
        dino_state_dict = pretrained.state_dict()
        classifier = model_dino.ReinDinov2(variant, dino_state_dict, args.num_classes)
        classifier.eval()

    print('----------Starting FedAvg Training----------')
    best_acc = 0.
    classifier.cuda()
    classifier.train()
    for round in range(args.round3):
        print(f'[Global Round {round+1}]')

        local_weights = []
        for client_id in range(args.num_clients):
            model_local = copy.deepcopy(classifier)
            model_local.cuda()
            model_local.train()

            # optimizer = torch.optim.SGD(model_local.parameters(), lr=args.lr, momentum=args.momentum)
            optimizer = torch.optim.AdamW(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)

            for epoch in range(args.local_ep):
                train_acc = train(clients_train_loader_list[client_id], epoch, model_local, optimizer, args)
                print(f'[Client {client_id+1}] Local Epoch {epoch+1} Train Acc {train_acc}')
            local_weights.append(copy.deepcopy(model_local.state_dict()))

        # FedAvg
        global_weights = average_weights(local_weights)
        classifier.load_state_dict(global_weights)

        # Evaluation
        val_acc = evaluate(val_loader, classifier)
        test_acc = evaluate(test_loader, classifier)
        print(f'Validation Accuracy: {val_acc:.4f} %, Test Accuracy: {test_acc:.4f} %')

        if test_acc > best_acc:
            best_acc = test_acc
            # torch.save(classifier.state_dict(), os.path.join(model_dir, 'fedavg_best_model.pth'))
            # print(f'New best model saved with test accuracy: {best_acc:.4f} %')

    print('----------Finished FedAvg Training----------')
    print(f'Best Test Accuracy: {best_acc:.4f} %')

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)