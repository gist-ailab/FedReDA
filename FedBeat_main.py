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

    # Load and prepare dataset
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    val_dataset = wrap_as_local_dataset(val_dataset, tag='val', dataset_type='ham10000')
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    print('Original data distribution (clean labels count):')
    print(f"→ Total Train samples: {len(train_dataset)}")
    print(f"→ Total Val samples  : {len(val_dataset)}")
    print(f"→ Total Test samples : {len(test_dataset)}")

    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    val_subsets = split_equally_across_clients(val_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)

    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_val_dataset_list = wrap_subsets_to_local_dataset(val_subsets, val_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)

    test_loader = DataLoader(combine_data(clients_test_dataset_list, args.dataset),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n[Client-wise Dataset Sizes]")
    clients_train_loader_list = []
    clients_train_class_num_list = []
    clients_val_loader_list = []
    clients_test_loader_list = []
    clients_train_loader_batch_list = []

    for i in range(args.num_clients):
        print(f"Client {i+1}:")
        print(f"  Train samples: {len(clients_train_dataset_list[i])}")
        print(f"  Test samples : {len(clients_test_dataset_list[i])}")

        train_loader = DataLoader(clients_train_dataset_list[i], batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(clients_val_dataset_list[i], batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        
        local_train_loader_batch = torch.utils.data.DataLoader(dataset=clients_train_dataset_list[i],
                                                               batch_size=args.batch_size,
                                                               num_workers=args.num_workers,
                                                               drop_last=False,
                                                               shuffle=False)

        class_num_list = [0 for _ in range(args.num_classes)]
        for j in range(len(clients_train_dataset_list[i])):
            class_num_list[int(clients_train_dataset_list[i][j][1])] += 1
        
        # Create a tensor from class_num_list and move it to the specified device
        class_num_tensor = torch.cuda.FloatTensor(class_num_list)

        # Perform calculations
        class_p_list = class_num_tensor / class_num_tensor.sum()
        class_p_list = torch.log(class_p_list)
        class_p_list = class_p_list.view(1, -1)
        
        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p_list)
        clients_train_loader_batch_list.append(local_train_loader_batch)
        clients_val_loader_list.append(val_loader)


    # construct model
    print('constructing model...')
    if args.dataset == 'svhn':
        classifier = model.ResNet18(10).cuda()
    if args.dataset == 'cifar10':
        classifier = model.ResNet34(10).cuda()
    if args.dataset == 'ham10000':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
        dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth')
        classifier = model_dino.ReinDinov2(variant, dino_state_dict, args.num_classes)
        classifier.eval()

    # Warm Up
    print('----------Starting Warm Up Classifier Model----------')
    best_acc = 0.
    best_round = 0
    best_model_weights_list = []
    classifier.cuda()
    criterion = nn.CrossEntropyLoss()
    for rd in range(args.round1):
        local_weights_list, local_acc_list = [], []
        selected_id = random.sample(range(args.num_clients), args.num_clients)
        selected_clients_train_loader_list = [clients_train_loader_list[i] for i in selected_id]

        for client_id, client_train_loader in zip(selected_id, selected_clients_train_loader_list):
            print('Warm up Round [%d] Training Client [%d]' % (rd + 1, client_id + 1))
            model_local = copy.deepcopy(classifier)
            model_local.cuda()
            train_acc = 0.
            for epoch in range(args.local_ep):
                model_local.train()
                optimizer_w = torch.optim.AdamW(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
                train_acc = train(client_train_loader, epoch, model_local, optimizer_w, criterion, args, clients_train_class_num_list[client_id])
            local_weights_list.append(copy.deepcopy(model_local.state_dict()))
            local_acc_list.append(train_acc)
        classifier_weights = average_weights(local_weights_list)
        classifier.load_state_dict(classifier_weights)
        val_acc = evaluate(val_loader, classifier)
        train_acc = evaluate(train_loader, classifier)
        print('Warm up Round [%d] Train Acc %.4f %% and Val Accuracy on the %s val data: Model1 %.4f %%' % (
            rd + 1, train_acc, len(val_dataset), val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_round = rd + 1
            best_model_weights_list = copy.deepcopy(local_weights_list)
            torch.save(classifier.state_dict(), model_dir + '/' + 'warmup_model.pth')
    print('Best Round [%d]' % best_round)
    print('----------Finishing Warm Up Classifier Model----------')

    # Distill
    print('----------Start Distilling----------')
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth',weights_only=True))
    classifier.cuda()
    base_model = copy.deepcopy(classifier.state_dict())
    w_avg, w_sq_avg, w_norm = compute_mean_sq(best_model_weights_list, base_model)
    w_var = compute_var(w_avg, w_sq_avg)
    threshold = args.tau
    var_scale = 0.1
    test_acc = evaluate(test_loader, classifier)
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (len(test_dataset), test_acc))
    distilled_dataset_clients_list = []
    distilled_loader_clients_list = []
    for client_id in range(args.num_clients):
        distilled_example_index_list = []
        distilled_example_labels_list = []
        classifier.eval()
        teachers_list = []
        for j in range(args.num_clients):
            mean_grad = copy.deepcopy(w_avg)
            for k in w_avg.keys():
                mean = w_avg[k]
                var = torch.clamp(w_var[k], 1e-6)
                eps = torch.randn_like(mean)
                safe_var = torch.clamp(w_var[k], min=1e-6, max=1e2)
                mean_grad[k] = mean + torch.sqrt(safe_var) * eps * var_scale
            for k in base_model:
                if torch.isnan(base_model[k]).any():
                    print(f"[!] NaN in base_model[{k}]")
            for k in w_avg.keys():
                mean_grad[k] = mean_grad[k] * w_norm[k] + base_model[k]
            teachers_list.append(copy.deepcopy(mean_grad))
        for i, (data, noisy_label, clean_label, indexes) in enumerate(clients_train_loader_batch_list[client_id]):
            data = data.cuda()
            for j in range(len(teachers_list)):
                classifier.load_state_dict(teachers_list[j], strict=False)
                classifier.cuda()
                classifier.eval()
                out = classifier(data)
                if j == 0:
                    logits1 = F.softmax(out, dim=1)
                else:
                    logits1 = torch.add(logits1, F.softmax(out, dim=1))
            logits1 = torch.div(logits1, len(teachers_list))
            logits1_max = torch.max(logits1, dim=1)
            mask = logits1_max[0] > threshold
            distilled_example_index_list.extend(indexes[mask.cpu()])
            distilled_example_labels_list.extend(logits1_max[1].cpu()[mask.cpu()])

        distilled_example_index = np.array(distilled_example_index_list)
        distilled_pseudo_labels = np.array(distilled_example_labels_list)

        if len(distilled_pseudo_labels) > 0:
            source_images = np.array(clients_train_dataset_list[client_id].local_data)
            source_noisy_labels = np.array(clients_train_dataset_list[client_id].local_noisy_labels)
            source_clean_labels = np.array(clients_train_dataset_list[client_id].local_clean_labels)

            distilled_images = source_images[distilled_example_index]
            distilled_noisy_labels = source_noisy_labels[distilled_example_index]
            distilled_clean_labels = source_clean_labels[distilled_example_index]
            
            distilled_acc = (distilled_pseudo_labels == distilled_clean_labels).sum() / len(distilled_pseudo_labels)
            print("Number of distilled examples:" + str(len(distilled_pseudo_labels)))
            print("Accuracy of distilled examples collection:" + str(distilled_acc))
        else:
            print("Number of distilled examples: 0")
            distilled_acc = 0
            distilled_images = np.array([])
            distilled_noisy_labels = np.array([])
            distilled_clean_labels = np.array([])

        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_images.npy', distilled_images)
        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_pseudo_labels.npy', distilled_pseudo_labels)
        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_noisy_labels.npy', distilled_noisy_labels)
        np.save(model_dir + '/' + str(client_id) + '_' + 'distilled_clean_labels.npy', distilled_clean_labels)

        print('building distilled dataset')
        distilled_images = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_images.npy')
        distilled_noisy_labels = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_noisy_labels.npy')
        distilled_pseudo_labels = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_pseudo_labels.npy')
        distilled_clean_labels = np.load(model_dir + '/' + str(client_id) + '_' + 'distilled_clean_labels.npy')
        if args.dataset == 'ham10000':
            distilled_dataset_ = dataset.distilled_dataset(distilled_images,
                                                           distilled_noisy_labels,
                                                           distilled_pseudo_labels,
                                                           transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                                    (0.229, 0.224, 0.225)), ]),
                                                           target_transform=transform_target
                                                           )
        distilled_dataset_clients_list.append(distilled_dataset_)
        train_loader_distilled = torch.utils.data.DataLoader(dataset=distilled_dataset_,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.num_workers,
                                                             drop_last=False,
                                                             shuffle=True)
        distilled_loader_clients_list.append(train_loader_distilled)
    print('----------Finishing Distilling----------')

    # Train Transition Matrix Estimation Network
    print('----------Starting Training Trans Matrix Estimation Model----------')
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    classifier.cuda()
    if args.dataset == 'ham10000':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
        dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth')
        classifier_trans = model_dino.ReinDinov2_trans(variant, dino_state_dict, 49)
        classifier_trans.eval()
        temp = OrderedDict()
        params = classifier_trans.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in params:
                temp[name] = parameter
        params.update(temp)
        classifier_trans.load_state_dict(params)

    classifier_trans.cuda()
    loss_function = nn.NLLLoss()
    lr = args.lr

    for rd in range(args.round2):
        lr = lr * 0.99
        local_weights_list = []
        for client_id in range(args.num_clients):
            client = distilled_loader_clients_list[client_id]
            model_local_trans = copy.deepcopy(classifier_trans)
            model_local_trans.cuda()
            print('Training Transition Estimation Model Round [%d] on Client [%d]' % (rd + 1, client_id + 1))
            for epoch in range(args.local_ep):
                loss_trans = 0.
                model_local_trans.train()
                optimizer_trans = torch.optim.AdamW(model_local_trans.parameters(), lr=args.lr)
                for data, noisy_labels, pseudo_labels, index in client:
                    data = data.cuda()
                    pseudo_labels, noisy_labels = pseudo_labels.cuda(), noisy_labels.cuda()
                    batch_matrix = model_local_trans(data)
                    noisy_class_post = torch.zeros((batch_matrix.shape[0], args.num_classes))
                    for j in range(batch_matrix.shape[0]):
                        pseudo_label_one_hot = torch.nn.functional.one_hot(pseudo_labels[j], args.num_classes).float()
                        pseudo_label_one_hot = pseudo_label_one_hot.unsqueeze(0)
                        noisy_class_post_temp = pseudo_label_one_hot.float().mm(batch_matrix[j])
                        noisy_class_post[j, :] = noisy_class_post_temp
                noisy_class_post = torch.log(noisy_class_post + 1e-12)
                loss = loss_function(noisy_class_post.cuda(), noisy_labels)
                optimizer_trans.zero_grad()
                loss.backward()
                optimizer_trans.step()
                loss_trans += loss.item()
                print('Training Epoch [%d], Loss: %.4f' % (epoch + 1, loss.item()))
            local_weights_list.append(copy.deepcopy(model_local_trans.state_dict()))
        classifier_trans_weights = average_weights_weighted(local_weights_list, distilled_dataset_clients_list)
        classifier_trans.load_state_dict(classifier_trans_weights)
    torch.save(classifier_trans.state_dict(), model_dir + '/' + 'trans_model.pth')
    print('----------Finishing Training Trans Matrix Estimation Model----------')

    # Finetuning
    print('----------Starting Finetuning Classifier Model----------')
    val_acc_list = []
    test_acc_list = []
    best_acc = 0.
    best_round = 0
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    classifier_trans.load_state_dict(torch.load(model_dir + '/' + 'trans_model.pth'))
    classifier.cuda()
    classifier_trans.cuda()
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (
        len(test_dataset), evaluate(test_loader, classifier)))

    for rd in range(args.round3):
        local_weights_list, local_acc_list = [], []
        selected_numb = random.sample(range(args.num_clients), args.num_clients)
        selected_clients = [clients_train_loader_list[i] for i in selected_numb]
        for client_id, client_train_loader in zip(selected_numb, selected_clients):
            model_local = copy.deepcopy(classifier)
            model_local.cuda()
            print('Final Train Round [%d] Training Client [%d]' % (rd + 1, client_id + 1))
            for epoch in range(args.local_ep2):
                model_local.train()
                classifier_trans.eval()
                optimizer_f = torch.optim.Adam(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
                train_acc = train_forward(model_local, client_train_loader, optimizer_f, classifier_trans)
                test_acc = evaluate(test_loader, model_local)
                print('Round [%d] Epoch [%d] Client [%d] Test: %.4f %%' % (rd + 1, epoch + 1, client_id + 1, test_acc))
            local_weights_list.append(copy.deepcopy(model_local.state_dict()))
        classifier_weights = average_weights(local_weights_list)
        classifier.load_state_dict(classifier_weights)
        test_acc = evaluate(test_loader, classifier)
        test_acc_list.append(test_acc)
        print('Round [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
            rd + 1, args.round3, len(test_dataset), test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            best_round = rd + 1
            torch.save(classifier.state_dict(), model_dir + '/' + 'final_model.pth')

    print('Best Round [%d]' % best_round)
    best_id = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[best_id]
    print('Test Acc: ')
    print(test_acc_max)
    return test_acc_max


if __name__ == '__main__':
    acc_list = []
    print(args.num_exp)
    for index_exp in range(args.num_exp):
        print(index_exp)
        args.seed = index_exp + 1
        print(args.print_txt)
        if args.print_txt:
            f = open(save_dir + '_' + str(args.dataset) + '_' +
                     str(args.noise_rate) + '_' + str(args.tau) + '.txt', 'a')
            print(save_dir + '_' + str(args.dataset) + '_' +
                     str(args.noise_rate) + '_' + str(args.tau) + '.txt')
            sys.stdout = f
        print('print finished')
        print('start main')
        acc = main(args)
        acc_list.append(acc)
    print(acc_list)
    print(np.array(acc_list).mean())
    print(np.array(acc_list).std(ddof=1))