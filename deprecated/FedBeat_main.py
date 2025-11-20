import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno
import sys, multiprocessing as mp
mp.set_executable(sys.executable)
mp.set_start_method("spawn", force=True)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import sys
sys.path.append('/home/work/Workspaces/yunjae_heo/FedLNL/')
sys.path.append('/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo/')
import numpy as np
import copy
import random
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader, Dataset
from utils.utils import add_noise, set_seed
from dataset.dataset import get_dataset
from model_dino import ReinDinov2, ReinDinov2_trans
from dino_variant import _small_variant
import argparse
# from util.averaging import average_weights, average_weights_weighted
from update import train, evaluate, train_forward
from ensemble import compute_var, compute_mean_sq
import dataset.dataset as dataset
from PIL import Image

def worker_init_fn(_):
    tmpdir = f"/home/work/DATA1/tmp/worker_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

_old_rmtree = shutil.rmtree
def safe_rmtree(path,*a,**kw):
    try: return _old_rmtree(path,*a,**kw)
    except OSError as e:
        if e.errno == errno.EBUSY: return
        raise
shutil.rmtree = safe_rmtree

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_weighted(w, dataset_list):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    total = 0.
    num_list = []
    for i in range(len(w)):
        num_list.append(len(dataset_list[i]))
        total += len(dataset_list[i])
    for key in w_avg.keys():
        w_avg[key] *= num_list[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * num_list[i]
        w_avg[key] = torch.div(w_avg[key], total)
    return w_avg

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers")
    parser.add_argument('--result_dir', type=str, default='./results/', help="results dir")
    parser.add_argument('--noise_rate', type=float, default=1.0, help="noise rate")


    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='ich', help='dataset name')
    parser.add_argument('--model', type=str,
                        default='ReinDinov2', help='model name')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='base learning rate')
    parser.add_argument('--pretrained', type=int,  default=1)
    parser.add_argument('--n_classes', type=int, default=5, help="number of classes")


    # for FL
    parser.add_argument('--n_clients', type=int,  default=20,
                        help='number of users') 
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
    parser.add_argument('--local_ep', type=int, default=5, help='local epoch')
    parser.add_argument('--round1', type=int,  default=10, help='rounds')
    parser.add_argument('--round2', type=int,  default=10, help='rounds')
    parser.add_argument('--round3', type=int,  default=40, help='rounds')



    parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a', type=float,  default=0.8, help='a')
    parser.add_argument('--warm', type=int,  default=1)

    # noise
    parser.add_argument('--level_n_system', type=float, default=1.0, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default="instance", help="type of noise")

    # FedBeat specific arguments
    parser.add_argument('--num_exp', type=int, default=1, help='number of experiments')
    parser.add_argument('--print_txt', type=bool, default=True, help='print txt')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for fine-tuning')
    parser.add_argument('--lr_f', type=float, default=3e-4, help='learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--tau', type=float, help='threshold', default=0.4)
    parser.add_argument('--local_ep2', type=int, help='number of local epochs for finetuning', default=5)
    parser.add_argument('--trans_rounds', type=int, default=50, help='rounds for training transition matrix estimation')
    
    args = parser.parse_args()
    return args

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, clean_labels):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.clean_labels = clean_labels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        clean_label = self.clean_labels[self.idxs[item]]
        return image, label, clean_label, self.idxs[item]

def main(args):
    # seed
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    save_dir = os.path.join(args.result_dir, args.dataset, 'iid_' + str(args.iid))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model_dir = os.path.join(save_dir, str(args.seed) + '_rate_' + str(args.noise_rate) + '_threshold_' + str(args.tau))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load and prepare dataset
    dataset_train, dataset_test, dict_users = get_dataset(args)
    print(f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    print(f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(
        args, y_train, dict_users, total_dataset=dataset_train)
    original_targets = dataset_train.targets
    dataset_train.targets = y_train_noisy

    test_loader = DataLoader(dataset_test,
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n[Client-wise Dataset Sizes]")
    clients_train_loader_list = []
    clients_train_class_num_list = []
    clients_train_loader_batch_list = []

    for i in range(args.n_clients):
        train_loader = DataLoader(DatasetSplit(dataset_train, dict_users[i], original_targets), batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        
        local_train_loader_batch = torch.utils.data.DataLoader(dataset=DatasetSplit(dataset_train, dict_users[i], original_targets),
                                                               batch_size=args.batch_size,
                                                               num_workers=args.num_workers,
                                                               drop_last=False,
                                                               shuffle=False)

        class_num_list = [0 for _ in range(args.n_classes)]
        for idx in dict_users[i]:
            class_num_list[int(dataset_train.targets[idx])] += 1
        
        class_num_tensor = torch.cuda.FloatTensor(class_num_list)

        class_p_list = class_num_tensor / class_num_tensor.sum()
        class_p_list = torch.log(class_p_list)
        class_p_list = class_p_list.view(1, -1)
        
        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p_list)
        clients_train_loader_batch_list.append(local_train_loader_batch)

    # construct model
    print('constructing model...')
    variant = _small_variant
    dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False)
    classifier = ReinDinov2(variant=variant, dino_state_dict=dino_state_dict, num_classes=args.n_classes)
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
        selected_id = random.sample(range(args.n_clients), args.n_clients)
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
        val_acc = evaluate(test_loader, classifier)
        train_acc = evaluate(clients_train_loader_list[0], classifier)
        print('Warm up Round [%d] Train Acc %.4f %% and Val Accuracy on the %s val data: Model1 %.4f %%' % (
            rd + 1, train_acc, len(dataset_test), val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_round = rd + 1
            best_model_weights_list = copy.deepcopy(local_weights_list)
            torch.save(classifier.state_dict(), os.path.join(model_dir, 'warmup_model.pth'))
    print('Best Round [%d]' % best_round)
    print('----------Finishing Warm Up Classifier Model----------')

    # Distill
    print('----------Start Distilling----------')
    classifier.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth'),weights_only=True))
    classifier.cuda()
    base_model = copy.deepcopy(classifier.state_dict())
    w_avg, w_sq_avg, w_norm = compute_mean_sq(best_model_weights_list, base_model)
    w_var = compute_var(w_avg, w_sq_avg)
    threshold = args.tau
    var_scale = 0.1
    test_acc = evaluate(test_loader, classifier)
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (len(dataset_test), test_acc))
    distilled_dataset_clients_list = []
    distilled_loader_clients_list = []
    for client_id in range(args.n_clients):
        distilled_example_index_list = []
        distilled_example_labels_list = []
        classifier.eval()
        teachers_list = []
        for j in range(args.n_clients):
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
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):  # ← 추가
                    out = classifier(data)
                # out = classifier(data)
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
            # source_images = np.array(dataset_train.data)[distilled_example_index]
            # source_images = np.array(dataset_train.images)[distilled_example_index]

            src_imgs = [dataset_train[i][0].numpy() for i in distilled_example_index]
            source_noisy_labels = np.array(dataset_train.targets)[distilled_example_index]
            source_clean_labels = np.array(original_targets)[distilled_example_index]
            
            print("Unique shapes:", {tuple(a.shape) for a in src_imgs})
            
            # distilled_images = source_images
            distilled_images = np.stack(src_imgs, axis=0).astype(np.uint8)  # 또는 .astype(np.float32)
            distilled_noisy_labels = source_noisy_labels
            distilled_clean_labels = source_clean_labels
            
            distilled_acc = (distilled_pseudo_labels == distilled_clean_labels).sum() / len(distilled_pseudo_labels)
            print("Number of distilled examples:" + str(len(distilled_pseudo_labels)))
            print("Accuracy of distilled examples collection:" + str(distilled_acc))
        else:
            print("Number of distilled examples: 0")
            distilled_acc = 0
            distilled_images = np.array([])
            distilled_noisy_labels = np.array([])
            distilled_clean_labels = np.array([])

        np.save(os.path.join(model_dir, str(client_id) + '_' + 'distilled_images.npy'), distilled_images)
        np.save(os.path.join(model_dir, str(client_id) + '_' + 'distilled_pseudo_labels.npy'), distilled_pseudo_labels)
        np.save(os.path.join(model_dir, str(client_id) + '_' + 'distilled_noisy_labels.npy'), distilled_noisy_labels)
        np.save(os.path.join(model_dir, str(client_id) + '_' + 'distilled_clean_labels.npy'), distilled_clean_labels)

        print('building distilled dataset')
        distilled_images = np.load(os.path.join(model_dir, str(client_id) + '_' + 'distilled_images.npy'))
        distilled_noisy_labels = np.load(os.path.join(model_dir, str(client_id) + '_' + 'distilled_noisy_labels.npy'))
        distilled_pseudo_labels = np.load(os.path.join(model_dir, str(client_id) + '_' + 'distilled_pseudo_labels.npy'))
        distilled_clean_labels = np.load(os.path.join(model_dir, str(client_id) + '_' + 'distilled_clean_labels.npy'))
        
        distilled_dataset_ = dataset.distilled_dataset(distilled_images,
                                                        distilled_noisy_labels,
                                                        distilled_pseudo_labels,
                                                        transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010)), ]),
                                                        )
        distilled_dataset_clients_list.append(distilled_dataset_)
        if len(distilled_dataset_) > 0:
            train_loader_distilled = torch.utils.data.DataLoader(dataset=distilled_dataset_,
                                                                batch_size=args.batch_size,
                                                                num_workers=args.num_workers,
                                                                drop_last=False,
                                                                shuffle=True)
        else:
            train_loader_distilled = []
        distilled_loader_clients_list.append(train_loader_distilled)
    print('----------Finishing Distilling----------')

    # Train Transition Matrix Estimation Network
    print('----------Starting Training Trans Matrix Estimation Model----------')
    classifier.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth')))
    classifier.cuda()
    
    classifier_trans = ReinDinov2_trans(variant, dino_state_dict, args.n_classes)
    classifier_trans.eval()
    temp = OrderedDict()
    params = classifier_trans.state_dict()
    classifier.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth')))
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
        for client_id in range(args.n_clients):
            client = distilled_loader_clients_list[client_id]
            if client == [] or len(client) == 0:
                print(f"[Skip] No distilled data for client {client_id}.")
                continue
            model_local_trans = copy.deepcopy(classifier_trans)
            model_local_trans.cuda()
            print('Training Transition Estimation Model Round [%d] on Client [%d]' % (rd + 1, client_id + 1))
            for epoch in range(args.local_ep):
                loss_trans = 0.
                model_local_trans.train()
                optimizer_trans = torch.optim.AdamW(model_local_trans.parameters(), lr=lr)
                for data, noisy_labels, pseudo_labels, index in client:
                    data = data.cuda()
                    pseudo_labels, noisy_labels = pseudo_labels.cuda(), noisy_labels.cuda()
                    batch_matrix = model_local_trans(data)
                    noisy_class_post = torch.zeros((batch_matrix.shape[0], args.n_classes))
                    for j in range(batch_matrix.shape[0]):
                        pseudo_label_one_hot = torch.nn.functional.one_hot(pseudo_labels[j], args.n_classes).float()
                        pseudo_label_one_hot = pseudo_label_one_hot.unsqueeze(0)
                        noisy_class_post_temp = pseudo_label_one_hot.float().mm(batch_matrix[j].view(args.n_classes, args.n_classes))
                        noisy_class_post[j, :] = noisy_class_post_temp
                noisy_class_post = torch.log(noisy_class_post + 1e-12)
                loss = loss_function(noisy_class_post.cuda(), noisy_labels)
                optimizer_trans.zero_grad()
                loss.backward()
                optimizer_trans.step()
                loss_trans += loss.item()
                print('Training Epoch [%d], Loss: %.4f' % (epoch + 1, loss.item()))
            local_weights_list.append(copy.deepcopy(model_local_trans.state_dict()))
        if local_weights_list:
            classifier_trans_weights = average_weights_weighted(local_weights_list, distilled_dataset_clients_list)
            classifier_trans.load_state_dict(classifier_trans_weights)
    torch.save(classifier_trans.state_dict(), os.path.join(model_dir, 'trans_model.pth'))
    print('----------Finishing Training Trans Matrix Estimation Model----------')

    # Finetuning
    print('----------Starting Finetuning Classifier Model----------')
    test_acc_list = []
    best_acc = 0.
    best_round = 0
    classifier.load_state_dict(torch.load(os.path.join(model_dir, 'warmup_model.pth')))
    classifier_trans.load_state_dict(torch.load(os.path.join(model_dir, 'trans_model.pth')))
    classifier.cuda()
    classifier_trans.cuda()
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (
        len(dataset_test), evaluate(test_loader, classifier)))

    for rd in range(args.round3):
        local_weights_list, local_acc_list = [], []
        selected_numb = random.sample(range(args.n_clients), args.n_clients)
        selected_clients = [clients_train_loader_list[i] for i in selected_numb]
        for client_id, client_train_loader in zip(selected_numb, selected_clients):
            model_local = copy.deepcopy(classifier)
            model_local.cuda()
            print('Final Train Round [%d] Training Client [%d]' % (rd + 1, client_id + 1))
            for epoch in range(args.local_ep2):
                model_local.train()
                classifier_trans.eval()
                optimizer_f = torch.optim.Adam(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
                train_forward(model_local, client_train_loader, optimizer_f, classifier_trans)
                test_acc = evaluate(test_loader, model_local)
                print('Round [%d] Epoch [%d] Client [%d] Test: %.4f %%' % (rd + 1, epoch + 1, client_id + 1, test_acc))
            local_weights_list.append(copy.deepcopy(model_local.state_dict()))
        classifier_weights = average_weights(local_weights_list)
        classifier.load_state_dict(classifier_weights)
        test_acc = evaluate(test_loader, classifier)
        test_acc_list.append(test_acc)
        print('Round [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
            rd + 1, args.round3, len(dataset_test), test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            best_round = rd + 1
            torch.save(classifier.state_dict(), os.path.join(model_dir, 'final_model.pth'))

    print('Best Round [%d]' % best_round)
    best_id = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[best_id]
    print('Test Acc: ')
    print(test_acc_max)
    return test_acc_max


if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    acc_list = []
    for index_exp in range(args.num_exp):
        print(f"--- Running Experiment {index_exp+1}/{args.num_exp} ---")
        args.seed = index_exp + 1
        
        if args.print_txt:
            log_file_path = os.path.join(args.result_dir, args.dataset, f'log_{args.noise_rate}_{args.tau}_{(args.level_n_upperb+args.level_n_lowerb)/2}.txt')
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            sys.stdout = open(log_file_path, 'a')

        print(f"Arguments: {args}")
        acc = main(args)
        acc_list.append(acc)
        
        if args.print_txt:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print(f"All Accuracies: {acc_list}")
    if len(acc_list) > 0:
        print(f"Average Accuracy: {np.array(acc_list).mean()}")
        print(f"Standard Deviation: {np.array(acc_list).std(ddof=1)}")
