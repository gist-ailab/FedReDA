import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist
import json
from math import inf
from scipy import stats
import torchvision.transforms as transforms
from torch.utils.data import Subset

from util import dataset

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.noisy_client_rate, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.min_noisy_level) * gamma_c_initial + args.min_noisy_level
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids

def cross_entropy_soft_label(pred_logits, soft_targets, reduction='none'):
    """
    Cross Entropy loss that supports soft targets.

    Args:
        pred_logits (Tensor): (B, C) logits output from the model (before softmax).
        soft_targets (Tensor): (B, C) soft labels (e.g. with label smoothing).
        reduction (str): 'none' | 'mean' | 'sum'
    
    Returns:
        loss (Tensor)
    """
    log_probs = F.log_softmax(pred_logits, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1)  # shape: (B,)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction
    
recall_level_default = 0.95

def validation_accuracy(model, loader, device, mode='rein'):
    total = 0
    correct = 0

    def linear(model, inputs):
        f = model(inputs)
        outputs = model.linear(f)
        return outputs

    def dual_rein(model, inputs):
        f = model.forward_fused_features(inputs)
        f = f[:, 0, :]
        outputs = model.linear_rein(f)
        return outputs

    def rein(model, inputs):
        f = model.forward_features(inputs)
        f = f[:, 0, :]
        outputs = model.linear_rein(f)
        return outputs
        
    def rein1(model, inputs):
        f = model.forward_features1(inputs)
        f = f[:, 0, :]
        outputs = model.linear_rein(f)
        return outputs
    
    def rein2(model, inputs):
        f = model.forward_features2(inputs)
        f = f[:, 0, :]
        outputs = model.linear_rein(f)
        return outputs

    def no_rein(model, inputs):
        f = model.forward_features_no_rein(inputs)
        f = f[:, 0, :]
        outputs = model.linear(f)
        return outputs

    if mode == 'rein':
        # Use rein() if model has forward_features
        out = rein
    elif mode == 'rein1':
        out = rein1
    elif mode == 'rein2':
        out = rein2
    elif mode == 'dual':
        out = dual_rein
    elif mode == 'no_rein':
        out = no_rein
    else:
        out = linear

    model.to(device)
    model.eval()
    with torch.no_grad():
        for datas in loader:
            if len(datas)==2:
                inputs, targets = datas
            if len(datas)==4:
                inputs, targets, _, _ = datas
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = out(model, inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    return correct / total


#################################FedBeat Utils#####################################

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()

    return target


def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.dataset == 'fmnist':
        train_dataset = dataset.fashionmnist_dataset(True,
                                                     transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.1307, ), (0.3081, )), ]),
                                                     target_transform=transform_target,
                                                     noise_rate=args.noise_rate,
                                                     split_percentage=args.split_percentage,
                                                     seed=args.seed)

        val_dataset = dataset.fashionmnist_dataset(False,
                                                   transform=transforms.Compose([
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.1307, ), (0.3081, )), ]),
                                                   target_transform=transform_target,
                                                   noise_rate=args.noise_rate,
                                                   split_percentage=args.split_percentage,
                                                   seed=args.seed)

        test_dataset = dataset.fashionmnist_test_dataset(transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.1307, ), (0.3081, )), ]),
                                                         target_transform=transform_target)

    if args.dataset == 'cifar10':
        train_dataset = dataset.cifar10_dataset(True,
                                                transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]),
                                                target_transform=transform_target,
                                                noise_rate=args.noise_rate,
                                                split_percentage=args.split_percentage,
                                                seed=args.seed)

        val_dataset = dataset.cifar10_dataset(False,
                                              transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]),
                                              target_transform=transform_target,
                                              noise_rate=args.noise_rate,
                                              split_percentage=args.split_percentage,
                                              seed=args.seed)

        test_dataset = dataset.cifar10_test_dataset(transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]),
                                                    target_transform=transform_target)

    if args.dataset == 'svhn':
        train_dataset = dataset.svhn_dataset(True,
                                             transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                             target_transform=transform_target,
                                             noise_rate=args.noise_rate,
                                             split_percentage=args.split_percentage,
                                             seed=args.seed)

        val_dataset = dataset.svhn_dataset(False,
                                           transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                           target_transform=transform_target,
                                           noise_rate=args.noise_rate,
                                           split_percentage=args.split_percentage,
                                           seed=args.seed)

        test_dataset = dataset.svhn_test_dataset(transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                                 target_transform=transform_target)
        
    if args.dataset == 'ham10000':
        train_dataset = dataset.ham10000_dataset(True,
                                                transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ]),
                                                target_transform=transform_target,
                                                noise_rate=args.noise_rate,
                                                split_percentage=args.split_percentage,
                                                seed=args.seed)

        val_dataset = dataset.ham10000_dataset(False,
                                              transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ]),
                                              target_transform=transform_target,
                                              noise_rate=args.noise_rate,
                                              split_percentage=args.split_percentage,
                                              seed=args.seed)

        test_dataset = dataset.ham10000_test_dataset(transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ]),
                                                    target_transform=transform_target)


    return train_dataset, val_dataset, test_dataset


def get_prob(num_clients, num_classes=10, p=1):
    return np.random.dirichlet(np.repeat(p, num_classes), num_clients)


def create_data(prob, size_per_client, total_dataset, num_classes=10, dataset_type='cifar10', seed=1):
    np.random.seed(int(seed))
    total_each_class = size_per_client * np.sum(prob, 0)
    data = total_dataset.local_data
    noisy_label = total_dataset.local_noisy_labels
    clean_label = total_dataset.local_clean_labels

    all_class_set = []
    for i in range(num_classes):
        sub_data = data[clean_label == i]
        sub_clean_label = clean_label[clean_label == i]
        sub_noisy_label = noisy_label[clean_label == i]

        num_to_sample = min(int(total_each_class[i]), len(sub_data))  # 안전하게 제한
        rand_index = np.random.choice(len(sub_data), size=num_to_sample, replace=False).astype(int)
        sub2_data = sub_data[rand_index]
        sub2_clean_label = sub_clean_label[rand_index]
        sub2_noisy_label = sub_noisy_label[rand_index]
        sub2_set = (sub2_data, sub2_clean_label, sub2_noisy_label)
        all_class_set.append(sub2_set)

    index = [0 for _ in range(num_classes)]
    clients = []

    for m in range(prob.shape[0]):
        clean_labels = []
        noisy_labels = []
        images = []

        for n in range(num_classes):
            image = all_class_set[n][0][index[n]:index[n] + int(prob[m][n] * size_per_client)]
            clean_label = all_class_set[n][1][index[n]:index[n] + int(prob[m][n] * size_per_client)]
            noisy_label = all_class_set[n][2][index[n]:index[n] + int(prob[m][n] * size_per_client)]
            index[n] = index[n] + int(prob[m][n] * size_per_client)

            clean_labels.extend(clean_label)
            noisy_labels.extend(noisy_label)
            images.extend(image)

        images = np.array(images)
        clean_labels = np.array(clean_labels)
        noisy_labels = np.array(noisy_labels)

        if dataset_type == 'fmnist':
            client_dataset = dataset.local_dataset(images,
                                                   noisy_labels,
                                                   clean_labels,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,), (0.3081,)), ]),
                                                   target_transform=transform_target
                                                   )
        if dataset_type == 'cifar10':
            client_dataset = dataset.local_dataset(images,
                                                   noisy_labels,
                                                   clean_labels,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010)), ]),
                                                   target_transform=transform_target
                                                   )
        if dataset_type == 'svhn':
            client_dataset = dataset.local_dataset(images,
                                                   noisy_labels,
                                                   clean_labels,
                                                   transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                                   target_transform=transform_target
                                                   )

        if dataset_type == 'ham10000':
            client_dataset = dataset.local_dataset(images,
                                                    noisy_labels,
                                                    clean_labels,
                                                    transform=transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ]),
                                                    target_transform=transform_target
                                                    )
        clients.append(client_dataset)
    return clients

def create_data_balanced_with_prob(prob, total_dataset, num_classes=7, dataset_type='ham10000'):
    np.random.seed(1)
    data = total_dataset.local_data
    noisy_label = total_dataset.local_noisy_labels
    clean_label = total_dataset.local_clean_labels

    class_indices = {i: np.where(clean_label == i)[0] for i in range(num_classes)}
    class_counters = {i: 0 for i in range(num_classes)}

    min_per_client = int(min([len(class_indices[i]) for i in range(num_classes)]) / prob.shape[0])

    clients = []
    for m in range(prob.shape[0]):
        client_indices = []
        for c in range(num_classes):
            count = int(prob[m][c] * min_per_client * num_classes)
            indices = class_indices[c][class_counters[c]:class_counters[c] + count]
            class_counters[c] += count
            client_indices.extend(indices)

        np.random.shuffle(client_indices)
        client_data = data[client_indices]
        client_noisy = noisy_label[client_indices]
        client_clean = clean_label[client_indices]

        client_dataset = dataset.local_dataset(client_data, client_noisy, client_clean,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                                        (0.229, 0.224, 0.225)), ]),
                                               target_transform=transform_target)
        clients.append(client_dataset)

    return clients


def combine_data(clients_dataset_list, dataset_type='cifar10'):
    clean_labels = []
    noisy_labels = []
    data = []
    for total_dataset in clients_dataset_list:
        data.extend(total_dataset.local_data)
        noisy_labels.extend(total_dataset.local_noisy_labels)
        clean_labels.extend(total_dataset.local_clean_labels)
    data = np.array(data)
    noisy_labels = np.array(noisy_labels)
    clean_labels = np.array(clean_labels)
    if dataset_type == 'fmnist':
        client_dataset = dataset.local_dataset(data,
                                               noisy_labels,
                                               clean_labels,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,)), ]),
                                               target_transform=transform_target
                                               )
    if dataset_type == 'cifar10':
        client_dataset = dataset.local_dataset(data,
                                               noisy_labels,
                                               clean_labels,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010)), ]),
                                               target_transform=transform_target
                                               )
    if dataset_type == 'svhn':
        client_dataset = dataset.local_dataset(data,
                                               noisy_labels,
                                               clean_labels,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                               target_transform=transform_target
                                               )
    if dataset_type == 'ham10000':
        client_dataset = dataset.local_dataset(data,
                                                noisy_labels,
                                                clean_labels,
                                                transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ]),
                                                target_transform=transform_target
                                                )

    return client_dataset


def balance_data(total_dataset, tag='train', num_classes=7, dataset_type='ham10000', seed=1):
    np.random.seed(int(seed))

    if tag == 'train':
        total_each_class = min(
            [len(total_dataset.train_data[total_dataset.train_clean_labels == i]) for i in range(num_classes)])
        data = total_dataset.train_data
        noisy_label = total_dataset.train_noisy_labels
        clean_label = total_dataset.train_clean_labels
    elif tag == 'val':
        total_each_class = min(
            [len(total_dataset.val_data[total_dataset.val_clean_labels == i]) for i in range(num_classes)])
        data = total_dataset.val_data
        noisy_label = total_dataset.val_noisy_labels
        clean_label = total_dataset.val_clean_labels
    elif tag == 'test':
        total_each_class = min(
            [len(total_dataset.test_data[total_dataset.test_labels == i]) for i in range(num_classes)])
        data = total_dataset.test_data
        noisy_label = total_dataset.test_labels
        clean_label = total_dataset.test_labels

    data_set = []
    noisy_label_set = []
    clean_label_set = []
    for i in range(num_classes):
        sub_data = data[clean_label == i]
        sub_clean_label = clean_label[clean_label == i]
        sub_noisy_label = noisy_label[clean_label == i]
        rand_index = np.random.choice(len(sub_data), size=int(total_each_class), replace=False).astype(int)
        sub2_data = sub_data[rand_index]
        sub2_clean_label = sub_clean_label[rand_index]
        sub2_noisy_label = sub_noisy_label[rand_index]
        data_set.extend(sub2_data)
        noisy_label_set.extend(sub2_noisy_label)
        clean_label_set.extend(sub2_clean_label)

    images = np.array(data_set)
    clean_labels = np.array(clean_label_set)
    noisy_labels = np.array(noisy_label_set)
    if dataset_type == 'ham10000':
        balanced_dataset = dataset.local_dataset(images,
                                                 noisy_labels,
                                                 clean_labels,
                                                 transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ]),
                                                 target_transform=transform_target
                                                 )
    
    return balanced_dataset

def wrap_as_local_dataset(client_dataset, tag='train', dataset_type='ham10000'):
    """
    기존 Dataset 객체를 local_dataset 형태로 변환합니다.
    """
    if tag == 'train':
        data = client_dataset.train_data
        noisy_labels = client_dataset.train_noisy_labels
        clean_labels = client_dataset.train_clean_labels
    elif tag == 'val':
        data = client_dataset.val_data
        noisy_labels = client_dataset.val_noisy_labels
        clean_labels = client_dataset.val_clean_labels
    elif tag == 'test':
        data = client_dataset.test_data
        noisy_labels = client_dataset.test_labels
        clean_labels = client_dataset.test_labels
    else:
        raise ValueError(f"Unknown tag: {tag}")

    return dataset.local_dataset(
        data,
        noisy_labels,
        clean_labels,
        transform=client_dataset.transform,
        target_transform=client_dataset.target_transform
    )

def create_data_balanced_with_prob(prob, total_dataset, num_classes=7, dataset_type='ham10000'):
    np.random.seed(1)
    data = total_dataset.local_data
    noisy_label = total_dataset.local_noisy_labels
    clean_label = total_dataset.local_clean_labels

    class_indices = {i: np.where(clean_label == i)[0] for i in range(num_classes)}
    class_counters = {i: 0 for i in range(num_classes)}

    min_per_client = int(min([len(class_indices[i]) for i in range(num_classes)]) / prob.shape[0])

    clients = []
    for m in range(prob.shape[0]):
        client_indices = []
        for c in range(num_classes):
            count = int(prob[m][c] * min_per_client * num_classes)
            indices = class_indices[c][class_counters[c]:class_counters[c] + count]
            class_counters[c] += count
            client_indices.extend(indices)

        np.random.shuffle(client_indices)
        client_data = data[client_indices]
        client_noisy = noisy_label[client_indices]
        client_clean = clean_label[client_indices]

        client_dataset = dataset.local_dataset(client_data, client_noisy, client_clean,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                                        (0.229, 0.224, 0.225)), ]),
                                               target_transform=transform_target)
        clients.append(client_dataset)

    return clients

def split_equally_across_clients(dataset, num_clients, seed=42):
    np.random.seed(seed)
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)

    split_sizes = [len(dataset) // num_clients] * num_clients
    for i in range(len(dataset) % num_clients):
        split_sizes[i] += 1

    split_indices = np.split(total_indices, np.cumsum(split_sizes)[:-1])
    client_datasets = [Subset(dataset, indices) for indices in split_indices]
    return client_datasets

def wrap_subsets_to_local_dataset(subsets, full_dataset):
    wrapped = []
    for subset in subsets:
        indices = subset.indices  # Subset에서 인덱스 추출
        images = full_dataset.local_data[indices]
        noisy_labels = full_dataset.local_noisy_labels[indices]
        clean_labels = full_dataset.local_clean_labels[indices]

        wrapped.append(dataset.local_dataset(
            images,
            noisy_labels,
            clean_labels,
            transform=full_dataset.transform,
            target_transform=full_dataset.target_transform
        ))
    return wrapped

def get_instance_noisy_label(n, total_dataset, labels, num_classes, feature_size, norm_std, seed):
    # n -> noise_rate
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed
    print("adding noise to dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (0.6 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(total_dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)


def data_split(data, clean_labels, noisy_labels, num_classes=10, split_percentage=0.9, seed=1):

    np.random.seed(int(seed))
    train_data_set = []
    train_clean_labels_set = []
    train_noisy_labels_set = []
    val_data_set = []
    val_clean_labels_set = []
    val_noisy_labels_set = []
    for i in range(num_classes):
        sub_data = data[clean_labels == i]
        sub_clean_label = clean_labels[clean_labels == i]
        sub_noisy_label = noisy_labels[clean_labels == i]
        num_per_classes = len(sub_data)
        index = np.arange(num_per_classes)
        train_rand_index = np.random.choice(num_per_classes, size=int(num_per_classes * split_percentage), replace=False).astype(int)
        val_rand_index = np.delete(index, train_rand_index)
        train_data, val_data = sub_data[train_rand_index, :], sub_data[val_rand_index, :]
        train_clean_labels, val_clean_labels = sub_clean_label[train_rand_index], sub_clean_label[val_rand_index]
        train_noisy_labels, val_noisy_labels = sub_noisy_label[train_rand_index], sub_noisy_label[val_rand_index]
        train_data_set.extend(train_data)
        val_data_set.extend(val_data)
        train_noisy_labels_set.extend(train_noisy_labels)
        val_noisy_labels_set.extend(val_noisy_labels)
        train_clean_labels_set.extend(train_clean_labels)
        val_clean_labels_set.extend(val_clean_labels)
    return np.array(train_data_set), np.array(val_data_set), np.array(train_noisy_labels_set), \
           np.array(val_noisy_labels_set), np.array(train_clean_labels_set), np.array(val_clean_labels_set)

if __name__=='__main__':
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000/train', transform=transform)
    labels = torch.tensor(dataset.targets)

    noisy_labels = get_instance_noisy_label(
    n=0.3,
    total_dataset=dataset,
    labels=labels,
    num_classes=7,
    feature_size=224*224*3,
    norm_std=0.1,
    seed=42
    )

    print(len(noisy_labels))