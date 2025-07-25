#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from sklearn import metrics

import copy
import torch
import numpy as np
import math
import random
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, attack_label=-1):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.attack_label = attack_label
        if self.attack_label >= 0:
            if self.attack_label != 0 and self.attack_label != 1 and self.attack_label != 3:
                print('currently attack label only supports 0 for loan dataset, 1 for mnist and 3 (Cat) for cifar, not {}'.format(
                    self.attack_label))
                exit(-1)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, _, _ = self.dataset[int(self.idxs[item])]
        if self.attack_label >= 0:
            if label == self.attack_label == 1:
                label = (label + 6) % 10
            elif label == self.attack_label == 3:   # cat to dog for cifar
                label = label + 2
            elif label == self.attack_label == 0:   # loan dataset
                label = 1

        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, tb, attack_label=-1, backdoor_label=-1, test_flag=False):
        self.args = args
        self.attack_label = attack_label
        self.backdoor_label = backdoor_label
        if self.attack_label >= 0 and self.backdoor_label >= 0: # if the args has something wrong
            self.attack_label = -1  # make sure that only perform one kind of attack
        self.backdoor_pixels = [[0, 0], [0, 1], [0, 2],
                                [0, 4], [0, 5], [0, 6],
                                [2, 0], [2, 1], [2, 2],
                                [2, 4], [2, 5], [2, 6],
                                ]  # temporarily hard code
        self.backdoor_loan_feat=[['pub_rec_bankruptcies',20,83] , # feature name; assigned poison value; feature index
                                 ['num_tl_120dpd_2m',10,77],
                                 ['acc_now_delinq',20,36],
                                 ['pub_rec',100,18],
                                 ['tax_liens',100,84],
                                 ['num_tl_90g_dpd_24m',80,79]
                                 ] # temporarily hard code
        # if args.model == 'mobilenet' or args.model == 'loannet':
        #     self.loss_func = nn.CrossEntropyLoss()
        # else:
        #     self.loss_func = nn.NLLLoss()
        self.loss_func = nn.CrossEntropyLoss()
        self.tb = tb
        self.dis_loss = nn.L1Loss()

        if test_flag: # test_flag -> idxs all for test; otherwise only 20% of idxs for test
            if backdoor_label >= 0: # backdoor attack will poison all data in training..
                self.ldr_test = DataLoader(DatasetSplit(dataset, idxs, -1), batch_size=args.local_bs,
                                           shuffle=True)
            else: # label-flipping attack
                self.ldr_test = DataLoader(DatasetSplit(dataset, idxs, self.attack_label), batch_size=args.local_bs,
                                                   shuffle=True)
            self.ldr_train=None
            self.ldr_val=None
        else:
            # self.ldr_train, self.ldr_val, self.ldr_test = self.train_val_test(dataset, list(idxs))
            self.ldr_train, self.ldr_test = dataset

    def train_val_test(self, dataset, idxs):
        # split train, validation, and test
        train_split = round(len(idxs) * 0.7)
        test_split = round(len(idxs) * 0.8)
        idxs_train = idxs[:int(train_split)]
        idxs_val = idxs[int(train_split):int(test_split)]
        idxs_test = idxs[int(test_split):]
        if self.backdoor_label >= 0:  # backdoor attack will poison part of data per batch in training..
            train = DataLoader(DatasetSplit(dataset, idxs_train, -1), batch_size=self.args.local_bs,
                               shuffle=True)
            val = DataLoader(DatasetSplit(dataset, idxs_val, -1), batch_size=int(len(idxs_val) / 10),
                             shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test, -1), batch_size=int(len(idxs_test) / 10),
                              shuffle=True)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs_train, self.attack_label), batch_size=self.args.local_bs,
                               shuffle=True)
            val = DataLoader(DatasetSplit(dataset, idxs_val, self.attack_label), batch_size=int(len(idxs_val) / 10),
                             shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test, self.attack_label), batch_size=int(len(idxs_test) / 10),
                              shuffle=True)
        return train, val, test

    def update_weights(self, net):
        global_net = dict()
        if self.backdoor_label >= 0:  # backdoor attack can scale, so it needs the original value of parameters
            for name, data in net.state_dict().items():
                global_net[name] = net.state_dict()[name].clone()
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _, _) in enumerate(self.ldr_train):
                # if any(labels == 1):
                #     print(labels)
                #     exit(-1)
                if batch_idx > self.args.local_iter > 0:
                    break
                if self.backdoor_label >= 0:  # backdoor attack
                    images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                         self.args.backdoor_per_batch,
                                                                         self.backdoor_label,
                                                                         evaluation=False)
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                if self.args.dataset== 'loan':
                    images=images.float()
                    labels=labels.long()

                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.backdoor_label >= 0:  # backdoor attack can scale
            for name, data in net.state_dict().items():
                new_value = global_net[name] + (data - global_net[name]) * self.args.backdoor_scale_factor
                net.state_dict()[name].copy_(new_value)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), net

    def update_gradients(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        client_grad = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _, _) in enumerate(self.ldr_train):
                if batch_idx > self.args.local_iter > 0:
                    break
                if self.backdoor_label >= 0:  # backdoor attack
                    images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                         self.args.backdoor_per_batch,
                                                                         self.backdoor_label,
                                                                         evaluation=False)
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                if self.args.dataset == 'loan':
                    images = images.float()
                    labels = labels.long()
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                # get gradients
                for i, (name, params) in enumerate(net.named_parameters()):
                    if params.requires_grad:
                        if iter == 0 and batch_idx == 0:
                            client_grad.append(params.grad.clone())
                        else:
                            client_grad[i] += params.grad.clone()

                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # there is no scale for backdoor attack when update gradients, because
        # foolsgold paper said it doesn't consider scaling malicious updates to overpower honest clients

        if self.backdoor_label >= 0:  # backdoor attack can scale
            for i, (name, params) in enumerate(net.named_parameters()):
                if params.requires_grad:
                    client_grad[i]*= self.args.backdoor_scale_factor
        return client_grad, sum(epoch_loss) / len(epoch_loss),net

    def add_dis_to_gradient(self, net, w_glob, alpha=0.9):
        total_dis = 0
        keys = list(w_glob.keys())
        for i, p in enumerate(net.parameters()):
            key = keys[i]
            dis = p - w_glob[key]
            p.grad = p.grad * alpha + dis * (1 - alpha) * 100
            total_dis += dis.abs().sum()
        return total_dis

    def update_weights_with_constrain(self, net, w_glob):
        global_net = dict()
        if self.backdoor_label >= 0:
            for name, data in net.state_dict().items():
                global_net[name] = net.state_dict()[name].clone()

        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _, _) in enumerate(self.ldr_train):
                if batch_idx > self.args.local_iter > 0:
                    break
                if self.backdoor_label >= 0:  # backdoor attack
                    images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                         self.args.backdoor_per_batch,
                                                                         self.backdoor_label,
                                                                         evaluation=False)
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                if self.args.dataset== 'loan':
                    images=images.float()
                    labels=labels.long()
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                # after_weight = net.state_dict()
                # dis = []
                # for k in w_glob.keys():
                #     dis.append((after_weight[k] - w_glob[k]).flatten())
                # dis = torch.cat(dis, dim=0)
                # param_size = len(dis)
                # d_loss = dis.abs().sum() / param_size
                before_loss = loss.item()
                # loss = loss * alpha + (1.0 - alpha ) * d_loss
                # print('Added model distance loss: {:6f}, train loss {:6f}'.format(d_loss, before_loss))

                loss.backward(retain_graph=True)
                total_dis = self.add_dis_to_gradient(net, w_glob)
                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDistance: {:6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), before_loss, total_dis))
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.backdoor_label >= 0:
            for name, data in net.state_dict().items():
                new_value = global_net[name] + (data - global_net[name]) * self.args.backdoor_scale_factor
                net.state_dict()[name].copy_(new_value)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test(self, net):
        acc = 0.0
        total_len = 0
        for batch_idx, (images, labels, _, _) in enumerate(self.ldr_test):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            if self.args.dataset == 'loan':
                images = images.float()
                labels = labels.long()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            y_pred = np.argmax(log_probs.data, axis=1)
            acc += metrics.accuracy_score(y_true=labels.data, y_pred=y_pred) * len(labels)
            total_len += len(labels)
        acc /= float(total_len)
        return acc, loss.item()

    def backdoor_test(self, net):
        if self.backdoor_label < 0:
            return None, None
        acc = 0.0
        total_len = 0
        for batch_idx, (images, labels, _, _) in enumerate(self.ldr_test):
            images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                 self.args.backdoor_per_batch, self.backdoor_label,
                                                                 evaluation=True)
            if self.args.dataset == 'loan':
                images = images.float()
                labels = labels.long()
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            y_pred = np.argmax(log_probs.data, axis=1)
            acc += metrics.accuracy_score(y_true=labels.data, y_pred=y_pred) * len(labels)
            total_len += len(labels)
        acc /= float(total_len)
        return acc, loss.item()

    def get_probs(self, net):
        all_probs = 0
        with torch.no_grad():
            for batch_idx, (images, labels, _, _) in enumerate(self.ldr_test):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                log_probs = net(images)
                probs = torch.exp(log_probs)
                all_probs += probs.gather(1, labels.view(-1, 1)).sum()
        return all_probs

    def get_poison_batch(self, images, targets, backdoor_per_batch, backdoor_label, evaluation=False):
        poison_count = 0
        new_images = torch.empty(images.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(images)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = backdoor_label
                new_images[index] = self.add_backdoor_pixels(images[index])
                poison_count += 1

            else:  # will poison backdoor_per_batch data when training
                if index < backdoor_per_batch:
                    new_targets[index] = backdoor_label
                    new_images[index] = self.add_backdoor_pixels(images[index])
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

    def add_backdoor_pixels(self, image):
        if self.args.dataset == 'loan':
            for i in range(0, len(self.backdoor_loan_feat)):
                name=self.backdoor_loan_feat[i][0]
                value=self.backdoor_loan_feat[i][1]
                index = self.backdoor_loan_feat[i][2]
                image[index]=value
        else:
            if image.shape[0] == 3:
                for i in range(0, len(self.backdoor_pixels)):
                    pos = self.backdoor_pixels[i]
                    image[0][pos[0]][pos[1]] = 1
                    image[1][pos[0]][pos[1]] = 1
                    image[2][pos[0]][pos[1]] = 1
            elif image.shape[0] == 1:
                for i in range(0, len(self.backdoor_pixels)):
                    pos = self.backdoor_pixels[i]
                    image[0][pos[0]][pos[1]] = 1
        return image
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import math
import random
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp


eps = np.finfo(float).eps


def aggregate_weights(args, w_locals, net_glob, reweights, fg):
    # update global weights
    # choices are ['average', 'median', 'trimmed_mean',
    #              'repeated', 'irls', 'simple_irls',
    #              'irls_median', 'irls_theilsen',
    #              'irls_gaussian', 'fg']
    if args.agg == 'median':
        print("using simple median Estimator")
        w_glob = simple_median(w_locals)
    elif args.agg == 'trimmed_mean':
        print("using trimmed mean Estimator")
        w_glob = trimmed_mean(w_locals, args.alpha)
    elif args.agg == 'repeated':
        print("using repeated median Estimator")
        w_glob = Repeated_Median_Shard(w_locals)
    elif args.agg == 'irls':
        print("using IRLS Estimator")
        w_glob, reweight = IRLS_aggregation_split_restricted(w_locals, args.Lambda, args.thresh)
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'simple_irls':
        print("using simple IRLS Estimator")
        w_glob, reweight = simple_IRLS(w_locals, args.Lambda, args.thresh, args.alpha)
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'irls_median':
        print("using median IRLS Estimator")
        w_glob, reweight = IRLS_other_split_restricted(w_locals, args.Lambda, args.thresh, mode='median')
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'irls_theilsen':
        print("using TheilSen IRLS Estimator")
        w_glob, reweight = IRLS_other_split_restricted(w_locals, args.Lambda, args.thresh, mode='theilsen')
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'irls_gaussian':
        print("using Gaussian IRLS Estimator")
        w_glob, reweight = IRLS_other_split_restricted(w_locals, args.Lambda, args.thresh, mode='gaussian')
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'fg':
        # Update model
        # Add all the gradients to the model gradient
        net_glob.train()
        # train and update
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer.zero_grad()
        agg_grads = fg.aggregate_gradients(w_locals)
        for i, (name, params) in enumerate(net_glob.named_parameters()):
            if params.requires_grad:
                params.grad = agg_grads[i].cuda()
        optimizer.step()
    elif args.agg == 'average':
        print("using average")
        w_glob = average_weights(w_locals)
    else:
        exit('Error: unrecognized aggregation method')
    return w_glob


def average_weights(w):
    cur_time = time.time()
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    print('model aggregation "average" took {}s'.format(time.time() - cur_time))
    return w_avg


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output


def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights


def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def gaussian_reweight_algorithm_restricted(y, sig, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)

    reweight = gaussian_zero_mean(e, sig=sig)
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def theilsen_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = theilsen(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def median_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    y_median = median_opt(y).unsqueeze(1).repeat(1, num_models)
    residual = y - y_median
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + y_median * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def simple_reweight(y, LAMBDA, thresh, alpha):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    # e = 1 / (residual.abs() + eps)
    # e_max = e.max(dim=-1)[0].unsqueeze(1).repeat(1, num_models)
    # reweight = e / e_max
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = 1 / e * torch.max(-K, torch.min(K, e))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std

    # sorted idx (remove alpha)
    sort_ids = e.abs().sort()[1].sort()[1]
    # remove_ids = sort_ids >= int((1 - alpha) * num_models)
    remove_ids = [i for i in sort_ids if i.item() >= int((1 - alpha) * num_models)]
    remove_ids = remove_ids * (reweight < thresh)
    keep_ids = (1 - remove_ids).type(torch.cuda.FloatTensor)
    remove_ids = remove_ids.type(torch.cuda.FloatTensor)
    restricted_y = y * keep_ids + line_y * remove_ids
    reweight_regulized = reweight_regulized * keep_ids
    return reweight_regulized, restricted_y


def is_valid_model(w):
    if isinstance(w, list):
        w_keys = list(range(len(w)))
    else:
        w_keys = w.keys()
    for k in w_keys:
        params = w[k]
        if torch.isnan(params).any():
            return False
        if torch.isinf(params).any():
            return False
    return True


def get_valid_models(w_locals, m_locals=None):
    w, invalid_model_idx = [], []
    if m_locals is not None:
        w, invalid_model_idx, m = [], [], []
    for i in range(len(w_locals)):
        if is_valid_model(w_locals[i]):
            w.append(w_locals[i])
            if m_locals is not None:
                m.append(m_locals[i])
        else:
            invalid_model_idx.append(i)
    if m_locals is not None:
        return w, invalid_model_idx, m
    return w, invalid_model_idx


def IRLS_aggregation_split_restricted(w_locals, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000
    cur_time = time.time()
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    reweight = (reweight / reweight.max()).to(torch.device("cpu"))
    weights = torch.zeros(len(w_locals))
    i = 0
    for j in range(len(w_locals)):
        if j not in invalid_model_idx:
            weights[j] = reweight[i]
            i += 1

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, weights


def IRLS_other_split_restricted(w_locals, LAMBDA=2, thresh=0.1, mode='median'):
    if mode == 'median':
        reweight_algorithm = median_reweight_algorithm_restricted
    elif mode == 'theilsen':
        reweight_algorithm = theilsen_reweight_algorithm_restricted
    elif mode == 'gaussian':
        reweight_algorithm = gaussian_reweight_algorithm_restricted     # in gaussian reweight algorithm, lambda is sigma

    SHARD_SIZE = 2000
    cur_time = time.time()
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm(transposed_y_list, LAMBDA, thresh)
            print(reweight.sum(dim=0))
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm(y, LAMBDA, thresh)
                print(reweight.sum(dim=0))
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    reweight = (reweight / reweight.max()).to(torch.device("cpu"))
    weights = torch.zeros(len(w_locals))
    i = 0
    for j in range(len(w_locals)):
        if j not in invalid_model_idx:
            weights[j] = reweight[i]
            i += 1

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, weights


def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med


def simple_IRLS(w, LAMBDA=2, thresh=0.03, alpha=1 / 11.0):
    SHARD_SIZE = 50000
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = simple_reweight(transposed_y_list, LAMBDA, thresh, alpha)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = simple_reweight(y, LAMBDA, thresh, alpha)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum  )
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, (reweight / reweight.max()).to(torch.device("cpu"))


def random_select(size, thresh=0.5):
    assert thresh < 1.0
    a = []
    while len(a) < 3:
        for i in range(size):
            if random.uniform(0, 1) > thresh:
                a.append(i)
    return a


def theilsen(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yy = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyj = yy
    yyi = yyj.transpose(-1, -2)
    xx = torch.cuda.FloatTensor(range(num_models))
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.cuda.FloatTensor([float('Inf')] * num_models)
    inf_lower = torch.tril(diag.repeat(num_models, 1), diagonal=0).repeat(total_num, 1, 1)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + inf_lower
    slopes, _ = torch.flatten(slopes, 1, 2).sort()
    raw_slopes = slopes[:, :int(num_models * (num_models - 1) / 2)]
    slopes = median_opt(raw_slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.cuda.FloatTensor(xx_median)
    intercepts = yy_median - slopes * xx_median
    return slopes, intercepts


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med


# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold(object):
    def __init__(self, args):
        self.memory = None
        self.wv_history = []
        self.args = args

    def aggregate_gradients(self, client_grads):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((num_clients, grad_len))

        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))


        if self.args.use_memory:
            self.memory += grads
            wv = foolsgold(self.memory)  # Use FG
        else:
            wv = foolsgold(grads)  # Use FG
        print(wv)
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
                len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads


# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    cur_time = time.time()
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    print('model aggregation "median" took {}s'.format(time.time() - cur_time))
    return w_med


def trimmed_mean(w, trim_ratio):
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    cur_time = time.time()
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    print('model aggregation "trimmed mean" took {}s'.format(time.time() - cur_time))
    return w_med


def gaussian_zero_mean(x, sig=1):
    return torch.exp(- x * x / (2 * sig * sig))


if __name__ == "__main__":
    # from matplotlib import pyplot as mp
    #
    # x_values = np.linspace(-3, 3, 120)
    # for mu, sig in [(0, 1)]:
    #     mp.plot(x_values, gaussian(x_values, mu, sig))
    #
    # mp.show()

    torch.manual_seed(0)
    y = torch.ones(1, 10).cuda()
    e = gaussian_reweight_algorithm_restricted(y, 2, thresh=0.1)
    print(y)
    print(e)