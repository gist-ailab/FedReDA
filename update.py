import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn, autograd
from sklearn.metrics import balanced_accuracy_score
from geomloss import SamplesLoss

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

def get_local_update_objects(args, dataset_train, dict_users=None, net_glob=None):
    local_update_objects = []
    for idx in range(args.num_clients):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )
        local_update_objects.append(LocalUpdateRFL(**local_update_args))

    return local_update_objects

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
            
    return w_avg

class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label, _, _ = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


# Train the Model
def train_one_step(net, data, label, optimizer, criterion, class_p_list=None):
    net.train()
    pred = net(data)
    if class_p_list is not None:
        pred = pred + 0.5*class_p_list
    loss = criterion(pred, label).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


def train(train_loader, epoch, model, optimizer1, criterion, args, class_p_list=None):
    model.train()
    train_total = 0
    train_correct = 0

    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = noisy_label.cuda()
        prec, loss = train_one_step(model, data, labels, optimizer1, criterion, class_p_list)
        train_total += 1
        train_correct += prec

        if (i+1)==250:
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                    % (epoch + 1, i + 1, 4000 // args.batch_size, prec, loss.item()))

    train_acc = float(train_correct) / float(train_total)

    return train_acc

def train_prox(train_loader, epoch, model, global_model, optimizer1, mu, args, class_p_list=None):
    # print('Training %s...' % model_str)
    model.train()
    train_total = 0
    train_correct = 0

    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = noisy_label.cuda()
        prec, loss = train_prox_one_step(model, global_model, data, labels, optimizer1, nn.CrossEntropyLoss(), mu, class_p_list)
        train_total += 1
        train_correct += prec

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, i + 1, 4000 // args.batch_size, prec, loss.item()))

    train_acc = float(train_correct) / float(train_total)

    return train_acc

def train_prox_one_step(net, global_model, data, label, optimizer, criterion, mu, class_p_list=None):
    net.train()
    pred = net(data)
    if class_p_list is not None:
        pred = pred + 0.5*class_p_list
    # compute proximal_term
    proximal_term = 0.0
    for w, w_t in zip(net.parameters(), global_model.parameters()):
        proximal_term += (w - w_t).norm(2)
    loss = criterion(pred, label) + (mu / 2) * proximal_term
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


# Evaluate the Model
def evaluate(val_loader, model1):
    # print('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                data, noisy_label, clean_label, _ = batch
            else:
                data, clean_label = batch
                noisy_label = clean_label # placeholder
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += noisy_label.size(0)
            
            all_preds.extend(pred1.cpu().numpy())
            all_targets.extend(clean_label.cpu().numpy())
            
        #     correct1 += (pred1.cpu() == clean_label.long()).sum()
        # acc1 = 100 * float(correct1) / float(total1)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)

    return balanced_acc*100


def train_forward(model, train_loader, optimizer, model_trans):
    model.train()
    train_total = 0
    train_correct = 0
    for i, (data, labels, _, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        logits = model(data)
        original_post = F.softmax(logits, dim=1)
        T = model_trans(data)
        noisy_post = torch.bmm(original_post.unsqueeze(1), T.cuda()).squeeze(1)
        log_noisy_post = torch.log(noisy_post + 1e-12)
        loss = nn.NLLLoss()(log_noisy_post.cuda(), labels.cuda())

        prec1, = accuracy(noisy_post, labels, topk=(1,))
        train_total += 1
        train_correct += prec1
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc

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

#########################################################################
def compute_transport_matrix(W1, W2, epsilon=0.01):
    """
    W1: [out_dim, in_dim] - global adapter weight
    W2: [out_dim, in_dim] - client adapter weight
    Returns:
        T: [out_dim, out_dim] transport matrix aligning W2 to W1
    """
    out_dim = W1.size(0)

    # Compute pairwise cosine distance as cost matrix
    cost_matrix = 1 - F.cosine_similarity(W1.unsqueeze(1), W2.unsqueeze(0), dim=-1)  # shape: [out_dim, out_dim]

    # OT using Sinkhorn distance
    # Assume uniform marginals
    a = torch.ones(out_dim, device=W1.device) / out_dim
    b = torch.ones(out_dim, device=W2.device) / out_dim

    # Use geomloss Sinkhorn
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon)
    loss = sinkhorn(a, W1, b, W2)  # returns scalar but internally computes T

    # Geomloss doesn't return T directly, so alternatively, use custom Sinkhorn if explicit T is needed
    # For now, as an example, we'll just use the cost matrix to form T via softmin:
    T = F.softmax(-cost_matrix / epsilon, dim=-1)  # Soft alignment approximation

    return T

def average_reins(global_model, client_models):
    # 1. Average entire reins
    with torch.no_grad():
        # 모든 client의 reins파라미터 가져오기
        reins_named_params = {}
        for name, _ in client_models[0].reins.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_models])
            reins_named_params[name] = stacked.mean(dim=0)

        # global_model.reins에 복사
        for name, param in global_model.reins.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])
                
        reins_named_params2 = {}
        for name, _ in client_models[0].reins2.named_parameters():
            stacked = torch.stack([dict(client.reins2.named_parameters())[name].data for client in client_models])
            reins_named_params2[name] = stacked.mean(dim=0)

        # global_model.reins에 복사
        for name, param in global_model.reins2.named_parameters():
            if name in reins_named_params2:
                param.data.copy_(reins_named_params2[name])

    # 2. Average linear_rein weights and biases
    weight_sum = sum(client.linear_rein.weight.data for client in client_models)
    bias_sum = sum(client.linear_rein.bias.data for client in client_models)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_models))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_models))
    
    weight_sum = sum(client.linear_rein2.weight.data for client in client_models)
    bias_sum = sum(client.linear_rein2.bias.data for client in client_models)
    global_model.linear_rein2.weight.data.copy_(weight_sum / len(client_models))
    global_model.linear_rein2.bias.data.copy_(bias_sum / len(client_models))

    # 3. Broadcast updated global reins → 각 client.reins에 전달
    for client in client_models:
        client.reins.load_state_dict(global_model.reins.state_dict())
        client.reins2.load_state_dict(global_model.reins2.state_dict())
        client.linear_rein.load_state_dict(global_model.linear_rein.state_dict())
        client.linear_rein2.load_state_dict(global_model.linear_rein2.state_dict())
        client.reins.train()

def PAPA_average_reins(global_model, client_models, alpha_papa=0.9):
    with torch.no_grad():
        # 1. Average reins across clients → global model 업데이트
        reins_named_params = {}
        for name, _ in client_models[0].reins.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_models])
            reins_named_params[name] = stacked.mean(dim=0)

        for name, param in global_model.reins.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])

        reins_named_params2 = {}
        for name, _ in client_models[0].reins2.named_parameters():
            stacked = torch.stack([dict(client.reins2.named_parameters())[name].data for client in client_models])
            reins_named_params2[name] = stacked.mean(dim=0)

        for name, param in global_model.reins2.named_parameters():
            if name in reins_named_params2:
                param.data.copy_(reins_named_params2[name])

        # 2. Average linear layers
        def average_linear(client_params):
            return sum(client_param.data for client_param in client_params) / len(client_params)

        global_model.linear_rein.weight.data.copy_(average_linear([client.linear_rein.weight for client in client_models]))
        global_model.linear_rein.bias.data.copy_(average_linear([client.linear_rein.bias for client in client_models]))

        global_model.linear_rein2.weight.data.copy_(average_linear([client.linear_rein2.weight for client in client_models]))
        global_model.linear_rein2.bias.data.copy_(average_linear([client.linear_rein2.bias for client in client_models]))

        global_model.linear_norein.weight.data.copy_(average_linear([client.linear_norein.weight for client in client_models]))
        global_model.linear_norein.bias.data.copy_(average_linear([client.linear_norein.bias for client in client_models]))

    # 3. Broadcast global → client with EMA update
    for client in client_models:
        for name, param in client.reins.named_parameters():
            global_param = dict(global_model.reins.named_parameters())[name]
            param.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_param.data)

        for name, param in client.reins2.named_parameters():
            global_param = dict(global_model.reins2.named_parameters())[name]
            param.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_param.data)

        client.linear_rein.weight.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_model.linear_rein.weight.data)
        client.linear_rein.bias.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_model.linear_rein.bias.data)

        client.linear_rein2.weight.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_model.linear_rein2.weight.data)
        client.linear_rein2.bias.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_model.linear_rein2.bias.data)

        client.linear_norein.weight.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_model.linear_norein.weight.data)
        client.linear_norein.bias.data.mul_(alpha_papa).add_((1 - alpha_papa) * global_model.linear_norein.bias.data)

        client.train()

def average_reins_with_transport(global_model, client_models):
    with torch.no_grad():
        # 1. Align and average reins
        for name, param in global_model.reins.named_parameters():
            aligned_params = []
            for client in client_models:
                client_param = dict(client.reins.named_parameters())[name].data
                global_param = param.data

                if len(client_param.shape) == 2:  # Only align weights of shape [out_dim, in_dim]
                    T = compute_transport_matrix(global_param, client_param)
                    aligned_client_param = torch.matmul(T, client_param)
                else:
                    aligned_client_param = client_param  # e.g., biases are just copied

                aligned_params.append(aligned_client_param)

            param.data.copy_(torch.stack(aligned_params).mean(dim=0))

        # 동일하게 reins2
        for name, param in global_model.reins2.named_parameters():
            aligned_params = []
            for client in client_models:
                client_param = dict(client.reins2.named_parameters())[name].data
                global_param = param.data

                if len(client_param.shape) == 2:
                    T = compute_transport_matrix(global_param, client_param)
                    aligned_client_param = torch.matmul(T, client_param)
                else:
                    aligned_client_param = client_param

                aligned_params.append(aligned_client_param)

            param.data.copy_(torch.stack(aligned_params).mean(dim=0))

        # 2. Average linear layers
        for layer_name in ['linear_rein', 'linear_rein2', 'linear_norein']:
            weight_sum = sum(getattr(client, layer_name).weight.data for client in client_models)
            bias_sum = sum(getattr(client, layer_name).bias.data for client in client_models)

            getattr(global_model, layer_name).weight.data.copy_(weight_sum / len(client_models))
            getattr(global_model, layer_name).bias.data.copy_(bias_sum / len(client_models))

        # 3. Broadcast
        for client in client_models:
            client.reins.load_state_dict(global_model.reins.state_dict())
            client.reins2.load_state_dict(global_model.reins2.state_dict())
            client.linear_rein.load_state_dict(global_model.linear_rein.state_dict())
            client.linear_rein2.load_state_dict(global_model.linear_rein2.state_dict())
            client.linear_norein.load_state_dict(global_model.linear_norein.state_dict())

            client.reins.train()

def average_reins1(global_model, client_models):
    # 1. Average entire reins
    with torch.no_grad():
        # 모든 client의 reins파라미터 가져오기
        reins_named_params = {}
        for name, _ in client_models[0].reins.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_models])
            reins_named_params[name] = stacked.mean(dim=0)

        # global_model.reins에 복사
        for name, param in global_model.reins.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])

    # 2. Average linear_rein weights and biases
    weight_sum = sum(client.linear_rein.weight.data for client in client_models)
    bias_sum = sum(client.linear_rein.bias.data for client in client_models)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_models))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_models))

    # 3. Broadcast updated global reins → 각 client.reins에 전달
    for client in client_models:
        client.reins.load_state_dict(global_model.reins.state_dict())
        client.linear_rein.load_state_dict(global_model.linear_rein.state_dict())
        client.reins.train()

def average_reins2(global_model, client_models):
    # 1. Average entire reins1 (not just learnable_tokens)
    with torch.no_grad():
        # 모든 client의 reins1 파라미터 가져오기
        reins_named_params = {}
        for name, _ in client_models[0].reins2.named_parameters():
            stacked = torch.stack([dict(client.reins2.named_parameters())[name].data for client in client_models])
            reins_named_params[name] = stacked.mean(dim=0)

        # global_model.reins에 복사
        for name, param in global_model.reins.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])

    # 2. Average linear_rein weights and biases
    weight_sum = sum(client.linear_rein.weight.data for client in client_models)
    bias_sum = sum(client.linear_rein.bias.data for client in client_models)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_models))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_models))

    # 3. Broadcast updated global reins → 각 client.reins2에 전달
    for client in client_models:
        client.reins2.load_state_dict(global_model.reins.state_dict())
        client.reins2.train()
        
def average_reins2_ema(global_model, client_models, alpha=0.95):
    """
    Update global_model using EMA of client models.
    alpha: decay factor for EMA. Higher = more weight on previous global_model.
    """
    with torch.no_grad():
        # EMA for reins2 (adapter parameters)
        for name, param in global_model.reins.named_parameters():
            if name in dict(client_models[0].reins2.named_parameters()):
                client_param_stack = torch.stack([
                    dict(client.reins2.named_parameters())[name].data
                    for client in client_models
                ])
                mean_param = client_param_stack.mean(dim=0)
                # EMA update
                param.data.mul_(alpha).add_((1 - alpha) * mean_param)

        # EMA for linear_rein layer
        weight_avg = sum(client.linear_rein.weight.data for client in client_models) / len(client_models)
        bias_avg = sum(client.linear_rein.bias.data for client in client_models) / len(client_models)

        global_model.linear_rein.weight.data.mul_(alpha).add_((1 - alpha) * weight_avg)
        global_model.linear_rein.bias.data.mul_(alpha).add_((1 - alpha) * bias_avg)
        
        # EMA for linear_rein layer
        weight_avg = sum(client.linear_norein.weight.data for client in client_models) / len(client_models)
        bias_avg = sum(client.linear_norein.bias.data for client in client_models) / len(client_models)

@torch.inference_mode()
def aggregate_reins_refdist(
    global_model,
    client_models,
    ref="fedavg",           # 'fedavg' | 'prev'
    prev_global=None,       # ref='prev'일 때 기준 모델
    metric="l2",            # 'l2' | 'cos'
    lambda_=1.0,            # 거리 가중 온도
    norm="mad",             # 'mad' | 'max'
    eps=1e-12,
):
    """
    DINOv2 + Rein 어댑터용 거리 기반 가중 집계.
    - 대상: model.reins, model.reins2, linear_rein, linear_rein2
    - weights_i ∝ exp(-lambda * D_i),  D_i = dist(client_i, ref)
    """

    K = len(client_models)
    if ref == "prev" and prev_global is None:
        raise ValueError("prev_global is required when ref='prev'.")

    # --- helpers ---
    def _named_vec(model):
        """rein 관련 파라미터를 단일 벡터로 평탄화(CPU float32)."""
        parts = []
        for mod in (model.reins, model.reins2):
            for _, p in mod.named_parameters():
                parts.append(p.detach().float().cpu().view(-1))
        parts.append(model.linear_rein.weight.detach().float().cpu().view(-1))
        parts.append(model.linear_rein.bias.detach().float().cpu().view(-1))
        parts.append(model.linear_rein2.weight.detach().float().cpu().view(-1))
        parts.append(model.linear_rein2.bias.detach().float().cpu().view(-1))
        return torch.cat(parts)

    def _dist(a, b):
        if metric == "l2":
            return torch.norm(a - b, p=2).item()
        elif metric == "cos":
            na = torch.norm(a, p=2) + eps
            nb = torch.norm(b, p=2) + eps
            cos = torch.dot(a, b) / (na * nb)
            return float(1.0 - cos.clamp(-1, 1).item())
        else:
            raise ValueError("metric must be 'l2' or 'cos'.")

    def _norm_scale(d):
        d = np.asarray(d, dtype=np.float64)
        if norm == "mad":
            med = np.median(d)
            mad = np.median(np.abs(d - med)) + eps
            D = np.abs(d - med) / mad
        elif norm == "max":
            D = d / (d.max() + eps)
        else:
            raise ValueError("norm must be 'mad' or 'max'.")
        return D

    def _weighted_avg_named(module_name):
        """모듈 이름('reins'|'reins2')의 파라미터를 가중 평균."""
        ref_client = client_models[0]
        ref_named = dict(getattr(ref_client, module_name).named_parameters())
        for name in ref_named.keys():
            stacked = torch.stack([
                dict(getattr(cm, module_name).named_parameters())[name].data.detach().float().cpu()
                for cm in client_models
            ], dim=0)  # [K, ...]
            wv = torch.from_numpy(weights).float().view(K, *([1] * (stacked.dim() - 1)))
            avg = (stacked * wv).sum(dim=0)
            # copy back to global (device/dtype 보존)
            gp = dict(getattr(global_model, module_name).named_parameters())[name]
            avg = avg.to(device=gp.device, dtype=gp.dtype)
            gp.copy_(avg)

    def _weighted_avg_linear(attr_name):
        """선형 레이어(linear_rein, linear_rein2) 가중 평균."""
        # weight
        stacked_w = torch.stack([getattr(cm, attr_name).weight.data.detach().float().cpu()
                                 for cm in client_models], dim=0)
        wv = torch.from_numpy(weights).float().view(K, *([1] * (stacked_w.dim() - 1)))
        avg_w = (stacked_w * wv).sum(dim=0)
        # bias
        stacked_b = torch.stack([getattr(cm, attr_name).bias.data.detach().float().cpu()
                                 for cm in client_models], dim=0)
        wv_b = torch.from_numpy(weights).float().view(K, *([1] * (stacked_b.dim() - 1)))
        avg_b = (stacked_b * wv_b).sum(dim=0)
        # copy back
        gm_layer = getattr(global_model, attr_name)
        gm_layer.weight.data.copy_(avg_w.to(device=gm_layer.weight.device, dtype=gm_layer.weight.dtype))
        gm_layer.bias.data.copy_(avg_b.to(device=gm_layer.bias.device, dtype=gm_layer.bias.dtype))

    # --- 1) reference vector ---
    if ref == "prev":
        ref_vec = _named_vec(prev_global)
    else:  # 'fedavg' 임시 평균로 ref 구성
        # 파라미터별 평균을 만들기 보다는 벡터 평균이 간단
        vecs = [_named_vec(cm) for cm in client_models]
        ref_vec = torch.stack(vecs, dim=0).mean(dim=0)

    # --- 2) distances & weights ---
    vecs = [_named_vec(cm) for cm in client_models]
    dists = [ _dist(v, ref_vec) for v in vecs ]

    # 모든 거리가 0이면 균등 가중(FedAvg와 동일)
    if max(dists) == 0.0:
        weights = np.ones(K, dtype=np.float64) / K
    else:
        D = _norm_scale(dists)
        weights = np.exp(-lambda_ * D)
        weights = weights / (weights.sum() + eps)

    # --- 3) weighted aggregation into global_model ---
    _weighted_avg_named("reins")
    _weighted_avg_named("reins2")
    _weighted_avg_linear("linear_rein")
    _weighted_avg_linear("linear_rein2")

    # --- 4) broadcast back to clients ---
    for cm in client_models:
        cm.reins.load_state_dict(global_model.reins.state_dict())
        cm.reins2.load_state_dict(global_model.reins2.state_dict())
        cm.linear_rein.load_state_dict(global_model.linear_rein.state_dict())
        cm.linear_rein2.load_state_dict(global_model.linear_rein2.state_dict())
        cm.reins.train()
        cm.reins2.train()

def client_update_with_refinement(client_model, global_model, optimizers, loader, config, mix_ratio, class_total, class_p_list=None, dynamic_boost=None):
    num_classes = config['num_classes']
    device = config['device']

    client_model.eval()
    client_model.train()

    optimizer_rein, optimizer_rein2 = optimizers

    if isinstance(class_total, dict):
        class_total_tensor = torch.tensor(
            [class_total.get(i, 0) for i in range(num_classes)],
            dtype=torch.float,
            device=device
        )
    else:
        class_total_tensor = torch.tensor(class_total).to(device).float()
    class_weights = F.softmax(class_total_tensor, dim=0)
    class_adjustment = 1.0 - class_weights

    ce_norein_vals, ce_vals, clean_ratios = [], [], []
    correct1, total1 = 0, 0
    class_correct_norein = defaultdict(int)
    class_correct_rein = defaultdict(int)
    class_correct = defaultdict(int)
    class_noise_correct_norein = defaultdict(int)
    class_noise_correct_rein = defaultdict(int)
    class_noise_correct = defaultdict(int)
    
    class_total = defaultdict(int)
    class_noise_correct = defaultdict(int)
    class_noise_total = defaultdict(int)

    # === Clean detection evaluation metrics ===
    detected_clean_total = 0
    true_clean_total = 0
    correct_detected_clean_total = 0
    false_detected_clean_total = 0
    missed_clean_total = 0
    
    for inputs, targets, clean_targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        clean_targets = clean_targets.to(device)

        # === Guide model ===
        with torch.no_grad():
            feats_global = global_model.forward_features(inputs)[:, 0, :]
            logits_global = global_model.linear_rein(feats_global)
            pred_global = logits_global.argmax(dim=1)
            
            # feats_global = global_model.forward_features2(inputs)[:, 0, :]
            # logits_global = global_model.linear_rein2(feats_global)
            # pred_global = logits_global.argmax(dim=1)

        # === Client model ===
        feats_rein = client_model.forward_features(inputs)[:, 0, :]
        logits_rein = client_model.linear_rein(feats_rein)
        pred_rein = logits_rein.argmax(dim=1)
        
        feats_rein2 = client_model.forward_features2(inputs)[:, 0, :]
        logits_rein2 = client_model.linear_rein2(feats_rein2)
        pred_rein2 = logits_rein2.argmax(dim=1)

        # === Soft label construction ===
        with torch.no_grad():
            targets_refine = targets
            targets_refine_dist = targets
            # targets_refine = mix_ratio * F.one_hot(targets, num_classes=config['num_classes']).to(targets.device)\
            #     + (1-mix_ratio)*F.softmax(logits_global, dim=1)
            
            # targets_refine_dist = F.softmax(targets_refine, dim=-1)
            # targets_refine = targets_refine.max(1).indices

            linear_accurate_global_rein = (pred_global == targets_refine)
            linear_accurate = (pred_rein == targets_refine)

            # === Clean detection statistics ===
            is_actually_clean = (targets_refine == clean_targets)
            total_clean = is_actually_clean.sum().item()
            correct_detected_clean = (linear_accurate & is_actually_clean).sum().item()
            false_detected_clean = (linear_accurate & ~is_actually_clean).sum().item()
            missed_clean = (~linear_accurate & is_actually_clean).sum().item()

            detected_clean_total += linear_accurate.sum().item()
            true_clean_total += total_clean
            correct_detected_clean_total += correct_detected_clean
            false_detected_clean_total += false_detected_clean
            missed_clean_total += missed_clean

        for i in range(targets.size(0)):
            # if linear_accurate[i]:  # linear accurate 조건을 만족한 경우만 카운트
            label = clean_targets[i].item()
            noise_label = targets_refine[i].item()
            pred_label_rein = pred_rein[i].item()
            pred_label = pred_rein2[i].item()
            class_total[label] += 1
            class_noise_total[noise_label] += 1
            if label == pred_label:
                class_correct[label] += 1
            if label == pred_label_rein:
                class_correct_rein[label] += 1
            if noise_label == pred_label:
                class_noise_correct[noise_label] += 1
            if noise_label == pred_label_rein:
                class_noise_correct_rein[noise_label] += 1
        
        if class_p_list is not None:
            logits_rein = logits_rein + class_p_list
            logits_rein2 = logits_rein2 + class_p_list
            
            # logits_rein = logits_rein + 0.5 * class_p_list + 0.5 * dynamic_boost
            # logits_rein2 = logits_rein2 + 0.5 * class_p_list + 0.5 * dynamic_boost
            
            # logits_rein = logits_rein + 0.5 * dynamic_boost
            # logits_rein2 = logits_rein2 + 0.5 * dynamic_boost
        
        loss_ce = F.cross_entropy(logits_rein, targets_refine, reduction='none')
        loss_ce2 = F.cross_entropy(logits_rein2, targets_refine, reduction='none')

        # loss_ce = cross_entropy_soft_label(logits_rein, targets_refine_dist, reduction='none')
        # loss_ce2 = cross_entropy_soft_label(logits_rein2, targets_refine_dist, reduction='none')
        
        # loss_ls = (linear_accurate*loss_ce2).mean()+(linear_accurate_norein*loss_ce).mean() + loss_ce_norein.mean()
        # loss_ls = (linear_accurate*loss_ce2).mean()+(linear_accurate_norein*loss_ce).mean() + loss_ce_norein.mean()+\
        #             (linear_accurate_global_rein*loss_ce2).mean()+(linear_accurate_global_norein*loss_ce).mean()   
        loss_ls = (linear_accurate_global_rein*loss_ce2).mean()+loss_ce.mean()
        # loss_ls = (linear_accurate*loss_ce).mean()

        optimizer_rein.zero_grad()
        optimizer_rein2.zero_grad()

        loss_ls.backward()
        optimizer_rein.step()
        optimizer_rein2.step()

        # === Stats ===
        ce_vals.append((linear_accurate*loss_ce).mean().item())
        
        clean_ratios.append(linear_accurate.float().mean().item())
        correct1 += (pred_rein2 == targets_refine).sum().item()
        total1 += targets.size(0)

    for cls in range(num_classes):
        if class_noise_total[cls] > 0:
            acc = class_noise_correct[cls] / class_noise_total[cls]
        else:
            acc = 0.0
        dynamic_boost[cls] = (acc) ** 2
    dynamic_boost = dynamic_boost / dynamic_boost.sum()
    dynamic_boost = torch.log(dynamic_boost+1e-6)
    
    train_acc1 = 100. * correct1 / total1
    print(f"\n[Step1] Avg CE (REIN): {np.mean(ce_vals):.4f}, "
          f"Target Match (by pred==label): {np.mean(clean_ratios)*100:.2f}%, Train Acc: {train_acc1:.2f}%"
    )
    
    print("[Step2] Per-Class Accuracy (Noise):")
    per_class_stats_rein = [
        f"Class {cls}: {class_noise_correct_rein[cls]}/{class_noise_total[cls]} ({100.0 * class_noise_correct_rein[cls] / class_noise_total[cls]:.2f}%)"
        if class_noise_total[cls] > 0 else f"Class {cls}: 0/0 (0.00%)"
        for cls in range(num_classes)
    ]
    print(" | ".join(per_class_stats_rein))
    per_class_stats = [
        f"Class {cls}: {class_noise_correct[cls]}/{class_noise_total[cls]} ({100.0 * class_noise_correct[cls] / class_noise_total[cls]:.2f}%)"
        if class_noise_total[cls] > 0 else f"Class {cls}: 0/0 (0.00%)"
        for cls in range(num_classes)
    ]
    print(" | ".join(per_class_stats))
    
    return class_total, dynamic_boost
    

def client_update_with_LA(client_model, global_model, optimizers, loader, config, mix_ratio, class_total, class_p_list=None, dynamic_boost=None):
    num_classes = config['num_classes']
    device = config['device']

    client_model.eval()
    client_model.train()

    optimizer_rein = optimizers
    total_loss = 0
    # 클래스 분포 기반 가중치 및 조정값
    if isinstance(class_total, dict):
        class_total_tensor = torch.tensor(
            [class_total.get(i, 0) for i in range(num_classes)],
            dtype=torch.float,
            device=device
        )
    else:
        class_total_tensor = torch.tensor(class_total).to(device).float()
    class_weights = F.softmax(class_total_tensor, dim=0)

    ce_vals, clean_ratios = [], []
    correct1, total1 = 0, 0
    class_total = defaultdict(int)
    class_correct_rein = defaultdict(int)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # === Guide model ===
        with torch.no_grad():
            feats_global = global_model.forward_features(inputs)[:, 0, :]
            logits_global = global_model.linear_rein(feats_global)
            pred_global = logits_global.argmax(dim=1)
            
        # === Client model ===
        # feats_rein = client_model.forward_features_LA(inputs)[:, 0, :]
        # logits_rein = client_model.linear_rein(feats_rein)
        # pred_rein = logits_rein.argmax(dim=1)
        # feats_norein = client_model.forward_features_no_rein(inputs)[:, 0, :]
        # logits_norein = client_model.linear_norein(feats_norein)
        
        feats_rein = client_model.forward_features(inputs)[:, 0, :]
        logits_rein = client_model.linear_rein(feats_rein)
        
        feats_rein2 = client_model.forward_features2(inputs)[:, 0, :]
        logits_rein2 = client_model.linear_rein2(feats_rein2)
        
        if class_p_list is not None:
            # logits_norein = logits_norein + class_p_list
            logits_rein = logits_rein + class_p_list
            logits_rein2 = logits_rein2 + class_p_list
        
        with torch.no_grad():
            # guide_norein = logits_norein.argmax(dim=1)
            guide_rein = logits_rein.argmax(dim=1)
        pred_rein = logits_rein2.argmax(dim=1)

        # === Soft label construction ===
        with torch.no_grad():
        #     one_hot = F.one_hot(targets, num_classes=num_classes).to(device)
        #     model_pred = logits_global.argmax(dim=1)
        #     model_one_hot = F.one_hot(model_pred, num_classes=num_classes).to(device)
        #     targets_refine_multi = torch.clamp(one_hot + model_one_hot, max=1)
            # linear_accurate_norein = (guide_norein == targets)
            linear_accurate = (guide_rein == targets)

        for i in range(targets.size(0)):
            label = targets[i].item()
            pred_label_rein = pred_rein[i].item()

            class_total[label] += 1

            if label == pred_label_rein:
                class_correct_rein[label] += 1

        # loss_bce = F.binary_cross_entropy_with_logits(logits_rein, targets_refine_multi.float())
        # loss_norein = F.cross_entropy(logits_norein, targets, reduction='none')
        loss_ce = F.cross_entropy(logits_rein/0.8, targets, reduction='none')
        loss_ce2 = F.cross_entropy(logits_rein2/0.8, targets, reduction='none')
        loss_soft = F.kl_div(torch.log_softmax(logits_rein2/0.8, dim=-1), torch.softmax(logits_global, dim=-1), reduction='none')
        loss_soft = 10*loss_soft.mean(dim=-1)
        
        # print(loss_ce, loss_soft)
        # loss_ce = loss_ce.mean()
        loss_agree = (linear_accurate*loss_ce2).mean() + loss_ce.mean()
        loss_disagree = ((~linear_accurate)*((1-mix_ratio)*loss_ce2 + mix_ratio*loss_soft)).mean()
        
        # loss_agree = (linear_accurate*loss_ce2).mean() + (linear_accurate_norein*loss_ce).mean() + loss_norein.mean()
        # loss_disagree = ((~linear_accurate)*((1-mix_ratio)*loss_ce2 + mix_ratio*loss_soft)).mean()
        
        # loss = loss_agree + loss_disagree
        loss = loss_agree

        optimizer_rein.zero_grad()
        loss.backward()
        optimizer_rein.step()

        ce_vals.append(loss.mean().item())
        clean_ratios.append(linear_accurate.float().mean().item())

        # correct_mask = targets_refine_multi[torch.arange(len(pred_rein)), pred_rein] == 1
        # correct1 += correct_mask.sum().item()
        
        correct1 += (pred_rein == targets).sum().item()
        total1 += targets.size(0)

    # === Dynamic Boost 업데이트 ===
    for cls in range(num_classes):
        if class_total[cls] > 0:
            acc = class_correct_rein[cls] / class_total[cls]
        else:
            acc = 0.0
        dynamic_boost[cls] = acc
    dynamic_boost = torch.log(dynamic_boost + 1e-6)

    # train_acc1 = 100. * correct1 / total1
    # print(f"\n[Step1] Avg CE (REIN): {np.mean(ce_vals):.4f}, "
    #       f"Target Match (pred==guide): {np.mean(clean_ratios) * 100:.2f}%, Train Acc: {train_acc1:.2f}%")

    # print("[Step2] Per-Class Accuracy :")
    # per_class_stats_rein = [
    #     f"Class {cls}: {class_correct_rein[cls]}/{class_total[cls]} "
    #     f"({100.0 * class_correct_rein[cls] / class_total[cls]:.2f}%)"
    #     if class_total[cls] > 0 else f"Class {cls}: 0/0 (0.00%)"
    #     for cls in range(num_classes)
    # ]
    # print(" | ".join(per_class_stats_rein))

    return class_total, dynamic_boost
    
    
    
    
    
    
    
    
# ROFL
class LocalUpdateRFL:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs
        
        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1) 
        self.loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=1, shuffle=True)
            
    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, lambda_cen, lambda_e, new_labels):
        mse = torch.nn.MSELoss(reduce=False)
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)
   
        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))
        
        if self.args.g_epoch < 100:
            lambda_cen = 0.01 * (self.args.g_epoch+1)
        
        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)
             
    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).cuda()
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]
        
        return ind_update
        
    def train(self, net, f_G, client_num):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        
        net.eval()
        f_k = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)
        
        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net.forward_wfeat(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)    
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1
            
        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1           
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()        
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net.forward_wfeat(images)
                feature = feature.detach()
                f_k = f_k.to(self.args.device)
                
                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)

                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1
 
                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:    
                        self.pseudo_labels[idx[i]] = labels[i]
                
                # For loss calculating
                idx = idx.cpu()  # <- 반드시 CPU로
                small_loss_idxs = small_loss_idxs.cpu()
                new_labels = mask[small_loss_idxs]*labels[small_loss_idxs] + (1-mask[small_loss_idxs])*self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)
                
                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, self.args.lambda_cen, self.args.lambda_e, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k