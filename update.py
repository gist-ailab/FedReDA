import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
from collections import defaultdict
import numpy as np

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
def train_one_step(net, data, label, optimizer, criterion):
    net.train()
    pred = net(data)
    # print(label)
    # print(type(label))
    loss = criterion(pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


def train(train_loader, epoch, model, optimizer1, args):
    # print('Training %s...' % model_str)
    model.train()
    train_total = 0
    train_correct = 0

    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = noisy_label.cuda()
        prec, loss = train_one_step(model, data, labels, optimizer1, nn.CrossEntropyLoss())
        train_total += 1
        train_correct += prec

        if (i+1)==len(enumerate(train_loader)):
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                    % (epoch + 1, i + 1, 4000 // args.batch_size, prec, loss.item()))

    train_acc = float(train_correct) / float(train_total)

    return train_acc


def train_prox(train_loader, epoch, model, global_model, optimizer1, mu, args):
    # print('Training %s...' % model_str)
    model.train()
    train_total = 0
    train_correct = 0

    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = noisy_label.cuda()
        prec, loss = train_prox_one_step(model, global_model, data, labels, optimizer1, nn.CrossEntropyLoss(), mu)
        train_total += 1
        train_correct += prec

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, i + 1, 4000 // args.batch_size, prec, loss.item()))

    train_acc = float(train_correct) / float(train_total)

    return train_acc


def train_prox_one_step(net, global_model, data, label, optimizer, criterion, mu):
    net.train()
    pred = net(data)
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
    with torch.no_grad():
        for data, noisy_label, clean_label, _ in val_loader:
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += noisy_label.size(0)
            correct1 += (pred1.cpu() == clean_label.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


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

def average_reins1(global_model, client_models):
    # 1. Average entire reins1 (not just learnable_tokens)
    with torch.no_grad():
        # 모든 client의 reins1 파라미터 가져오기
        reins_named_params = {}
        for name, _ in client_models[0].reins1.named_parameters():
            stacked = torch.stack([dict(client.reins1.named_parameters())[name].data for client in client_models])
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
        for p in client.reins2.parameters():
            p.requires_grad = True
        client.reins2.train()

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
        for p in client.reins2.parameters():
            p.requires_grad = True
        client.reins2.train()


def client_update_with_refinement(client_model, global_model, optimizer, loader, config, mix_ratio):
    """
    FedDAT-style two-step optimization with diagnostics:
    Step 1: Tune A^s (shared adapter) to match (A^c + A^s)-based fused features and refined labels.
    Step 2: Tune A^c (client adapter) to match improved A^s predictions and refined labels.
    """
    import copy
    import numpy as np
    import torch.nn.functional as F

    T = config['kd_temperature']
    lambda_kd = config['kd_lambda']
    mkd_lambda = config['mkd_lambda']
    num_classes = config['num_classes']
    device = config['device']

    # ----------- Step 1: Tune A^s only ----------- #
    fused_model = copy.deepcopy(client_model)
    fused_model.eval()

    client_model.eval()
    client_model.train2()

    mse_vals, ce_vals, clean_ratios_1 = [], [], []
    correct1, total1 = 0, 0

    for inputs, targets, _, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            fused_feats = fused_model.forward_fused_features(inputs)[:, 0, :]
            softmax_fused = F.softmax(
                fused_model.linear_rein(fused_feats), dim=1
            )
            pred_global = softmax_fused.argmax(dim=1)
            # soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_fused
            # refined_labels = soft_targets.argmax(dim=1) 

        feats2 = client_model.forward_features2(inputs)[:, 0, :]
        logits2 = client_model.linear_rein(feats2)
        pred2 = logits2.argmax(dim=1)
        # linear_accurate = (pred_global == refined_labels)
        linear_accurate = (pred_global == targets) 

        loss_mse = F.mse_loss(feats2, fused_feats)
        # ce = F.cross_entropy(logits2, refined_labels, reduction='none')
        ce = F.cross_entropy(logits2, targets, reduction='none')
        loss_ce = (linear_accurate * ce).mean()
        loss_ls = 0.005*loss_mse + loss_ce

        mse_vals.append(0.005*loss_mse.item())
        ce_vals.append(loss_ce.item())
        clean_ratios_1.append(linear_accurate.float().mean().item())
        # correct1 += (pred2 == refined_labels).sum().item()
        correct1 += (pred2 == targets).sum().item()
        # total1 += refined_labels.size(0)
        total1 += targets.size(0)

        optimizer.zero_grad()
        loss_ls.backward()
        optimizer.step()

    train_acc1 = 100. * correct1 / total1
    print(f"[Step1] Avg MSE: {np.mean(mse_vals):.4f}, Avg CE: {np.mean(ce_vals):.4f}, "
          f"Clean Match: {np.mean(clean_ratios_1)*100:.2f}%, Train Acc: {train_acc1:.2f}%")

    # ----------- Average A^s into global model ----------- #
    average_reins2(client_model, global_model)
    print("[Averaging] Shared adapter (A^s) has been merged into global model.")

    # ----------- Step 2: Tune A^c only ----------- #
    client_model.eval()
    client_model.train1()

    ce_vals, kd_vals, mkd_vals, clean_ratios_2 = [], [], [], []
    correct2, total2 = 0, 0

    for inputs, targets, _, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            global_model.eval()
            softmax_global = F.softmax(global_model.linear_rein(global_model.forward_features(inputs)[:, 0, :]), dim=1)
            pred_global = softmax_global.argmax(dim=1)
            # soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_global
            # refined_labels = soft_targets.argmax(dim=1)

        feats_fused = client_model.forward_fused_features(inputs)[:, 0, :]
        logits_fused = client_model.linear_rein(feats_fused)
        pred_fused = logits_fused.argmax(dim=1)
        # linear_accurate = (pred_global == refined_labels)
        linear_accurate = (pred_global == targets)

        feats2 = client_model.forward_features2(inputs)[:, 0, :].detach()
        logits2 = client_model.linear_rein(feats2)

        # ce = F.cross_entropy(logits_fused, refined_labels, reduction='none')
        ce = F.cross_entropy(logits_fused, targets, reduction='none')
        kd = F.kl_div(
            F.log_softmax(logits_fused / T, dim=1),
            F.softmax(logits2 / T, dim=1),
            reduction='none'
        ).sum(dim=1) * (T * T)
        loss_ac = (linear_accurate * (ce + lambda_kd * kd)).mean()

        logits2_upd = client_model.linear_rein(feats2)
        loss_mkd = F.kl_div(
            F.log_softmax(logits2_upd / T, dim=1),
            F.softmax(logits_fused.detach() / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        ce_vals.append(ce.mean().item())
        kd_vals.append(kd.mean().item())
        mkd_vals.append(loss_mkd.item())
        clean_ratios_2.append(linear_accurate.float().mean().item())
        # correct2 += (pred_fused == refined_labels).sum().item()
        # total2 += refined_labels.size(0)
        correct2 += (pred_fused == targets).sum().item()
        total2 += targets.size(0)

        loss = loss_ac + mkd_lambda * loss_mkd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc2 = 100. * correct2 / total2
    print(f"[Step2] CE: {np.mean(ce_vals):.4f}, KD: {np.mean(kd_vals):.4f}, MKD: {np.mean(mkd_vals):.4f}, "
          f"Clean Match: {np.mean(clean_ratios_2)*100:.2f}%, Train Acc: {train_acc2:.2f}%")
    
def client_update_step1(client_model, optimizer, loader, config, mix_ratio):
    num_classes = config['num_classes']
    device = config['device']

    fused_model = copy.deepcopy(client_model)
    fused_model.eval()

    client_model.eval()
    client_model.train2()

    mse_vals, ce_vals, clean_ratios_1 = [], [], []
    correct1, total1 = 0, 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for inputs, targets, _, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            # fused_feats = fused_model.forward_fused_features(inputs)[:, 0, :]
            fused_feats_list = fused_model.forward_fused_features_wfeats(inputs)
            fused_feats = fused_feats_list[-1][:, 0, :]
            softmax_fused = F.softmax(fused_model.linear_rein(fused_feats), dim=1)
            pred_global = softmax_fused.argmax(dim=1)
            # soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_fused
            # refined_labels = soft_targets.argmax(dim=1)

        # feats2 = client_model.forward_features2(inputs)[:, 0, :]
        feats2_list = client_model.forward_features2_wfeats(inputs)
        feats2 = feats2_list[-1][:, 0, :]
        logits2 = client_model.linear_rein(feats2)
        pred2 = logits2.argmax(dim=1)
        linear_accurate = (pred_global == targets)
        # mask_weight = 0.1 + 0.9 * linear_accurate.float()
        mask_weight = mix_ratio + (1-mix_ratio) * linear_accurate.float() 
        
        # loss_mse = (mask_weight.unsqueeze(1)*F.mse_loss(feats2, fused_feats, reduction='none')).mean()
        loss_mse = 0
        for f_fused, f_client in zip(fused_feats_list, feats2_list):
            loss_mse += F.mse_loss(f_client, f_fused, reduction='none').mean()
        loss_ce = (mask_weight*F.cross_entropy(logits2, targets, reduction='none')).mean()

        loss_mse = F.mse_loss(feats2, fused_feats, reduction='none').mean()
        # loss_ce = F.cross_entropy(logits2, targets, reduction='none').mean()

        # loss_ls = loss_ce
        loss_ls = 5*loss_mse+loss_ce
        # loss_ls = loss_mse+10*loss_ce

        mse_vals.append(5*loss_mse.item())
        ce_vals.append(loss_ce.item())
        clean_ratios_1.append(linear_accurate.float().mean().item())
        correct1 += (pred2 == targets).sum().item()
        total1 += targets.size(0)

        for t, p in zip(targets, pred2):
            class_total[t.item()] += 1
            if t.item() == p.item():
                class_correct[t.item()] += 1

        optimizer.zero_grad()
        loss_ls.backward()
        optimizer.step()

    train_acc1 = 100. * correct1 / total1
    print(f"\n[Step1] Avg MSE: {np.mean(mse_vals):.4f}, Avg CE: {np.mean(ce_vals):.4f}, "
          f"Clean Match: {np.mean(clean_ratios_1)*100:.2f}%, Train Acc: {train_acc1:.2f}%")

    print("[Step1] Per-Class Accuracy:", end=' ')
    per_class_stats = [
        f"Class {cls}: {class_correct[cls]}/{class_total[cls]} ({100.0 * class_correct[cls] / class_total[cls]:.2f}%)"
        if class_total[cls] > 0 else f"Class {cls}: 0/0 (0.00%)"
        for cls in range(num_classes)
    ]
    print(" | ".join(per_class_stats))

def client_update_step2(client_model, global_model, optimizer, loader, config, mix_ratio):
    T = config['kd_temperature']
    lambda_kd = config['kd_lambda']
    mkd_lambda = config['mkd_lambda']
    num_classes = config['num_classes']
    device = config['device']

    client_model.eval()
    client_model.train1()  # Adapter1만 학습

    ce_vals, kd_vals, mkd_vals, clean_ratios_2 = [], [], [], []
    correct2, total2 = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for inputs, targets, _, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # === Global model forward ===
        with torch.no_grad():
            feats_global = global_model.forward_features(inputs)[:, 0, :]
            logits_global = global_model.linear_rein(feats_global)
            pred_global = logits_global.argmax(dim=1)

        # === Client model forward ===
        feats_fused = client_model.forward_fused_features(inputs)[:, 0, :]
        logits_fused = client_model.linear_rein(feats_fused)
        pred_fused = logits_fused.argmax(dim=1)

        linear_accurate = (pred_global == targets)
        # mask_weight = 0.1 + 0.9 * linear_accurate.float()
        mask_weight = mix_ratio + (1-mix_ratio) * linear_accurate.float() 

        # print("logits_fused shape:", logits_fused.shape)
        # print("logits_global shape:", logits_global.shape)

        # === KD from global model ===
        ce = F.cross_entropy(logits_fused, targets, reduction='none')
        kd = F.kl_div(
            F.log_softmax(logits_fused / T, dim=1),
            F.softmax(logits_global / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        # === MKD: adapter2 → fused ===
        with torch.no_grad():
            feats2 = client_model.forward_features2(inputs)[:, 0, :]
            logits2 = client_model.linear_rein(feats2)

        loss_mkd = F.kl_div(
            F.log_softmax(logits2 / T, dim=1),
            F.softmax(logits_fused.detach() / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        # === Total Loss ===
        loss = (mask_weight * ce).mean() + lambda_kd * kd + mkd_lambda * loss_mkd
        # loss = ce.mean() + lambda_kd * kd + mkd_lambda * loss_mkd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Stats ===
        ce_vals.append(ce.mean().item())
        kd_vals.append(kd.mean().item())
        mkd_vals.append(loss_mkd.item())
        clean_ratios_2.append(linear_accurate.float().mean().item())
        correct2 += (pred_fused == targets).sum().item()
        total2 += targets.size(0)

        for t, p in zip(targets, pred_fused):
            class_total[t.item()] += 1
            if t.item() == p.item():
                class_correct[t.item()] += 1

    train_acc2 = 100. * correct2 / total2
    print(f"\n[Step2] CE: {np.mean(ce_vals):.4f}, KD: {np.mean(kd_vals):.4f}, MKD: {np.mean(mkd_vals):.4f}, "
          f"Clean Match: {np.mean(clean_ratios_2)*100:.2f}%, Train Acc: {train_acc2:.2f}%")

    print("[Step2] Per-Class Accuracy:", end=' ')
    per_class_stats = [
        f"Class {cls}: {class_correct[cls]}/{class_total[cls]} ({100.0 * class_correct[cls] / class_total[cls]:.2f}%)"
        if class_total[cls] > 0 else f"Class {cls}: 0/0 (0.00%)"
        for cls in range(num_classes)
    ]
    print(" | ".join(per_class_stats))