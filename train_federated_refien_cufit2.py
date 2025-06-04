import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from imbalanced_dataset_sampler.torchsampler.imbalanced import ImbalancedDatasetSampler
import util
import dino_variant
from rein import LoRAFusedDualReinsDinoVisionTransformer, ReinsDinoVisionTransformer

def average_reins2(global_model, client_models):
    tokens = torch.stack([client.reins2.learnable_tokens.data for client in client_models])
    global_model.reins.learnable_tokens.data.copy_(tokens.mean(dim=0))

    # 2. Average fusion_alpha
    fusion_alphas = torch.stack([client.fusion_alpha.data for client in client_models])
    global_model.fusion_alpha.data.copy_(fusion_alphas.mean(dim=0))

    # 3. Average linear_rein weights and biases
    weight_sum = sum(client.linear_rein.weight.data for client in client_models)
    bias_sum = sum(client.linear_rein.bias.data for client in client_models)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_models))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_models))

    # 4. Broadcast global reins2, fusion_alpha, and linear head to clients
    for client in client_models:
        # Update reins2
        client.reins2.load_state_dict(global_model.reins.state_dict())
        for p in client.reins2.parameters():
            p.requires_grad = True
        client.reins2.train()

        # Update fusion_alpha
        client.fusion_alpha.data.copy_(global_model.fusion_alpha.data)
        client.fusion_alpha.requires_grad = True

        # Update linear head
        client.linear_rein.load_state_dict(global_model.linear_rein.state_dict())

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
    client_model.eval()
    fused_model = copy.deepcopy(client_model)
    fused_model.eval()
    for p in fused_model.parameters():
        p.requires_grad = False

    client_model.eval()
    client_model.train2()

    mse_vals, ce_vals, clean_ratios_1 = [], [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            fused_feats = fused_model.forward_fused_features(inputs)[:, 0, :]
            global_model.eval()
            softmax_global = F.softmax(
                global_model.linear_rein(global_model.forward_features(inputs)[:, 0, :]), dim=1
            )
            soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_global
            refined_labels = soft_targets.argmax(dim=1)

        feats2 = client_model.forward_features2(inputs)[:, 0, :]
        logits2 = client_model.linear_rein(feats2)
        pred2 = logits2.argmax(dim=1)
        linear_accurate = (pred2 == refined_labels)

        loss_mse = F.mse_loss(feats2, fused_feats)
        ce = F.cross_entropy(logits2, refined_labels, reduction='none')
        # loss_ce = (linear_accurate * ce).mean()

        loss_ce = (0.3 * ce + 0.7 * linear_accurate * ce).mean()

        loss_ls = loss_mse + loss_ce

        mse_vals.append(loss_mse.item())
        ce_vals.append(loss_ce.item())
        clean_ratios_1.append(linear_accurate.float().mean().item())

        optimizer.zero_grad()
        loss_ls.backward()
        optimizer.step()

    print(f"[Step1] Avg MSE: {np.mean(mse_vals):.4f}, Avg CE: {np.mean(ce_vals):.4f}, Clean Match: {np.mean(clean_ratios_1)*100:.2f}%")

    # ----------- Step 2: Tune A^c only ----------- #
    client_model.eval()
    client_model.train1()

    ce_vals, kd_vals, mkd_vals, clean_ratios_2 = [], [], [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            global_model.eval()
            softmax_global = F.softmax(global_model.linear_rein(global_model.forward_features(inputs)[:, 0, :]), dim=1)
            soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_global
            refined_labels = soft_targets.argmax(dim=1)

        feats_fused = client_model.forward_fused_features(inputs)[:, 0, :]
        logits_fused = client_model.linear_rein(feats_fused)
        pred_fused = logits_fused.argmax(dim=1)
        linear_accurate = (pred_fused == refined_labels)

        feats2 = client_model.forward_features2(inputs)[:, 0, :].detach()
        logits2 = client_model.linear_rein(feats2)

        ce = F.cross_entropy(logits_fused, refined_labels, reduction='none')
        kd = F.kl_div(
            F.log_softmax(logits_fused / T, dim=1),
            F.softmax(logits2 / T, dim=1),
            reduction='none'
        ).sum(dim=1) * (T * T)
        # loss_ac = (linear_accurate * (ce + lambda_kd * kd)).mean()
        loss_ac = (0.3* (ce + lambda_kd * kd) + 0.7 * linear_accurate * (ce + lambda_kd * kd)).mean()


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

        loss = loss_ac + mkd_lambda * loss_mkd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Step2] CE: {np.mean(ce_vals):.4f}, KD: {np.mean(kd_vals):.4f}, MKD: {np.mean(mkd_vals):.4f}, Clean Match: {np.mean(clean_ratios_2)*100:.2f}%")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--low', type=float, default=0.4)
    parser.add_argument('--high', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--noisy_mod', type=str, choices=['iid', 'non_iid'])
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7)
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--noisy_client_rate', '-p', type=float, default=1.0)
    parser.add_argument('--min_noisy_level', '-l', type=float, default=0.0)
    parser.add_argument('--num_users', '-c', type=int, default=10)
    parser.add_argument('--epoch1', type=int, default=5)
    parser.add_argument('--epoch2', type=int, default=100)
    parser.add_argument('--frequency', type=int, default=5)
    args = parser.parse_args()

    config = util.read_conf('conf/' + args.data + '_h100.json')
    device = 'cuda:' + args.gpu
    config['device'] = device
    config['kd_lambda'] = 3.0
    config['mkd_lambda'] = 3.0
    config['kd_temperature'] = 3.0

    save_path = os.path.join(config['save_path'], args.save_path)
    os.makedirs(save_path, exist_ok=True)
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    lr_decay = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]

    dataset_train, dataset_test, dict_users = util.get_dataset(args)
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = util.add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    else:
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    pretrained = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = pretrained.state_dict()

    # Step 1: Server Initialization (Reins only)
    global_model = ReinsDinoVisionTransformer(**variant)
    global_model.load_state_dict(dino_state_dict, strict=False)
    global_model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    global_model.to(device)
    global_model.eval()
    global_model.fusion_alpha = nn.Parameter(torch.tensor(0.5, device=device), requires_grad=True)

    # Step 2: Clients Initialization (dual adapter + fusion)
    client_model_list, optimizer_list, scheduler_list = [], [], []
    for _ in range(args.num_users):
        model = LoRAFusedDualReinsDinoVisionTransformer(**variant)
        model.load_state_dict(dino_state_dict, strict=False)
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)

        model.reins2.load_state_dict(global_model.reins.state_dict())
        for p in model.reins2.parameters():
            p.requires_grad = False
        model.reins2.eval()
        model.fusion_alpha.requires_grad_(True)

        client_model_list.append(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    # Step 3: Local Pretraining (epoch1) on reins1 + linear
    print("\n[Step 3] Local Pretraining (epoch1)")
    valid_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=16)
    
    for client_idx in range(args.num_users):
        model = client_model_list[client_idx]
        optimizer = optimizer_list[client_idx]
        scheduler = scheduler_list[client_idx]
        model.train1()

        sample_idx = list(dict_users[client_idx])
        train_dataset = Subset(dataset_train, sample_idx)
        train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size, num_workers=16)

        for ep in range(args.epoch1):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                feats = model.forward_features1(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        acc = util.validation_accuracy(model, valid_loader, device, mode='rein')
        print(f"[Client {client_idx}] Pretrain Acc (reins1): {acc*100:.2f}%")

    ### ✅ Step 3.5: Pretrained reins1, linear → 서버 어댑터(reins2) 및 head로 초기화
    print("\n[Step 3.5] Initial Global Adapter Aggregation")
    # global_model.reins ← average of client.reins1
    tokens = torch.stack([client.reins1.learnable_tokens.data for client in client_model_list])
    global_model.reins.learnable_tokens.data.copy_(tokens.mean(dim=0))

    # linear head average
    weight_sum = sum(client.linear_rein.weight.data for client in client_model_list)
    bias_sum = sum(client.linear_rein.bias.data for client in client_model_list)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_model_list))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_model_list))

    # broadcast to client.reins2
    for client in client_model_list:
        client.reins2.load_state_dict(global_model.reins.state_dict())
        client.linear_rein.load_state_dict(global_model.linear_rein.state_dict())
        for p in client.reins2.parameters():
            p.requires_grad = False
        client.reins2.eval()

    # Step 4: Federated Learning with Label Refinement (epoch2)
    valid_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=16)
    best_accuracy = 0.0

    for epoch in range(args.epoch2):
        print(f"\n→ Global Round {epoch+1}/{args.epoch2}")
        step = (args.high - args.low) / args.duration
        decay_step = epoch // max(1, args.epoch2 // args.duration)
        mix_ratio = args.high - step * decay_step

        for client_idx in range(args.num_users):
            model = client_model_list[client_idx]
            optimizer = optimizer_list[client_idx]
            scheduler = scheduler_list[client_idx]
            sample_idx = list(dict_users[client_idx])
            train_dataset = Subset(dataset_train, sample_idx)
            train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size, num_workers=16)

            client_update_with_refinement(model, global_model, optimizer, train_loader, config, mix_ratio)
            scheduler.step()
            acc = util.validation_accuracy(model, valid_loader, device, mode='dual')
            print(f"[Client {client_idx}] Eval Acc: {acc*100:.2f}%")

        if (epoch + 1) % args.frequency == 0 or (epoch + 1) == args.epoch2:
            print("\n→ Step 5: Federated Averaging of A^s and Evaluation")
            average_reins2(global_model, client_model_list)
            acc = util.validation_accuracy(global_model, valid_loader, device, mode='rein')
            print(f"[Server Eval] Global A^s Acc = {acc*100:.2f}%")
            if acc > best_accuracy:
                best_accuracy = acc
                torch.save(global_model.state_dict(), os.path.join(save_path, "best_global_model.pth"))
            torch.save(global_model.state_dict(), os.path.join(save_path, f"global_model_round_{epoch+1}.pth"))

    torch.save(global_model.state_dict(), os.path.join(save_path, "final_global_model.pth"))
    print("\n→ Final global model saved.")

if __name__ == "__main__":
    train()
