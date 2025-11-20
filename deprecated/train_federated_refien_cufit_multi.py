import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from multiprocessing import get_context

import util
import dino_variant
from rein import LoRAFusedDualReinsDinoVisionTransformer, ReinsDinoVisionTransformer

# average and update global model
def average_reins2(global_model, client_models):
    tokens = torch.stack([client.reins2.learnable_tokens.data for client in client_models])
    global_model.reins.learnable_tokens.data.copy_(tokens.mean(dim=0))
    fusion_alphas = torch.stack([client.fusion_alpha.data for client in client_models])
    global_model.fusion_alpha.data.copy_(fusion_alphas.mean(dim=0))

    weight_sum = sum(client.linear_rein.weight.data for client in client_models)
    bias_sum = sum(client.linear_rein.bias.data for client in client_models)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_models))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_models))

    for client in client_models:
        client.reins2.load_state_dict(global_model.reins.state_dict())
        for p in client.reins2.parameters():
            p.requires_grad = True
        client.reins2.train()
        client.fusion_alpha.data.copy_(global_model.fusion_alpha.data)
        client.fusion_alpha.requires_grad = True
        client.linear_rein.load_state_dict(global_model.linear_rein.state_dict())

# local update with optional distillation
def client_update_with_refinement(model, global_model, optimizer, loader, config, mix_ratio, client_idx):
    import copy
    T = config['kd_temperature']
    lambda_kd = config['kd_lambda']
    mkd_lambda = config['mkd_lambda']
    num_classes = config['num_classes']
    device = config['device']

    model.eval()
    fused_model = copy.deepcopy(model).to(device)
    fused_model.eval()
    for p in fused_model.parameters():
        p.requires_grad = False

    model.train2()
    mse_vals, ce_vals, clean_ratios_1 = [], [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            fused_feats = fused_model.forward_fused_features(inputs)[:, 0, :]
            softmax_global = F.softmax(global_model.linear_rein(global_model.forward_features(inputs)[:, 0, :]), dim=1)
            soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_global
            refined_labels = soft_targets.argmax(dim=1)

        feats2 = model.forward_features2(inputs)[:, 0, :]
        logits2 = model.linear_rein(feats2)
        pred2 = logits2.argmax(dim=1)
        linear_accurate = (pred2 == refined_labels)
        loss_mse = F.mse_loss(feats2, fused_feats)
        ce = F.cross_entropy(logits2, refined_labels, reduction='none')
        loss_ce = (linear_accurate * ce).mean()
        loss = loss_mse + loss_ce

        mse_vals.append(loss_mse.item())
        ce_vals.append(loss_ce.item())
        clean_ratios_1.append(linear_accurate.float().mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Client {client_idx}] [Step1] MSE: {np.mean(mse_vals):.4f}, CE: {np.mean(ce_vals):.4f}, Clean: {np.mean(clean_ratios_1)*100:.2f}%")

    model.train1()
    ce_vals, kd_vals, mkd_vals, clean_ratios_2 = [], [], [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            softmax_global = F.softmax(global_model.linear_rein(global_model.forward_features(inputs)[:, 0, :]), dim=1)
            soft_targets = mix_ratio * F.one_hot(targets, num_classes=num_classes).float() + (1 - mix_ratio) * softmax_global
            refined_labels = soft_targets.argmax(dim=1)

        feats_fused = model.forward_fused_features(inputs)[:, 0, :]
        logits_fused = model.linear_rein(feats_fused)
        pred_fused = logits_fused.argmax(dim=1)
        linear_accurate = (pred_fused == refined_labels)

        feats2 = model.forward_features2(inputs)[:, 0, :].detach()
        logits2 = model.linear_rein(feats2)
        ce = F.cross_entropy(logits_fused, refined_labels, reduction='none')
        kd = F.kl_div(F.log_softmax(logits_fused / T, dim=1),
                      F.softmax(logits2 / T, dim=1), reduction='none').sum(dim=1) * (T * T)
        loss_ac = (linear_accurate * (ce + lambda_kd * kd)).mean()

        logits2_upd = model.linear_rein(feats2)
        loss_mkd = F.kl_div(
            F.log_softmax(logits2_upd / T, dim=1),
            F.softmax(logits_fused.detach() / T, dim=1),
            reduction='batchmean') * (T * T)

        loss = loss_ac + mkd_lambda * loss_mkd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ce_vals.append(ce.mean().item())
        kd_vals.append(kd.mean().item())
        mkd_vals.append(loss_mkd.item())
        clean_ratios_2.append(linear_accurate.float().mean().item())

    print(f"[Client {client_idx}] [Step2] CE: {np.mean(ce_vals):.4f}, KD: {np.mean(kd_vals):.4f}, MKD: {np.mean(mkd_vals):.4f}, Clean: {np.mean(clean_ratios_2)*100:.2f}%")

    return model.cpu().state_dict()

# worker wrapper
def train_one(args_tuple):
    model, global_model, optimizer, scheduler, train_loader, config, mix_ratio, client_idx, valid_loader = args_tuple
    model.to(config['device'])
    global_model.to(config['device'])
    global_model.eval()
    state_dict = client_update_with_refinement(model, global_model, optimizer, train_loader, config, mix_ratio, client_idx)
    scheduler.step()
    acc = util.validation_accuracy(model, valid_loader, config['device'], mode='dual')
    print(f"[Client {client_idx}] Eval Acc: {acc*100:.2f}%")
    return client_idx, model.cpu().state_dict()

# main training function
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--seed', type=int, default=13)
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
    parser.add_argument('--num_processes', type=int, default=10)
    args = parser.parse_args()

    config = util.read_conf(f'conf/{args.data}_h100.json')
    config['device'] = f'cuda:{args.gpu}'
    config.update({'kd_lambda': 1.0, 'mkd_lambda': 1.0, 'kd_temperature': 1.0})

    save_path = os.path.join(config['save_path'], args.save_path)
    os.makedirs(save_path, exist_ok=True)
    batch_size = int(config['batch_size'])

    dataset_train, dataset_test, dict_users = util.get_dataset(args)
    dataset_train.targets = util.add_noise(args, np.array(dataset_train.targets), dict_users)[0]

    model_load, variant = {
        's': (dino_variant._small_dino, dino_variant._small_variant),
        'b': (dino_variant._base_dino, dino_variant._base_variant),
        'l': (dino_variant._large_dino, dino_variant._large_variant)
    }[args.netsize]

    dino_state_dict = torch.hub.load('facebookresearch/dinov2', model_load).state_dict()
    global_model = ReinsDinoVisionTransformer(**variant)
    global_model.load_state_dict(dino_state_dict, strict=False)
    global_model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    global_model.to(config['device'])
    global_model.fusion_alpha = nn.Parameter(torch.tensor(0.5, device=config['device']), requires_grad=True)

    client_model_list, optimizer_list, scheduler_list = [], [], []
    for _ in range(args.num_users):
        model = LoRAFusedDualReinsDinoVisionTransformer(**variant)
        model.load_state_dict(dino_state_dict, strict=False)
        model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(config['device'])
        model.reins2.load_state_dict(global_model.reins.state_dict())
        model.reins2.eval()
        model.fusion_alpha.requires_grad_(True)

        client_model_list.append(model)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.5*args.epoch2), int(0.75*args.epoch2)])
        optimizer_list.append(opt)
        scheduler_list.append(sch)

    valid_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False)
    ctx = get_context('spawn')
    best_accuracy = 0.0

    # Pretraining
    print("\n[Step 3] Local Pretraining (epoch1)")
    args_list = []
    for cid in range(args.num_users):
        subset = Subset(dataset_train, list(dict_users[cid]))
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False)
        args_list.append((client_model_list[cid], global_model, optimizer_list[cid],
                          scheduler_list[cid], loader, config, 1.0, cid, valid_loader))
    with ctx.Pool(processes=min(args.num_users, args.num_processes)) as pool:
        results = pool.map(train_one, args_list)
    for cid, state in results:
        client_model_list[cid].load_state_dict(state)

    # Aggregate
    print("\n[Step 3.5] Initial Global Adapter Aggregation")
    average_reins2(global_model, client_model_list)

    # Main Federated Rounds
    for epoch in range(args.epoch2):
        print(f"\n→ Global Round {epoch+1}/{args.epoch2}")
        mix_ratio = args.high - (args.high - args.low) / args.duration * (epoch // max(1, args.epoch2 // args.duration))
        args_list = []
        for cid in range(args.num_users):
            subset = Subset(dataset_train, list(dict_users[cid]))
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False)
            args_list.append((client_model_list[cid], global_model, optimizer_list[cid],
                              scheduler_list[cid], loader, config, mix_ratio, cid, valid_loader))
        with ctx.Pool(processes=min(args.num_users, args.num_processes)) as pool:
            results = pool.map(train_one, args_list)
        for cid, state in results:
            client_model_list[cid].load_state_dict(state)

        if (epoch + 1) % args.frequency == 0 or (epoch + 1) == args.epoch2:
            print("\n→ Step 5: Federated Averaging of A^s and Evaluation")
            average_reins2(global_model, client_model_list)
            acc = util.validation_accuracy(global_model, valid_loader, config['device'], mode='rein')
            print(f"[Server Eval] Global A^s Acc = {acc*100:.2f}%")
            if acc > best_accuracy:
                best_accuracy = acc
                torch.save(global_model.state_dict(), os.path.join(save_path, "best_global_model.pth"))
            torch.save(global_model.state_dict(), os.path.join(save_path, f"global_model_round_{epoch+1}.pth"))

    torch.save(global_model.state_dict(), os.path.join(save_path, "final_global_model.pth"))
    print("\n→ Final global model saved.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
