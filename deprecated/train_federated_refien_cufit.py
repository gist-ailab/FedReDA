import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.nn.functional as F

import util
import rein
import dino_variant

def average_tokens_and_head(global_model, client_model_list, variant):
    # Average Rein Tokens
    tokens_all = torch.stack([client_model.reins.learnable_tokens.data.clone() for client_model in client_model_list], dim=0)
    global_model.reins.learnable_tokens.data.copy_(tokens_all.mean(dim=0))

    # Average linear_rein head
    weight_sum = sum(client_model.linear_rein.weight.data for client_model in client_model_list)
    bias_sum = sum(client_model.linear_rein.bias.data for client_model in client_model_list)
    global_model.linear_rein.weight.data.copy_(weight_sum / len(client_model_list))
    global_model.linear_rein.bias.data.copy_(bias_sum / len(client_model_list))

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
    args = parser.parse_args()

    config = util.read_conf('conf/' + args.data + '_h100.json')
    device = 'cuda:' + args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    os.makedirs(save_path, exist_ok=True)

    lr_decay = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]

    dataset_train, dataset_test, dict_users = util.get_dataset(args)
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = util.add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    global_model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = global_model.state_dict()

    global_model = rein.ReinsDinoVisionTransformer(**variant)
    global_model.load_state_dict(dino_state_dict, strict=False)
    global_model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    global_model.to(device)
    global_criterion = nn.CrossEntropyLoss(reduction='none')
    global_model.eval()

    client_model_list, optimizer_list, scheduler_list = [], [], []
    for client_idx in range(args.num_users):
        client_model = rein.ReinsDinoVisionTransformer(**variant)
        client_model.load_state_dict(dino_state_dict, strict=False)
        client_model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
        client_model.to(device)
        client_model.train()
        client_model_list.append(client_model)

        optimizer = torch.optim.AdamW(client_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    print("\n[Step 0] Local Pretraining for Each Client")
    for client_idx in range(args.num_users):
        client_model = client_model_list[client_idx]
        optimizer = optimizer_list[client_idx]
        scheduler = scheduler_list[client_idx]

        sample_idx = list(dict_users[client_idx])
        client_dataset = torch.utils.data.Subset(dataset_train, sample_idx)
        client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        for epoch in range(args.epoch1):
            client_model.train()
            for inputs, targets in client_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = client_model.linear_rein(client_model.forward_features(inputs)[:, 0, :])
                loss = global_criterion(outputs, targets).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            valid_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
            client_model.eval()
            acc = util.validation_accuracy(client_model, valid_loader, device, mode='rein')
            print(f"[Client {client_idx}] Epoch {epoch+1} Eval Acc: {acc*100:.2f}%")

    print("\n[Step 1] Averaging all Rein Tokens and linear_rein → Global Model")
    average_tokens_and_head(global_model, client_model_list, variant)

    print("\n→ Evaluating Averaged Global Adapter on Validation Set")
    valid_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    init_valid_accuracy = util.validation_accuracy(global_model, valid_loader, device, mode='rein')
    print(f"Initial Validation Accuracy after Averaging: {init_valid_accuracy*100:.2f}%\n")
    best_accuracy = init_valid_accuracy

    print("\n[Step 2] Federated Learning with Global Adapter-based Label Refinement")
    for epoch in range(args.epoch2):
        print(f"\n→ Global Round {epoch+1}/{args.epoch2}")
        step = (args.high - args.low) / args.duration
        decay_step = epoch // max(1, args.epoch2 // args.duration)
        mix_ratio = args.high - step * decay_step

        for client_idx in range(args.num_users):
            client_model = client_model_list[client_idx]
            optimizer = optimizer_list[client_idx]
            scheduler = scheduler_list[client_idx]

            sample_idx = list(dict_users[client_idx])
            client_dataset = torch.utils.data.Subset(dataset_train, sample_idx)
            train_loader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            client_model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    global_model.eval()
                    softmax_global = F.softmax(global_model.linear_rein(global_model.forward_features(inputs)[:, 0, :]), dim=1)
                    global_pred = softmax_global.max(1).indices

                soft_targets = mix_ratio * F.one_hot(targets, num_classes=config['num_classes']).float() \
                               + (1 - mix_ratio) * softmax_global
                hard_targets = torch.softmax(soft_targets, dim=-1).max(1).indices
                linear_accurate = (global_pred == hard_targets)

                outputs = client_model.linear_rein(client_model.forward_features(inputs)[:, 0, :])
                loss = linear_accurate*util.cross_entropy_soft_label(outputs, soft_targets, reduction='none')
                loss = loss.mean()
                loss.backward()
                optimizer.step()

            scheduler.step()
            acc = util.validation_accuracy(client_model, valid_loader, device, mode='rein')
            print(f"[Client {client_idx}] Post-Epoch Eval Acc: {acc*100:.2f}%")

        if (epoch + 1) % args.frequency == 0 or (epoch + 1) == args.epoch2:
            print("\n→ Federated Averaging and Global Evaluation")
            average_tokens_and_head(global_model, client_model_list, variant)
            valid_accuracy = util.validation_accuracy(global_model, valid_loader, device, mode='rein')
            print(f"Validation Acc = {valid_accuracy*100:.2f}%")
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save(global_model.state_dict(), os.path.join(save_path, "best_global_model.pth"))
                print(f"→ Best model saved with acc {best_accuracy*100:.2f}%")
            torch.save(global_model.state_dict(), os.path.join(save_path, f"global_model_round_{epoch+1}.pth"))
            print(f"→ Model saved at round {epoch+1}")

    torch.save(global_model.state_dict(), os.path.join(save_path, "final_global_model.pth"))
    print("\n→ Final global model saved.")

if __name__ == "__main__":
    train()
