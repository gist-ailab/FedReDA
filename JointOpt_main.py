import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import copy
import numpy as np
import random
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.data as Data
from tqdm import tqdm

from util.options import args_parser
from util.util import load_data, wrap_as_local_dataset, split_equally_across_clients, wrap_subsets_to_local_dataset
from model_dino import ReinDinov2
from dino_variant import _small_variant
from update import evaluate, average_weights

# Path to pretrained DINO checkpoint
dino_ckpt = '/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'

def main(args):
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.alpha = 1.0
    args.beta = 0.5

    # === Directory setup ===
    save_dir = os.path.join(args.result_dir, args.dataset, f'iid_{args.iid}')
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, f'seed_{args.seed}_rate_{args.noise_rate}')
    os.makedirs(model_dir, exist_ok=True)

    # === Load data and split ===
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type=args.dataset)
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type=args.dataset)

    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)

    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # === Initialize soft labels ===
    for dataset in clients_train_dataset_list:
        N = len(dataset)
        dataset.soft_labels = np.eye(args.num_classes)[dataset.local_noisy_labels]
        dataset.prediction = np.zeros((N, 10, args.num_classes), dtype=np.float32)
        dataset.count = 0

    # === Global model declaration ===
    global_model = ReinDinov2(_small_variant, torch.load(dino_ckpt), args.num_classes).cuda()

    # === Stage 1: Soft Label Refinement ===
    for round in range(args.round1):
        print(f"\n[Stage 1 - Round {round+1}] Soft Label Refinement")

        for client_id, dataset in enumerate(clients_train_dataset_list):
            model = copy.deepcopy(global_model).cuda()
            model.train()

            loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)

            for inputs, _, _, indices in loader:
                inputs = inputs.cuda()
                indices = indices.cuda()
                soft_targets = torch.tensor(dataset.soft_labels[indices.cpu().numpy()],
                                            dtype=torch.float32, device=inputs.device)

                outputs = model(inputs)
                prob = F.softmax(outputs, dim=1)
                prob_mean = prob.mean(dim=0)
                prior = torch.ones(args.num_classes, device=prob.device) / args.num_classes

                loss_c = -(soft_targets * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                loss_p = -(torch.log(prob_mean + 1e-6) * prior).sum()
                loss_e = -(F.log_softmax(outputs, dim=1) * prob).sum(dim=1).mean()
                loss = loss_c + args.alpha * loss_p + args.beta * loss_e

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Save prediction for averaging
                pred_np = prob.detach().cpu().numpy()
                for i, idx in enumerate(indices):
                    dataset.prediction[idx, dataset.count % 10] = pred_np[i]

            dataset.count += 1

            if dataset.count >= args.round1:
                dataset.soft_labels = dataset.prediction.mean(axis=1)
                dataset.local_noisy_labels = np.argmax(dataset.soft_labels, axis=1)

            # if dataset.count == args.round1:
            #     np.save(os.path.join(model_dir, f'client{client_id}_soft_labels.npy'), dataset.soft_labels)
            #     np.save(os.path.join(model_dir, f'client{client_id}_refined_labels.npy'), dataset.local_noisy_labels)

    # === Stage 2: Retraining with Fixed Soft Labels ===
    print(f"\n[Stage 2] Retraining with Soft Labels")

    for round in range(args.round3):
        print(f"\n[Stage 2 - Round {round+1}]")
        
        client_weights = []
        for client_id, dataset in enumerate(clients_train_dataset_list):
            model = copy.deepcopy(global_model)
            model.train()

            loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)

            for inputs, _, _, indices in loader:
                inputs = inputs.cuda()
                indices = indices.cuda()
                soft_targets = torch.tensor(dataset.soft_labels[indices.cpu().numpy()],
                                            dtype=torch.float32, device=inputs.device)

                outputs = model(inputs)
                prob = F.softmax(outputs, dim=1)
                prob_mean = prob.mean(dim=0)
                prior = torch.ones(args.num_classes, device=prob.device) / args.num_classes

                loss_c = -(soft_targets * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                loss_p = -(torch.log(prob_mean + 1e-6) * prior).sum()
                loss_e = -(F.log_softmax(outputs, dim=1) * prob).sum(dim=1).mean()
                loss = loss_c + args.alpha * loss_p + args.beta * loss_e

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            client_weights.append(copy.deepcopy(model.state_dict()))
        
        # Global model averaging
        global_model.load_state_dict(average_weights(client_weights))

    # === Final Evaluation ===
    print("\n[Evaluation] Final model testing...")
    test_acc = evaluate(test_loader, global_model)
    print(f"Test Accuracy: {test_acc:.4f} %")


if __name__ == "__main__":
    args = args_parser()
    main(args)
