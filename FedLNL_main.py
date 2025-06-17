import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from util.util import *
from util.options import args_parser
from dino_variant import _small_dino, _small_variant
from rein import LoRAFusedDualReinsDinoVisionTransformer, ReinsDinoVisionTransformer
from update import average_reins1, average_reins2, client_update_step1, client_update_step2
import util
from imbalanced_dataset_sampler.torchsampler.imbalanced import ImbalancedDatasetSampler

def main(args):
    device = torch.device(f"cuda:{args.gpu}")
    config = {
        'device': device,
        'kd_lambda': 0.5,
        'mkd_lambda': 0.5,
        'kd_temperature': 0.5,
        'num_classes': args.num_classes,
    }

    # Load and prepare dataset (FedBeat 방식 유지)
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    val_dataset = wrap_as_local_dataset(val_dataset, tag='val', dataset_type='ham10000')
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    print('Original data distribution (clean labels count):')
    print(f"→ Total Train samples: {len(train_dataset)}")
    print(f"→ Total Val samples  : {len(val_dataset)}")
    print(f"→ Total Test samples : {len(test_dataset)}")

    model_dir = os.path.join(args.result_dir, args.dataset, f'iid_{args.iid}')
    os.makedirs(model_dir, exist_ok=True)

    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)

    # Subset → local_dataset 형태로 래핑
    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)

    # 전체 평가용 test_loader
    test_loader = DataLoader(combine_data(clients_test_dataset_list, args.dataset),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 클라이언트별 데이터 수 출력 및 DataLoader 구성
    print("\n[Client-wise Dataset Sizes]")
    clients_train_loader_list = []
    clients_test_loader_list = []

    # def get_label_from_local_dataset(dataset, index):
    #     return dataset.local_noisy_labels[index]

    for i in range(args.num_clients):
        print(f"Client {i+1}:")
        print(f"  Train samples: {len(clients_train_dataset_list[i])}")
        print(f"  Test samples : {len(clients_test_dataset_list[i])}")

        train_loader = DataLoader(clients_train_dataset_list[i], batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        
        # sampler = ImbalancedDatasetSampler(
        #     dataset=clients_train_dataset_list[i],  # 또는 local_dataset instance
        #     callback_get_label=get_label_from_local_dataset
        # )

        # train_loader = DataLoader(clients_train_dataset_list[i], sampler=sampler, batch_size=args.batch_size,
        #                         shuffle=False, num_workers=args.num_workers)
        test_loader_i = DataLoader(clients_test_dataset_list[i], batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

        clients_train_loader_list.append(train_loader)
        clients_test_loader_list.append(test_loader_i)

    global_model = ReinsDinoVisionTransformer(**_small_variant)
    global_model.load_state_dict(torch.hub.load('facebookresearch/dinov2', _small_dino).state_dict(), strict=False)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.to(device)
    global_model.eval()
    global_model.fusion_alpha = nn.Parameter(torch.tensor(0.5, device=device), requires_grad=True)

    # 클라이언트 초기화
    client_model_list, optimizer_list, scheduler_list = [], [], []
    for _ in range(args.num_clients):
        model = LoRAFusedDualReinsDinoVisionTransformer(**_small_variant)
        model.load_state_dict(global_model.state_dict(), strict=False)
        model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
        model.to(device)

        model.reins2.load_state_dict(global_model.reins.state_dict())

        client_model_list.append(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
            int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)
        ])
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    # Pretrain reins1
    print("\n[Step 3] Local Pretraining (reins1)")
    for client_idx in range(args.num_clients):
        dataset_len = len(clients_train_dataset_list[client_idx])
        print(f"Client {client_idx+1} has {dataset_len} training samples.")
        model = client_model_list[client_idx]
        model.eval()
        model.train1()
        optimizer = optimizer_list[client_idx]
        scheduler = scheduler_list[client_idx]

        # ✅ 사전 정의된 DataLoader 사용
        train_loader = clients_train_loader_list[client_idx]
        test_loader_i = clients_test_loader_list[client_idx]

        for ep in range(args.round1):
            for inputs, targets, _, _ in train_loader:  # ← 4개 항목 unpack
                inputs, targets = inputs.to(device), targets.to(device)
                feats = model.forward_features1(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            if ep == args.round1 - 1:
                acc = util.validation_accuracy(model, test_loader, device, mode='rein1')
                print(f"[Client {client_idx+1}] Epoch {ep+1}: Test Acc = {acc * 100:.2f}%")

    # Step 3.5 Aggregation 후 평가
    print("\n[Step 3.5] Aggregation of reins1 and linear head")
    average_reins1(global_model, client_model_list)

    acc = validation_accuracy(global_model, test_loader, device, mode='rein')
    print(f"[Eval after Step 3.5] Test Accuracy: {acc * 100:.2f}%")

    # Federated Refinement
    for round in range(args.round3):
        print(f"\n→ Global Round {round + 1}/{args.round3}")
        step = (args.tau) / args.round3
        # mix_ratio = 1.0 - step * round
        mix_ratio = 0.5 - step * round
        
        for client_idx in range(args.num_clients):
            for local_idx in range(args.local_ep):
                client = client_model_list[client_idx]
                optimizer = optimizer_list[client_idx]
                train_loader = clients_train_loader_list[client_idx]
                client_update_step1(client, optimizer, train_loader, config, mix_ratio)
            acc = validation_accuracy(client, test_loader, device, mode='rein2')
            print(f"[Step1 ep{local_idx+1} Client{client_idx+1} Rein2 Eval after Round {round + 1}] Test Accuracy: {acc * 100:.2f}%")

        average_reins2(global_model, client_model_list)
        acc = validation_accuracy(global_model, test_loader, device, mode='rein')

        # Step 2: Tune A^c using refined global predictions
        for client_idx in range(args.num_clients):
            client = client_model_list[client_idx]
            optimizer = optimizer_list[client_idx]
            train_loader = clients_train_loader_list[client_idx]
            client_update_step2(client, global_model, optimizer, train_loader, config, mix_ratio)
            scheduler_list[client_idx].step()

            acc = validation_accuracy(client, test_loader, device, mode='dual')
            print(f"[Step2 Client{client_idx+1} dual Eval after Round {round + 1}] Test Accuracy: {acc * 100:.2f}%")

        if (round + 1) % args.print_freq == 0 or round == args.round3 - 1:
            average_reins1(global_model, client_model_list)
            acc = validation_accuracy(global_model, test_loader, device, mode='rein')
            print(f"[Global model Eval after Round {round + 1}] Test Accuracy: {acc * 100:.2f}%")

    # 모델 저장 및 최종 평가
    torch.save(global_model.state_dict(), os.path.join(model_dir, 'final_global_model.pth'))
    print("→ Final model saved.")
    acc = validation_accuracy(global_model, test_loader, device, mode='rein')
    print(f"[Final Eval] Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
