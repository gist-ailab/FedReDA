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
from rein import LoRAFusedDualReinsDinoVisionTransformer, ReinsDinoVisionTransformer, SelectiveReinsDinoVisionTransformer
from update import average_reins, client_update_with_refinement, client_update_with_rce, average_weights, PAPA_average_reins
import util
from imbalanced_dataset_sampler.torchsampler.imbalanced import ImbalancedDatasetSampler
import copy

def main(args):
    device = torch.device(f"cuda:{args.gpu}")
    config = {
        'device': device,
        'kd_lambda1': 10.0,
        'kd_lambda2': 10.0,
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
    val_subsets = split_equally_across_clients(val_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)

    # Subset → local_dataset 형태로 래핑
    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_val_dataset_list = wrap_subsets_to_local_dataset(val_subsets, val_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)

    # 전체 평가용 test_loader
    # train_loader = DataLoader(combine_data(clients_train_dataset_list, args.dataset),
    #                         batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(combine_data(clients_test_dataset_list, args.dataset),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    # 클라이언트별 데이터 수 출력 및 DataLoader 구성
    print("\n[Client-wise Dataset Sizes]")
    clients_train_loader_list = []
    clients_train_class_num_list = []
    clients_val_loader_list = []
    clients_test_loader_list = []

    def get_label_from_local_dataset(dataset, index):
        return dataset.local_noisy_labels[index]

    
    for i in range(args.num_clients):
        print(f"Client {i+1}:")
        print(f"  Train samples: {len(clients_train_dataset_list[i])}")
        print(f"  Test samples : {len(clients_test_dataset_list[i])}")

        train_loader = DataLoader(clients_train_dataset_list[i], batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(clients_val_dataset_list[i], batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        
        class_num_list = [0 for _ in range(args.num_classes)]
        for j in range(len(clients_train_dataset_list[i])):
            # print("Dataset input: ",clients_train_dataset_list[i][j])
            class_num_list[int(clients_train_dataset_list[i][j][1])] += 1
        class_num_list = torch.cuda.FloatTensor(class_num_list)

        class_p_list = class_num_list / class_num_list.sum()
        class_p_list = torch.log(class_p_list)
        class_p_list = class_p_list.view(1,-1)
        
        # sampler = ImbalancedDatasetSampler(
        #     dataset=clients_train_dataset_list[i],  # 또는 local_dataset instance
        #     callback_get_label=get_label_from_local_dataset
        # )

        # train_loader = DataLoader(clients_train_dataset_list[i], sampler=sampler, batch_size=args.batch_size,
        #                         shuffle=False, num_workers=args.num_workers)
        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p_list)
    # print(clients_train_class_num_list)
    # exit()

    global_model = ReinsDinoVisionTransformer(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    global_model.reins2 = copy.deepcopy(global_model.reins)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.linear_rein2 = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.linear_norein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.to(device)
    global_model.eval()
    global_model.train()
    
    print("GLOBAL MODEL NAMED PARAMETER")
    for n, p in global_model.named_parameters():
        if p.requires_grad == True:
            print(n)

    # 클라이언트 초기화
    client_model_list, optimizer_list, scheduler_list = [], [], []
    for _ in range(args.num_clients):
        model = copy.deepcopy(global_model)
        model.to(device)
        model.eval()
        model.train()
        
        client_model_list.append(model)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
        # optimizer_norein = torch.optim.AdamW(model.linear_norein.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
        optimizer_rein = torch.optim.AdamW(list(model.linear_rein.parameters())+list(model.reins.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_rein2 = torch.optim.AdamW(list(model.linear_rein2.parameters())+list(model.reins2.parameters()), lr=args.lr_f, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum)
        # scheduler_norein = torch.optim.lr_scheduler.MultiStepLR(optimizer_norein, milestones=[
        #     int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)
        # ])
        scheduler_rein = torch.optim.lr_scheduler.MultiStepLR(optimizer_rein, milestones=[
            int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)
        ])
        scheduler_rein2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_rein2, milestones=[
            int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)
        ])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(0.1*args.round3), eta_min=0.1*args.lr_f)
        # optimizer_list.append(optimizer)
        optimizer_list.append([optimizer_rein, optimizer_rein2])
        scheduler_list.append([scheduler_rein, scheduler_rein2])
    print("CLIENT MODEL NAMED PARAMETER")
    for n, p in client_model_list[0].named_parameters():
        if p.requires_grad == True:
            print(n)

    # Pretrain clients
    print("\n[Step 3] Local Pretraining (reins1)")
    for client_idx in range(args.num_clients):
        dataset_len = len(clients_train_dataset_list[client_idx])
        print(f"Client {client_idx+1} has {dataset_len} training samples.")
        model = client_model_list[client_idx]
        model.eval()
        model.train()
        optimizer_rein, optimizer_rein2 = optimizer_list[client_idx]

        # ✅ 사전 정의된 DataLoader 사용
        train_loader = clients_train_loader_list[client_idx]

        focal_loss_fn = FocalLossWithLogitAdjustment(
            gamma=2.0,
            class_log_prior=clients_train_class_num_list[client_idx],  # 기존 log-p list
            alpha=None, 
            reduction='none'
        )
        for ep in range(args.round1):
            for inputs, targets, _, _ in train_loader:  # ← 4개 항목 unpack
                inputs, targets = inputs.to(device), targets.to(device)
                
                # with torch.no_grad():
                #     feats_norein = model.forward_features_no_rein(inputs)[:, 0, :]
                # logits_norein = model.linear_norein(feats_norein)
                # pred_norein = logits_norein.argmax(dim=1)
                
                feats = model.forward_features(inputs)[:, 0, :]
                logits_rein = model.linear_rein(feats)
                pred_rein = logits_rein.argmax(dim=1)
                
                feats2 = model.forward_features2(inputs)[:, 0, :]
                logits_rein2 = model.linear_rein2(feats2)
                pred_rein2 = logits_rein2.argmax(dim=1)
                
                # logits_norein = logits_norein + clients_train_class_num_list[client_idx]
                # logits_rein = logits_rein + clients_train_class_num_list[client_idx]
                # logits_rein2 = logits_rein2 + clients_train_class_num_list[client_idx]
                
                with torch.no_grad():
                    # linear_accurate_norein = (pred_norein==targets)
                    linear_accurate_rein = (pred_rein==targets)
                
                # loss_norein = F.cross_entropy(logits_norein, targets)
                loss_rein = F.cross_entropy(logits_rein, targets, reduction='none')
                loss_rein2 = F.cross_entropy(logits_rein2, targets, reduction='none')

                # loss_norein = focal_loss_fn(logits_norein, targets)
                # loss_rein = focal_loss_fn(logits_rein, targets)
                # loss_rein2 = focal_loss_fn(logits_rein2, targets)
                
                loss = (linear_accurate_rein*loss_rein2).mean() + loss_rein.mean()
                # loss = loss_rein
                # optimizer_norein.zero_grad()
                optimizer_rein.zero_grad()
                optimizer_rein2.zero_grad()
                loss.backward()
                # optimizer_norein.step()
                optimizer_rein.step()
                optimizer_rein2.step()

            if ep == args.round1 - 1:
                bacc, nacc = util.validation_accuracy(model, test_loader, device, mode='rein2')
                print(f"[Client {client_idx+1}] Epoch {ep+1}: Rein Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
                
                # bacc, nacc = util.validation_accuracy(model, test_loader, device, mode='no_rein')
                # print(f"[Client {client_idx+1}] Epoch {ep+1}: No Rein Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")

    # Step 3.5 Aggregation 후 평가
    print("\n[Step 3.5] Aggregation of reins and linear head")
    average_reins(global_model, client_model_list)
    bacc, nacc = validation_accuracy(global_model, test_loader, device, mode='rein2')
    print(f"[Eval after Step 3.5] Rein Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
    bacc, nacc = util.validation_accuracy(global_model, test_loader, device, mode='no_rein')
    print(f"[Eval after Step 3.5] No Rein Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")

    # exit()
    # Federated Refinement
    class_total = dict()
    dynamic_boost_list = [torch.zeros(args.num_classes, device=device) for _ in range(args.num_clients)]
    Final_acc_Rein1 = 0
    Final_acc_Rein2 = 0
    for i in range(args.num_classes):
        class_total[i] = 1
    class_total = [1 for i in range(args.num_classes)]
    for round in range(args.round3):
        local_weights_list = []
        print(f"\n→ Global Round {round + 1}/{args.round3}")
        mix_ratio = args.mixhigh - (args.mixhigh-args.mixlow)*round/args.round3
        
        for client_idx in range(args.num_clients):
            for _ in range(args.local_ep):
                client = client_model_list[client_idx]
                optimizers = optimizer_list[client_idx]
                train_loader = clients_train_loader_list[client_idx]
                class_total, dynamic_boost = client_update_with_refinement(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, dynamic_boost=dynamic_boost_list[client_idx])
                # class_total, dynamic_boost = client_update_with_refinement(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, clients_train_class_num_list[client_idx],dynamic_boost=dynamic_boost_list[client_idx])
                # class_total, dynamic_boost = client_update_with_rce(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, clients_train_class_num_list[client_idx],dynamic_boost=dynamic_boost_list[client_idx])
                dynamic_boost_list[client_idx] = 0.9*dynamic_boost_list[client_idx] + 0.1*dynamic_boost
            
            bacc, nacc = validation_accuracy(client, test_loader, device, mode='rein')
            print(f"[Client{client_idx+1} Rein Eval after Round {round + 1}] BAccuracy: {bacc * 100:.2f}% NAccuracy: {nacc * 100:.2f}%")
            bacc, nacc = validation_accuracy(client, test_loader, device, mode='rein2')
            print(f"[Client{client_idx+1} Rein2 Eval after Round {round + 1}] BAccuracy: {bacc * 100:.2f}% NAccuracy: {nacc * 100:.2f}%")
            scheduler_list[client_idx][0].step()
            scheduler_list[client_idx][1].step()
            # scheduler_list[client_idx][2].step()
            local_weights_list.append(copy.deepcopy(client.state_dict()))
            
        average_reins(global_model, client_model_list)
        # PAPA_average_reins(global_model, client_model_list)
        # average_reins_with_transport(global_model, client_model_list)
        
        # global_weights = average_weights(local_weights_list)
        # global_model.load_state_dict(global_weights)
        
        # for client in client_model_list:
        #     client.reins.load_state_dict(global_model.reins.state_dict())
        #     client.train()
        
        bacc1, nacc1 = validation_accuracy(global_model, test_loader, device, mode='rein')
        print(f"[Global Eval Rein after Round {round + 1}] BAccuracy: {bacc1 * 100:.2f}% NAccuracy: {nacc1 * 100:.2f}%") 
        bacc2, nacc2 = validation_accuracy(global_model, test_loader, device, mode='rein2')
        print(f"[Global Eval Rein2 after Round {round + 1}] BAccuracy: {bacc2 * 100:.2f}% NAccuracy: {nacc2 * 100:.2f}%") 

        if round+1 > args.round3-10:
            Final_acc_Rein1 += bacc1
            Final_acc_Rein2 += bacc2

    # 모델 저장 및 최종 평가
    # torch.save(global_model.state_dict(), os.path.join(model_dir, 'final_global_model.pth'))
    # print("→ Final model saved.")
    # bacc, nacc = validation_accuracy(global_model, test_loader, device, mode='rein2')
    Final_acc_Rein1 /= 10
    Final_acc_Rein2 /= 10
    print(f"[Final Eval] Test Accuracy: {Final_acc_Rein1 * 100:.2f}%")
    print(f"[Final Eval] Test Accuracy: {Final_acc_Rein2 * 100:.2f}%")

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
