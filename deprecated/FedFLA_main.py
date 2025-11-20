import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import copy
import sys
from collections import defaultdict
from torch import nn, autograd

# --- FedNoRo 모듈을 임포트하기 위해 경로 추가 ---
# 이 코드는 FedNoRo의 utils와 dataset 폴더를 찾을 수 있도록 합니다.
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)
# ---------------------------------------------

# --- 기존 FedLNL 의존성 ---
from util.options import args_parser
from dino_variant import _small_dino, _small_variant
from rein import LoRAFusedDualReinsDinoVisionTransformer, ReinsDinoVisionTransformer, SelectiveReinsDinoVisionTransformer
from update import average_reins, average_reins1, client_update_with_refinement, average_weights, PAPA_average_reins
import util # util.validation_accuracy를 위해 유지
# ---------------------------

# --- FedNoRo 데이터셋 관련 모듈 임포트 ---
from dataset.dataset import get_dataset
from utils.utils import add_noise
# ---------------------------------------

T = 0.8

def client_update_clean(client_model, global_model, optimizers, loader, config, mix_ratio, class_total, class_p_list=None):
    num_classes = config['num_classes']
    device = config['device']

    client_model.eval()
    client_model.train()

    optimizer_rein = optimizers
    total_loss = 0

    ce_vals, clean_ratios = [], []
    correct1, total1 = 0, 0
    class_total = defaultdict(int)
    class_correct_rein = defaultdict(int)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # === Guide model ===
        # with torch.no_grad():
        #     feats_global = global_model.forward_features(inputs)[:, 0, :]
        #     logits_global = global_model.linear_rein(feats_global)
        #     pred_global = logits_global.argmax(dim=1)
            
        # === Client model ===        
        feats_rein = client_model.forward_features(inputs)[:, 0, :]
        logits_rein = client_model.linear_rein(feats_rein)
        pred_rein = logits_rein.argmax(dim=1)
        
        if class_p_list is not None:
            # logits_norein = logits_norein + class_p_list
            logits_rein = logits_rein + class_p_list
        
        # with torch.no_grad():
            # guide_norein = logits_norein.argmax(dim=1)
            # guide_rein = logits_rein.argmax(dim=1)

        with torch.no_grad():
            linear_accurate = (pred_rein == targets)
        
        for i in range(targets.size(0)):
            label = targets[i].item()
            pred_label_rein = pred_rein[i].item()

            class_total[label] += 1

            if label == pred_label_rein:
                class_correct_rein[label] += 1

        loss = F.cross_entropy(logits_rein/T, targets, reduction='none').mean()

        optimizer_rein.zero_grad()
        loss.backward()
        optimizer_rein.step()

        ce_vals.append(loss.mean().item())
        clean_ratios.append(linear_accurate.float().mean().item())
        
        correct1 += (pred_rein == targets).sum().item()
        total1 += targets.size(0)

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

    return class_total

def client_update_noisy(client_model, global_model, optimizers, loader, config, mix_ratio, class_total, class_p_list=None):
    num_classes = config['num_classes']
    device = config['device']

    client_model.eval()
    client_model.train()

    optimizer_rein = optimizers
    total_loss = 0

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
        feats_rein = client_model.forward_features(inputs)[:, 0, :]
        logits_rein = client_model.linear_rein(feats_rein)
        pred_rein = logits_rein.argmax(dim=1)
        
        if class_p_list is not None:
            # logits_norein = logits_norein + class_p_list
            logits_rein = logits_rein + class_p_list
        
        # with torch.no_grad():
            # guide_norein = logits_norein.argmax(dim=1)
            # guide_rein = logits_rein.argmax(dim=1)

        # === Soft label construction ===
        with torch.no_grad():
            # refined_targets = 0.6 * logits_rein + 0.4 * F.one_hot(targets, num_classes=args.num_classes).float()
            # refined_targets = torch.softmax(refined_targets, dim=1)
            # refined_targets = refined_targets.argmax(dim=1)
            
            # linear_accurate = (pred_global == refined_targets)
            linear_accurate = (pred_global == targets)
            linear_accuracte_local = (pred_rein == targets)

        for i in range(targets.size(0)):
            label = targets[i].item()
            pred_label_rein = pred_rein[i].item()

            class_total[label] += 1

            if label == pred_label_rein:
                class_correct_rein[label] += 1

        loss_ce = F.cross_entropy(logits_rein/T, targets, reduction='none')
        loss_soft = F.kl_div(torch.log_softmax(logits_rein/T, dim=-1), torch.softmax(logits_global, dim=-1), reduction='none')
        loss_soft = 10*loss_soft.mean(dim=-1)
        
        loss_agree = (linear_accurate*loss_ce).mean() + loss_ce.mean()
        loss_disagree = loss_soft.mean()
        # print(loss_agree.item())
        # print(loss_disagree.item())
        loss = loss_agree + loss_disagree
        loss = loss_agree

        optimizer_rein.zero_grad()
        loss.backward()
        optimizer_rein.step()

        ce_vals.append(loss.mean().item())
        clean_ratios.append(linear_accurate.float().mean().item())
        
        correct1 += (pred_rein == targets).sum().item()
        total1 += targets.size(0)

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

    return class_total

# FedNoRo의 dict_users를 DataLoader에 맞게 변환하기 위한 헬퍼 클래스
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # FedLNL의 학습 루프는 4개의 값을 반환받으므로, 동일한 형식으로 맞춰줍니다.
        image, label = self.dataset[self.idxs[item]]
        # (inputs, targets, original_index, noisy_label_check)
        return image, label

def main(args):
    device = torch.device(f"cuda:{args.gpu}")
    config = {
        'device': device,
        'kd_lambda1': 10.0,
        'kd_lambda2': 10.0,
        'num_classes': args.num_classes,
    }

    # =============================================================================
    # == FedNoRo 방식으로 데이터셋 로드 및 노이즈 추가 (기존 코드 교체) ==
    # =============================================================================
    print("--- Loading dataset using FedNoRo methodology ---")
    # FedNoRo의 args와 FedLNL의 args가 호환되도록 일부 값 설정
    args.num_users = args.num_clients
    args.n_clients = args.num_clients
    # args.n_type = args.noise_type
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    print(f"FedNoRo train set size: {len(dataset_train)}")
    print(f"FedNoRo test set size: {len(dataset_test)}")
    print(f"FedNoRo user dictionary created for {len(dict_users)} clients.")

    # FedNoRo의 유틸리티를 사용하여 노이즈 추가
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_train_noisy
    print("--- Noise applied to train dataset using FedNoRo methodology ---")
    print(f"Train data distribution (noisy labels): {Counter(dataset_train.targets)}")
    # =============================================================================

    # 전체 평가용 test_loader (FedNoRo 데이터셋 사용)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 클라이언트별 DataLoader 및 클래스 분포 구성
    print("[Client-wise Dataset Configuration]")
    clients_train_loader_list = []
    clients_train_class_num_list = []

    for i in range(args.num_clients):
        client_indices = dict_users[i]
        client_dataset = DatasetSplit(dataset_train, client_indices)
        
        print(f"Client {i+1}: Train samples: {len(client_dataset)}")

        train_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        clients_train_loader_list.append(train_loader)

        # 클래스별 샘플 수 계산
        class_num_list = [0] * args.num_classes
        client_labels = [dataset_train.targets[idx] for idx in client_indices]
        class_counts = Counter(client_labels)
        for c, count in class_counts.items():
            class_num_list[c] = count
        
        class_num_list = torch.cuda.FloatTensor(class_num_list)
        class_p_list = class_num_list / class_num_list.sum()
        class_p_list = torch.log(class_p_list+1e-6)
        class_p_list = class_p_list.view(1, -1)
        clients_train_class_num_list.append(class_p_list)
        # print(class_p_list)

    # =============================================================================
    # == 모델 초기화 및 학습 루프 (이 부분은 기존 FedLNL_main.py와 동일) ==
    # =============================================================================
    noisy_clients_list = [1,6,8,9,10,13,16,17]
    clean_clients_list = [0,2,3,4,5,7,11,12,14,15,18,19]
    
    global_model = ReinsDinoVisionTransformer(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    global_model.reins2 = copy.deepcopy(global_model.reins)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.linear_rein2 = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.to(device)
    global_model.train()
    
    # print("GLOBAL MODEL NAMED PARAMETERS (requires_grad=True)")
    # for n, p in global_model.named_parameters():
    #     if p.requires_grad:
    #         print(n)

    # 클라이언트 초기화
    client_model_list, optimizer_list, scheduler_list = [], [], []
    for _ in range(args.num_clients):
        model = copy.deepcopy(global_model)
        model.to(device)
        model.train()
        
        client_model_list.append(model)
        optimizer_rein = torch.optim.AdamW(list(model.linear_rein.parameters())+list(model.reins.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_rein = torch.optim.lr_scheduler.MultiStepLR(optimizer_rein, milestones=[int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)])
        optimizer_list.append(optimizer_rein)
        scheduler_list.append(scheduler_rein)
        
    # Pretrain clients
    step1_model_path = os.path.join('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints',
                            args.dataset,
                            'step1_main_model_state_dict.pth')
    
    if not os.path.exists(step1_model_path):
        print("[Step 1] Local Pretraining (reins1)")
        for ep in range(args.round1): 
            for client_idx in range(args.num_clients):
                model = client_model_list[client_idx]
                model.train()
                optimizer_rein = optimizer_list[client_idx]
                train_loader = clients_train_loader_list[client_idx]
                for _ in range(args.local_ep):
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        feats = model.forward_features(inputs)[:, 0, :]
                        logits_rein = model.linear_rein(feats)
                            
                        logits_rein = logits_rein + clients_train_class_num_list[client_idx]
                        pred_rein = logits_rein.argmax(dim=1)
                        
                        loss_rein = F.cross_entropy(logits_rein/T, targets, reduction='none')
                        
                        loss = loss_rein.mean()
                        
                        optimizer_rein.zero_grad()
                        loss.backward()
                        optimizer_rein.step()

                # if ep == args.round1 - 1:
                #     bacc, nacc = util.validation_accuracy(model, test_loader, device, mode='rein')
                #     print(f"[Client {client_idx+1}] Epoch {ep+1}: Rein2 Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
            average_reins1(global_model, client_model_list)
            bacc, nacc = util.validation_accuracy(global_model, test_loader, device, mode='rein')
            print(f"[Eval after Aggregation] Rein2 Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
                
        torch.save(global_model.state_dict(), 
                os.path.join('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints',
                                args.dataset,
                                'step1_main_model_state_dict.pth'))
    else:
        model_state_dict = torch.load(step1_model_path)
        global_model.load_state_dict(model_state_dict)
        for client_idx in range(args.num_clients):
            model = client_model_list[client_idx]
            model.load_state_dict(model_state_dict)
    
    class_total = [1] * args.num_classes
    Final_acc_Rein1 = 0
    
    print("[Step 2] Federated Refinement")
    for round_num in range(args.round3):
        print(f"→ Global Round {round_num + 1}/{args.round3}")
        mix_ratio = args.mixhigh - (args.mixhigh - args.mixlow) * round_num / args.round3
        
        for client_idx in range(args.num_clients):
            for _ in range(args.local_ep):
                client = client_model_list[client_idx]
                
                client.reins2 = copy.deepcopy(global_model.reins)
                for param in client.reins2.parameters():
                    param.requires_grad = False
                
                optimizers = optimizer_list[client_idx]
                train_loader = clients_train_loader_list[client_idx]
                if client_idx in clean_clients_list:
                    # class_total = client_update_clean(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, 
                    #                                                         class_p_list=clients_train_class_num_list[client_idx])
                    class_total = client_update_noisy(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, 
                                                                            class_p_list=clients_train_class_num_list[client_idx])
                if client_idx in noisy_clients_list:
                    # class_total = client_update_clean(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, 
                    #                                                         class_p_list=clients_train_class_num_list[client_idx])
                    class_total = client_update_noisy(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, 
                                                                            class_p_list=clients_train_class_num_list[client_idx])
        average_reins1(global_model, client_model_list)
        
        bacc1, nacc1 = util.validation_accuracy(global_model, test_loader, device, mode='rein')
        print(f"[Global Eval Rein after Round {round_num + 1}] BAccuracy: {bacc1 * 100:.2f}% NAccuracy: {nacc1 * 100:.2f}%") 
        
        if round_num + 1 > args.round3 - 10:
            Final_acc_Rein1 += bacc1
            
    Final_acc_Rein1 /= 10
    print(f"[Final Eval] Average BAcc over last 10 rounds (Rein1): {Final_acc_Rein1 * 100:.2f}%")
    
if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)