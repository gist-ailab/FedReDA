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
from update import average_reins, average_reins1, client_update_with_refinement, client_update_with_LA, average_weights, PAPA_average_reins
import util # util.validation_accuracy를 위해 유지
# ---------------------------

# --- FedNoRo 데이터셋 관련 모듈 임포트 ---
from dataset.dataset import get_dataset
from utils.utils import add_noise
# ---------------------------------------

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
    
    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()

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
        class_num_list = client_dataset.get_num_of_each_class(args)
        
        class_num_list = torch.cuda.FloatTensor(class_num_list)
        class_p_list = class_num_list / class_num_list.sum()
        class_p_list = torch.log(class_p_list+1e-6)
        class_p_list = class_p_list.view(1, -1)
        clients_train_class_num_list.append(class_p_list)
        # print(class_p_list)

    # =============================================================================
    # == 모델 초기화 및 학습 루프 (이 부분은 기존 FedLNL_main.py와 동일) ==
    # =============================================================================
    global_model = ReinsDinoVisionTransformer(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    global_model.reins2 = copy.deepcopy(global_model.reins)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.linear_rein2 = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    global_model.linear_norein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
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
        optimizer_rein = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_rein = torch.optim.lr_scheduler.MultiStepLR(optimizer_rein, milestones=[int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)])
        optimizer_list.append(optimizer_rein)
        scheduler_list.append(scheduler_rein)
        
    # Pretrain clients
    step1_model_path = os.path.join('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints',
                            args.dataset,
                            'step1_model_triple_state_dict.pth')
    
    if not os.path.exists(step1_model_path):
        print("[Step 1] Local Pretraining (reins1)")
        for ep in range(args.round1): 
            for client_idx in range(args.num_clients):
                model = client_model_list[client_idx]
                model.train()
                optimizer_rein = optimizer_list[client_idx]
                train_loader = clients_train_loader_list[client_idx]
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    feats_norein = model.forward_features_no_rein(inputs)[:, 0, :]
                    logits_norein = model.linear_norein(feats_norein)
                    
                    feats = model.forward_features(inputs)[:, 0, :]
                    logits_rein = model.linear_rein(feats)
                    
                    feats2 = model.forward_features2(inputs)[:, 0, :]
                    logits_rein2 = model.linear_rein2(feats2)
                    
                    logits_norein = logits_norein + clients_train_class_num_list[client_idx]
                    logits_rein = logits_rein + clients_train_class_num_list[client_idx]
                    logits_rein2 = logits_rein2 + clients_train_class_num_list[client_idx]
                    
                    with torch.no_grad():
                        pred_norein = logits_norein.argmax(dim=1)
                        pred_rein = logits_rein.argmax(dim=1)
                        
                        linear_accurate_norein = (pred_norein == targets)
                        linear_accurate = (pred_rein == targets)
                    
                    # pred_rein = logits_rein.argmax(dim=1)
                    # pred_rein2 = logits_rein2.argmax(dim=1)
                    
                    loss_norein = F.cross_entropy(logits_norein/1.2, targets, reduction='none')
                    loss_rein = F.cross_entropy(logits_rein/1.2, targets, reduction='none')
                    loss_rein2 = F.cross_entropy(logits_rein2/1.2, targets, reduction='none')
                    
                    # loss = loss_norein.mean() + \
                    #        (linear_accurate_norein * loss_rein).mean() + \
                    #        (linear_accurate * loss_rein2).mean()
                           
                    loss = loss_norein.mean() + \
                           loss_rein.mean() + \
                           loss_rein2.mean()
                    
                    optimizer_rein.zero_grad()
                    loss.backward()
                    optimizer_rein.step()

                # if ep == args.round1 - 1:
                #     bacc, nacc = util.validation_accuracy(model, test_loader, device, mode='rein')
                #     print(f"[Client {client_idx+1}] Epoch {ep+1}: Rein2 Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
            average_reins(global_model, client_model_list)
            bacc, nacc = util.validation_accuracy(global_model, test_loader, device, mode='no_rein')
            print(f"[Eval after Aggregation] Rein Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
            bacc, nacc = util.validation_accuracy(global_model, test_loader, device, mode='rein')
            print(f"[Eval after Aggregation] Rein Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
            bacc, nacc = util.validation_accuracy(global_model, test_loader, device, mode='rein2')
            print(f"[Eval after Aggregation] Rein2 Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
            
        # print("[Step 2] Aggregation of reins and linear head")
        # average_reins1(global_model, client_model_list)
        # bacc, nacc = util.validation_accuracy(global_model, test_loader, device, mode='rein')
        # print(f"[Eval after Aggregation] Rein2 Test bAcc = {bacc * 100:.2f}% nAcc = {nacc * 100:.2f}%")
        
        torch.save(global_model.state_dict(), 
                os.path.join('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints',
                                args.dataset,
                                'step1_model_triple_state_dict.pth'))
    else:
        model_state_dict = torch.load(step1_model_path)
        global_model.load_state_dict(model_state_dict)
        for client_idx in range(args.num_clients):
            model = client_model_list[client_idx]
            model.load_state_dict(model_state_dict)
    
    class_total = [1] * args.num_classes
    dynamic_boost_list = [torch.zeros(args.num_classes, device=device) for _ in range(args.num_clients)]
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
                class_total, dynamic_boost = client_update_with_LA(client, global_model, optimizers, train_loader, config, mix_ratio, class_total, 
                                                                           class_p_list=clients_train_class_num_list[client_idx], dynamic_boost=dynamic_boost_list[client_idx])
                dynamic_boost_list[client_idx] = 0.5 * dynamic_boost_list[client_idx] + 0.5 * dynamic_boost
            
            # scheduler_list[client_idx].step()

        average_reins(global_model, client_model_list)
        
        bacc1, nacc1 = util.validation_accuracy(global_model, test_loader, device, mode='no_rein')
        print(f"[Global Eval Linear after Round {round_num + 1}] BAccuracy: {bacc1 * 100:.2f}% NAccuracy: {nacc1 * 100:.2f}%")
        bacc1, nacc1 = util.validation_accuracy(global_model, test_loader, device, mode='rein')
        print(f"[Global Eval Rein after Round {round_num + 1}] BAccuracy: {bacc1 * 100:.2f}% NAccuracy: {nacc1 * 100:.2f}%") 
        bacc1, nacc1 = util.validation_accuracy(global_model, test_loader, device, mode='rein2')
        print(f"[Global Eval Rein after Round {round_num + 1}] BAccuracy: {bacc1 * 100:.2f}% NAccuracy: {nacc1 * 100:.2f}%") 
        
        if round_num + 1 > args.round3 - 10:
            Final_acc_Rein1 += bacc1
            
    Final_acc_Rein1 /= 10
    print(f"[Final Eval] Average BAcc over last 10 rounds (Rein1): {Final_acc_Rein1 * 100:.2f}%")
    
if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)