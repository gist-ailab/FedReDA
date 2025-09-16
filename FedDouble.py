
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
import logging
from collections import defaultdict

# --- FedNoRo 모듈을 임포트하기 위해 경로 추가 ---
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)
# --------------------------------------------

# --- 기존 FedLNL 의존성 ---
from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from update import average_reins
from sklearn.metrics import balanced_accuracy_score, accuracy_score
# ---------------------------

# --- FedNoRo 데이터셋 관련 모듈 임포트 ---
from dataset.dataset import get_dataset
from utils.utils import add_noise
# --------------------------------------

# --- 로거 설정 ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # 파일 핸들러
    file_handler = logging.FileHandler('FedDouble.txt', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

def calculate_accuracy(model, dataloader, device, mode='rein'):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            if mode == 'rein':
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'rein2':
                feats = model.forward_features2(inputs)[:, 0, :]
                logits = model.linear_rein2(feats)
            elif mode == 'fusion':
                feats = model.forward_fusion2(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            else: # Default to rein
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    
    model.train() # Set model back to train mode
    return bacc, acc

def ema_update(target_module, source_module, alpha=0.9):
    for (name_t, param_t), (name_s, param_s) in zip(
        target_module.named_parameters(), source_module.named_parameters()
    ):
        assert name_t == name_s, "Parameter name mismatch!"
        param_t.data.mul_(alpha).add_(param_s.data * (1 - alpha))

# FedNoRo의 dict_users를 DataLoader에 맞게 변환하기 위한 헬퍼 클래스
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        index = self.idxs[item]
        return image, label, index

# 모델 정의 수정
class FedDoubleModel(ReinsDinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # linear layers will be added dynamically later

    def forward_fusion1(self, x, masks=None):
        # Reins1+Reins2 -> logit
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x1 = blk(x)
            x = self.reins.forward(x1, idx, batch_first=True, has_cls_token=True)
            x = x + self.reins2.forward(x1, idx, batch_first=True, has_cls_token=True)
        return x
    
    def forward_fusion2(self, x, masks=None):
        # Reins1->Reins2 -> logit
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
            x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x
    
    def forward_fusion3(self, x):
        return self.forward_features(x) + self.forward_features2(x)

def main(args):
    setup_logging()
    logging.info("="*50)
    logging.info("="*50)
    logging.info("Starting FedDouble training process...")
    
    device = torch.device(f"cuda:{args.gpu}")
    
    # =============================================================================
    # Step 0: Global모델 및 노이즈 데이터셋 세팅, 각 client모델 및 데이터셋 선언
    # =============================================================================
    logging.info("="*50)
    logging.info("Step 0: Initializing models, datasets, and clients")
    logging.info("="*50)

    args.num_users = args.num_clients
    args.n_clients = args.num_clients
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_train_noisy
    logging.info(f"Loaded dataset '{args.dataset}' with {len(dict_users)} clients.")
    logging.info(f"Noisy train data distribution: {Counter(dataset_train.targets)}")
    
    # 각 클라이언트별 soft label 캐시 (sample_idx 기반)
    client_soft_label_cache = [defaultdict(lambda: None) for _ in range(args.num_clients)]
    EMA_ALPHA = 0.9  # 지수이동평균 alpha
    
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    clients_train_loader_list = []
    clients_train_class_num_list = []
    for i in range(args.num_clients):
        client_indices = dict_users[i]
        client_dataset = DatasetSplit(dataset_train, client_indices)
        train_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        class_num_list = [0 for _ in range(args.n_classes)]
        for idx in dict_users[i]:
            class_num_list[int(dataset_train.targets[idx])] += 1
        class_num_tensor = torch.cuda.FloatTensor(class_num_list)
        class_p_list = class_num_tensor / class_num_tensor.sum()
        class_p_list = torch.log(class_p_list)
        class_p_list = class_p_list.view(1, -1)
        
        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p_list)

    # Global model은 rein2만 활용
    # Local model은 local adapter가 rein1, global adapter가 rein2를 사용
    global_model = FedDoubleModel(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.reins2 = copy.deepcopy(global_model.reins) # rein2 어댑터 추가
    global_model.linear_rein2 = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.to(device)

    client_model_list = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    
    logging.info("Step 0 Finished: Initialization complete.")

    # =============================================================================
    # Step 1: 클라이언트 모델들 epoch1만큼 사전학습. Rein adapter만 학습.
    # =============================================================================
    logging.info("="*50)
    logging.info("Step 1: Pre-training client 'rein' adapters for epoch1")
    logging.info("="*50)
    
    for epoch in range(args.round1):
        for client_idx in range(args.num_clients):
            model = client_model_list[client_idx]
            model.train()
            
            # rein 어댑터와 linear_rein만 학습하도록 설정
            for name, param in model.named_parameters():
                if 'reins.' in name or 'linear_rein.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]
            
            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    feats = model.forward_features(inputs)[:, 0, :]
                    logits = model.linear_rein(feats)
                    
                    logits = logits + 0.5*class_list
                    
                    # 초기 Warm up 중 noise의 영향을 최소화 하기 위해 loss기반 filtering 진행
                    ce_losses = F.cross_entropy(logits, targets, reduction='none')
                    sorted_loss, indices = torch.sort(ce_losses)
                    keep_ratio = args.warmup_keep_ratio if hasattr(args, 'warmup_keep_ratio') else 0.8
                    keep_num = int(len(indices) * keep_ratio)
                    select_idx = indices[:keep_num]
                    
                    loss = ce_losses[select_idx].mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        logging.info(f"epoch {epoch} pre-training finished.")

    # =============================================================================
    # Step 2: 클라이언트 모델들 어댑터를 averaging하여 global 모델 어댑터에 초기화
    # =============================================================================
    logging.info("="*50)
    logging.info("Step 2: Averaging pre-trained 'rein' adapters to global model")
    logging.info("="*50)
    
    # local rein1 -> global rein2
    with torch.no_grad():
        reins_named_params = {}
        for name, _ in client_model_list[0].reins.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
            reins_named_params[name] = stacked.mean(dim=0)
            
        for name, param in global_model.reins2.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])
                
        weight_sum = sum(client.linear_rein.weight.data for client in client_model_list)
        bias_sum = sum(client.linear_rein.bias.data for client in client_model_list)
        
        global_model.linear_rein2.weight.data.copy_(weight_sum / len(client_model_list))
        global_model.linear_rein2.bias.data.copy_(bias_sum / len(client_model_list))
    
    bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
    logging.info(f"Global Model after Step 2 - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")

    # =============================================================================
    # Main Training Loop (Epochs 3)
    # =============================================================================
    for epoch in range(args.round3):
        logging.info("="*80)
        logging.info(f"Main Training Loop: Starting Epoch {epoch + 1}/{args.round3}")
        logging.info("="*80)

        # =============================================================================
        # Step 3: 클라이언트 모델 별 rein2 어댑터를 학습되지 않은 상태로 선언 및 초기화하고 
        #         rein어댑터는 글로벌 모델의 rein 어댑터로 선언하고 rein어댑터만은 freeze해서 학습되지 않게 함.
        # =============================================================================
        logging.info(f"--- [Epoch {epoch+1}] Step 3: Initializing/Freezing Adapters ---")
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            
            # Local rein 어댑터 재초기화
            client_model.reins = global_model.reins
            # ema_update(client_model.reins, global_model.reins2, alpha=0.9)
            client_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
            # with torch.no_grad():
            #     client_model.linear_rein.weight.data.mul_(0.9).add_(
            #         global_model.linear_rein2.weight.data * 0.1
            #     )
            #     client_model.linear_rein.bias.data.mul_(0.9).add_(
            #         global_model.linear_rein2.bias.data * 0.1
            #     )
            
            # Global rein2 -> Local rein2 
            client_model.reins2.load_state_dict(global_model.reins2.state_dict())
            client_model.linear_rein2.load_state_dict(global_model.linear_rein2.state_dict())

            # rein 학습, rein2 동결 설정
            for name, param in client_model.named_parameters():
                if 'reins.' in name or 'linear_rein.' in name:
                    param.requires_grad = True
                elif 'reins2.' in name or 'linear_rein2.' in name:
                    param.requires_grad = False
                else: # backbone
                    param.requires_grad = False
        logging.info(f"--- [Epoch {epoch+1}] Step 3 Finished ---")

        # =============================================================================
        # Step 4: 각 클라이언트를 각각의 데이터셋으로 학습 (rein만 업데이트)
        # =============================================================================
        logging.info(f"--- [Epoch {epoch+1}] Step 4: Training 'rein' adapter using fusion ---")
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            client_model.train()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, client_model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]
            
            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    fusion_feats = client_model.forward_fusion2(inputs)[:, 0, :]
                    logits = client_model.linear_rein(fusion_feats)
                    
                    logits = logits + 0.5*class_list
                    
                    loss = F.cross_entropy(logits, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            logging.info(f"--- Client {client_idx} 'rein2' training finished.")
            bacc, acc = calculate_accuracy(client_model, test_loader, device, mode='fusion')
            logging.info(f"Client Model after Epoch {epoch+1} - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
        logging.info(f"--- [Epoch {epoch+1}] Step 4 Finished ---")

        # =============================================================================
        # Step 5: 클라이언트 모델의 rein의 freeze를 해제.
        # =============================================================================
        logging.info(f"--- [Epoch {epoch+1}] Step 5: Unfreezing 'rein2' adapter ---")
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            for name, param in client_model.named_parameters():
                    if 'reins.' in name or 'linear_rein.' in name:
                        param.requires_grad = False
                    elif 'reins2.' in name or 'linear_rein2.' in name:
                        param.requires_grad = True
                    else: # backbone
                        param.requires_grad = False
            logging.info(f"--- [Epoch {epoch+1}] Step 5 Finished ---")

        # =============================================================================
        # Step 6: Teacher-Student 기반으로 rein 어댑터 학습
        # =============================================================================
        logging.info(f"--- [Epoch {epoch+1}] Step 6: Training 'rein' via Distillation ---")
        T = 2.0  # Temperature for softening
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            
            # Teacher 모델은 현재 client_model의 복사본 (그래디언트 흐름 없음)
            teacher_model = copy.deepcopy(client_model)
            teacher_model.eval()

            # rein만 학습하도록 옵티마이저 재설정
            optimizer = torch.optim.AdamW(
                list(client_model.reins2.parameters()) + list(client_model.linear_rein2.parameters()),
                lr=args.lr
            )
            
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]
            
            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 현재 배치 인덱스 가져오기
                    batch_indices = loader.dataset.idxs
                    
                    # Teacher Logits (rein + rein2, no_grad)
                    with torch.no_grad():
                        teacher_feats = teacher_model.forward_fusion2(inputs)[:, 0, :]
                        teacher_logits = teacher_model.linear_rein(teacher_feats)
                        teacher_logits = teacher_logits+0.5*class_list
                        teacher_probs = F.softmax(teacher_logits / T, dim=1)
                        
                        # 라벨 노이즈 가능성 계산: teacher confidence 기준
                        confidence_threshold = max(0.9 - epoch * 0.004, 0.5)
                        top1_confidence, _ = teacher_probs.max(dim=1)  # [B]
                        noise_mask = top1_confidence < confidence_threshold
                        
                        refined_targets_list = []
                        for j in range(len(targets)):
                            idx = batch_indices[j].item()
                            prev_soft = client_soft_label_cache[client_idx][idx]
                            curr_soft = teacher_probs[j]

                            if prev_soft is None:
                                updated_soft = curr_soft.detach().to(device=inputs.device, dtype=torch.float)
                            else:
                                if noise_mask[j]:  # 노이즈로 추정됨
                                    updated_soft = EMA_ALPHA * prev_soft + (1 - EMA_ALPHA) * curr_soft.detach()
                                else:  # clean으로 추정되면 이전 soft 유지
                                    updated_soft = prev_soft.detach()
                                    
                            # 2. blending weight (alpha) 계산
                            conf = teacher_probs[j].max()
                            alpha = conf.clamp(0.3, 0.8)  # scalar

                            one_hot = F.one_hot(targets[j], num_classes=args.num_classes).float()
                            refined_target = alpha * one_hot + (1 - alpha) * updated_soft
                            
                            # 캐시 갱신
                            client_soft_label_cache[client_idx][idx] = updated_soft
                            refined_targets_list.append(refined_target.unsqueeze(0))

                        refined_targets = torch.cat(refined_targets_list, dim=0)  # [B, C]
                    
                    # Student Logits (rein only)
                    student_feats = client_model.forward_features2(inputs)[:, 0, :]
                    student_logits = client_model.linear_rein2(student_feats)
                    student_logits = student_logits+0.5*class_list
                    student_log_probs = F.log_softmax(student_logits / T, dim=1)
                    
                    # Loss Calculation
                    kl_loss = kl_loss_fn(student_log_probs, teacher_probs) * (T ** 2)
                    if noise_mask.logical_not().any():
                        # clean_ce_loss = F.cross_entropy(
                        #     student_logits[noise_mask.logical_not()],
                        #     targets[noise_mask.logical_not()]
                        # )
                        
                        refined_target_subset = refined_targets[noise_mask.logical_not()]
                        student_log_prob_subset = F.log_softmax(student_logits[noise_mask.logical_not()], dim=1)
                        
                        clean_ce_loss = F.kl_div(
                            student_log_prob_subset,
                            refined_target_subset,
                            reduction='batchmean'
                        )
                        
                    else:
                        clean_ce_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
                        
                    loss = kl_loss + clean_ce_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            logging.info(f"--- Client {client_idx} 'rein' distillation finished.")
        logging.info(f"--- [Epoch {epoch+1}] Step 6 Finished ---")

        # =============================================================================
        # Step 7: 학습된 클라이언트 모델의 rein 어댑터를 모아서 global 모델 업데이트
        # =============================================================================
        logging.info(f"--- [Epoch {epoch+1}] Step 7: Averaging 'rein' adapters to global model ---")

        with torch.no_grad():
            reins_named_params = {}
            for name, _ in client_model_list[0].reins2.named_parameters():
                stacked = torch.stack([dict(client.reins2.named_parameters())[name].data for client in client_model_list])
                reins_named_params[name] = stacked.mean(dim=0)
                
            for name, param in global_model.reins2.named_parameters():
                if name in reins_named_params:
                    param.data.copy_(reins_named_params[name])
                    
            weight_sum = sum(client.linear_rein2.weight.data for client in client_model_list)
            bias_sum = sum(client.linear_rein2.bias.data for client in client_model_list)
            
            global_model.linear_rein2.weight.data.copy_(weight_sum / len(client_model_list))
            global_model.linear_rein2.bias.data.copy_(bias_sum / len(client_model_list))
        
        bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
        logging.info(f"Global Model after Epoch {epoch+1} - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
        logging.info(f"--- [Epoch {epoch+1}] Step 7 Finished ---")

    logging.info("="*50)
    logging.info("FedDouble training process finished.")
    logging.info("="*50)


if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
