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
from model_dino import ReinDinov2
from model_lora import LoRADinov2

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
    file_handler = logging.FileHandler('FedCUFIT.txt', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

def calculate_accuracy(model, dataloader, device, mode='LAM'):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            if mode == 'LPM':
                logits = model.forward_LPM(inputs)
            elif mode == 'ALPM':
                logits = model.forward_ALPM(inputs)
            elif mode == 'ILPM':
                logits = model.forward_ILPM(inputs, indexes=[-1])
            elif mode == 'IAM':
                logits = model.forward_IAM(inputs)
            elif mode == 'LAM':
                logits = model.forward_LAM(inputs)
            else: # Default to rein
                print("Error: No Suitable Mode")

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
        true_label = self.dataset.true_labels[self.idxs[item]]
        index = self.idxs[item]
        return image, label, true_label, index

class FedCUFIT(ReinsDinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reins_LPM = copy.deepcopy(self.reins)
        self.reins_IAM = copy.deepcopy(self.reins)
        self.reins_LAM = copy.deepcopy(self.reins)
        self.linear_LPM = nn.Linear(_small_variant['embed_dim'], args.num_classes)
        self.linear_IAM = nn.Linear(_small_variant['embed_dim'], args.num_classes)
        self.linear_LAM = nn.Linear(_small_variant['embed_dim'], args.num_classes)

        # for name, param in self.reins_LPM.named_parameters():
        #     param.requires_grad = False
                        
    def forward_LPM(self, x, masks=None, return_feats=False):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        out = self.linear_LPM(x[:,0,:])
        if return_feats:
            return out, x
        return out
    
    def forward_ALPM(self, x, masks=None, return_feats=False):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins_LPM.forward(x, idx, batch_first=True, has_cls_token=True)
        out = self.linear_LPM(x[:,0,:])
        if return_feats:
            return out, x
        return out
    
    def forward_ILPM(self, x, indexes, masks=None, return_feats=False):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in indexes:
                x = self.reins_LPM.forward(x, idx, batch_first=True, has_cls_token=True)
        out = self.linear_LPM(x[:,0,:])
        if return_feats:
            return out, x
        return out
    
    def forward_IAM(self, x, masks=None, return_feats=False):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins_IAM.forward(x, idx, batch_first=True, has_cls_token=True)
        out = self.linear_IAM(x[:,0,:])
        if return_feats:
            return out, x
        return out
    
    def forward_LAM(self, x, masks=None, return_feats=False):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins_LAM.forward(x, idx, batch_first=True, has_cls_token=True)
        out = self.linear_LAM(x[:,0,:])
        if return_feats:
            return out, x
        return out

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
    
    args.n_classes = args.num_clients
    args.n_clients = args.num_clients
    args.num_users = args.num_clients
    
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


    global_model = FedCUFIT(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False), strict=False)
    global_model.to(device)
    client_model_list = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    
    # 데이터셋 크기는 유지하면서 클라이언트 수만 줄여서 빠른 학습을 하게 하기 위함 (num clients = 5)
    client_model_list = client_model_list[:5]
    args.num_clients = 5
    
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
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]
            
            for _ in range(args.local_ep):
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # logits = model.forward_LAM(inputs)
                    logits = model.forward_ILPM(inputs, [-1]) + 0.5*class_list
                    
                    # print(logits.shape)
                    # print(targets.shape)
                    # 초기 Warm up 중 noise의 영향을 최소화 하기 위해 loss기반 filtering 진행
                    ce_losses = F.cross_entropy(logits, targets, reduction='none')
                    # sorted_loss, indices = torch.sort(ce_losses)
                    # keep_ratio = args.warmup_keep_ratio if hasattr(args, 'warmup_keep_ratio') else 0.8
                    # keep_num = int(len(indices) * keep_ratio)
                    # select_idx = indices[:keep_num]
                    
                    # loss = ce_losses[select_idx].mean()
                    loss = ce_losses.mean()
                    
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
    
    with torch.no_grad():
        reins_named_params = {}
        for name, _ in client_model_list[0].reins_LPM.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
            reins_named_params[name] = stacked.mean(dim=0)
        
        for name, param in global_model.reins_LPM.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])
        
        # for name, param in global_model.reins_IAM.named_parameters():
        #     if name in reins_named_params:
        #         param.data.copy_(reins_named_params[name])
                
        # for name, param in global_model.reins_LAM.named_parameters():
        #     if name in reins_named_params:
        #         param.data.copy_(reins_named_params[name])
                
        weight_sum = sum(client.linear_LPM.weight.data for client in client_model_list)
        bias_sum = sum(client.linear_LPM.bias.data for client in client_model_list)
        
        global_model.linear_LPM.weight.data.copy_(weight_sum / len(client_model_list))
        global_model.linear_LPM.bias.data.copy_(bias_sum / len(client_model_list))
        
        global_model.linear_IAM.weight.data.copy_(weight_sum / len(client_model_list))
        global_model.linear_IAM.bias.data.copy_(bias_sum / len(client_model_list))
        
        global_model.linear_LAM.weight.data.copy_(weight_sum / len(client_model_list))
        global_model.linear_LAM.bias.data.copy_(bias_sum / len(client_model_list))
        
        for i in range(len(client_model_list)):
            client_model_list[i] = copy.deepcopy(global_model)
    
    bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='ILPM')
    logging.info(f"Global Model after Step 2 - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
    
    # =============================================================================
    # Main Training Loop (Epochs 3)
    # =============================================================================
    for epoch in range(args.round3):
        logging.info("="*80)
        logging.info(f"Main Training Loop: Starting Epoch {epoch + 1}/{args.round3}")
        logging.info("="*80)
        
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            # client_model = copy.deepcopy(global_model)
            client_model.train()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, client_model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]
            
            for lep in range(args.local_ep):
                acc = 0
                sum_total = 0
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    logit_LPM = client_model.forward_ILPM(inputs, [-1]) + 0.5*class_list
                    pred_LPM = torch.argmax(logit_LPM, dim=1)
                    with torch.no_grad():
                        acc += torch.sum(pred_LPM==targets).float().mean().item()
                        sum_total += torch.sum(targets)
                    # acc_LPM = (pred_LPM == targets).float().mean().item()
                    # print(f"LPM step acc: {acc_LPM:.4f}")
                    
                    loss_LPM = F.cross_entropy(logit_LPM, targets, reduction='none')
                    loss = loss_LPM.mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f'Client{client_idx} Local Epoch {lep} LPM Train Acc : {100*acc/sum_total}')
            
            for lep in range(args.local_ep):
                acc = 0
                sum_accurate = 0
                sum_total = 0
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    logit_IAM = client_model.forward_IAM(inputs) + 0.5*class_list
                    pred_IAM = torch.argmax(logit_IAM, dim=1)
                    # acc_IAM = (pred_IAM == targets).float().mean().item()
                    # print(f"IAM step acc: {acc_IAM:.4f}")
                    
                    with torch.no_grad():
                        logit_LPM = client_model.forward_ILPM(inputs, [-1]) + 0.5*class_list
                        pred_LPM = torch.argmax(logit_LPM, dim=1)                        
                        LPM_Agree = (pred_LPM==targets)
                        
                        acc += torch.sum(pred_IAM==targets).float().mean().item()
                        sum_accurate += torch.sum(LPM_Agree)
                        sum_total += torch.sum(targets)
                    
                    loss_IAM = F.cross_entropy(logit_IAM, targets, reduction='none')     
                    loss = LPM_Agree * loss_IAM
                    loss = loss.mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f'Client{client_idx} Local Epoch {lep} LPM agree ratio : {100*sum_accurate/sum_total}')
                print(f'Client{client_idx} Local Epoch {lep} IAM Train Acc : {100*acc/sum_total}')
                    
            for lep in range(args.local_ep):
                acc = 0
                sum_accurate = 0
                sum_total = 0
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    logit_LAM = client_model.forward_LAM(inputs) + 0.5*class_list
                    pred_LAM = torch.argmax(logit_LAM, dim=1)
                    # acc_LAM = (pred_LAM == targets).float().mean().item()
                    # print(f"LAM step acc: {acc_LAM:.4f}")
                    
                    with torch.no_grad():
                        logit_IAM = client_model.forward_IAM(inputs) + 0.5*class_list
                        pred_IAM = torch.argmax(logit_IAM, dim=1)                        
                        IAM_Agree = (pred_IAM==targets)
                        
                        acc += torch.sum(pred_LAM==targets).float().mean().item()
                        sum_accurate += torch.sum(IAM_Agree)
                        sum_total += torch.sum(targets)
                    
                    loss_LAM = F.cross_entropy(logit_LAM, targets, reduction='none')
                    loss = IAM_Agree * loss_LAM
                    loss = loss.mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f'Client{client_idx} Local Epoch {lep} IAM agree ratio : {100*sum_accurate/sum_total}')
                print(f'Client{client_idx} Local Epoch {lep} LAM Train Acc : {100*acc/sum_total}')
    
        # =============================================================================
        # Step 3: 학습된 클라이언트 모델의 rein 어댑터를 모아서 global 모델 업데이트
        # =============================================================================
        logging.info(f"--- [Epoch {epoch+1}] Step 3: Averaging adapters to global model ---")
        
        with torch.no_grad():
            reins_named_params = {}
            for name, _ in client_model_list[0].reins_LPM.named_parameters():
                stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
                reins_named_params[name] = stacked.mean(dim=0)
            for name, param in global_model.reins_LPM.named_parameters():
                if name in reins_named_params:
                    param.data.copy_(reins_named_params[name])
            
            reins_named_params = {}
            for name, _ in client_model_list[0].reins_IAM.named_parameters():
                stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
                reins_named_params[name] = stacked.mean(dim=0)
            for name, param in global_model.reins_IAM.named_parameters():
                if name in reins_named_params:
                    param.data.copy_(reins_named_params[name])
                    
            reins_named_params = {}
            for name, _ in client_model_list[0].reins_LAM.named_parameters():
                stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
                reins_named_params[name] = stacked.mean(dim=0)
            # for name, param in global_model.reins_LPM.named_parameters():
            #     if name in reins_named_params:
            #         param.data.copy_(reins_named_params[name])
            # for name, param in global_model.reins_IAM.named_parameters():
            #     if name in reins_named_params:
            #         param.data.copy_(reins_named_params[name])
            for name, param in global_model.reins_LAM.named_parameters():
                if name in reins_named_params:
                    param.data.copy_(reins_named_params[name])
                    
            weight_sum = sum(client.linear_LPM.weight.data for client in client_model_list)
            bias_sum = sum(client.linear_LPM.bias.data for client in client_model_list)
            
            global_model.linear_LPM.weight.data.copy_(weight_sum / len(client_model_list))
            global_model.linear_LPM.bias.data.copy_(bias_sum / len(client_model_list))
            
            weight_sum = sum(client.linear_IAM.weight.data for client in client_model_list)
            bias_sum = sum(client.linear_IAM.bias.data for client in client_model_list)
            
            global_model.linear_IAM.weight.data.copy_(weight_sum / len(client_model_list))
            global_model.linear_IAM.bias.data.copy_(bias_sum / len(client_model_list))
            
            weight_sum = sum(client.linear_LAM.weight.data for client in client_model_list)
            bias_sum = sum(client.linear_LAM.bias.data for client in client_model_list)
            
            # global_model.linear_LPM.weight.data.copy_(weight_sum / len(client_model_list))
            # global_model.linear_LPM.bias.data.copy_(bias_sum / len(client_model_list))
            
            # global_model.linear_IAM.weight.data.copy_(weight_sum / len(client_model_list))
            # global_model.linear_IAM.bias.data.copy_(bias_sum / len(client_model_list))
            
            global_model.linear_LAM.weight.data.copy_(weight_sum / len(client_model_list))
            global_model.linear_LAM.bias.data.copy_(bias_sum / len(client_model_list))
            
            for i in range(len(client_model_list)):
                client_model_list[i] = copy.deepcopy(global_model)
        
        bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='ILPM')
        logging.info(f"Global Model after Step 3 ILPM - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
        
        bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='IAM')
        logging.info(f"Global Model after Step 3 IAM - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
        
        bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='LAM')
        logging.info(f"Global Model after Step 3 LAM - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
        
if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
