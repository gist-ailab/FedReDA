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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 로거 설정 ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # 파일 핸들러
    file_handler = logging.FileHandler('Model_Adapter.txt', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

def calculate_accuracy(model, dataloader, device, mode='rein', index=None):
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
            elif mode == 'index':
                if type(index)==int:
                    index = [index]
                feats = model.forward_features_widx(inputs, idxs=index)[:, 0, :]
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
        true_label = self.dataset.true_labels[self.idxs[item]]
        index = self.idxs[item]
        return image, label, true_label, index

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
        
    clients_train_loader_list = clients_train_loader_list[:5]
    args.num_clients = 5

    # =============================================================================
    # Step 1: 클라이언트 모델을 Linear Probing Layer만으로 학습.
    # =============================================================================
    # logging.info("="*50)
    # logging.info("Step 1: Training Linear Probing Model")
    # logging.info("="*50)
    
    # # Global model은 rein2만 활용
    # # Local model은 local adapter가 rein1, global adapter가 rein2를 사용
    # # global_model = FedDoubleModel(**_small_variant)
    # global_model = ReinsDinoVisionTransformer(**_small_variant)
    # global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False), strict=False)
    # global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    # global_model.to(device)
    # client_model_list = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    
    # loss_records = []
    # for epoch in range(5):
    #     for client_idx in range(args.num_clients):
    #         model = client_model_list[client_idx]
    #         model.train()
            
    #         # rein 어댑터와 linear_rein만 학습하도록 설정
    #         for name, param in model.named_parameters():
    #             if 'linear_rein.' in name:
    #                 param.requires_grad = True
    #             else:
    #                 param.requires_grad = False
                    
    #         optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    #         loader = clients_train_loader_list[client_idx]
    #         class_list = clients_train_class_num_list[client_idx]
            
    #         for _ in range(args.local_ep):
    #             for inputs, targets, true_label, batch_indices in loader:
    #                 inputs, targets = inputs.to(device), targets.to(device)
                    
    #                 feats = model.forward_features_no_rein(inputs)[:, 0, :]
    #                 logits = model.linear_rein(feats)+ 0.5*class_list
    #                 ce_losses = F.cross_entropy(logits, targets, reduction='none')
                    
    #                 for i in range(len(batch_indices)):
    #                     is_noisy = (targets[i] != true_label[i]).item()
    #                     loss_records.append({
    #                         "epoch": epoch,
    #                         "client": client_idx,
    #                         "index": batch_indices[i].item(),
    #                         "loss": ce_losses[i].item(),
    #                         "is_noisy": is_noisy,
    #                         "label": targets[i].item(),       # noisy label 기준
    #                         "true_label": true_label[i].item() # clean 기준
    #                     })
    #                 loss = ce_losses.mean()
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
                    
    # with torch.no_grad():
    #     reins_named_params = {}
    #     for name, _ in client_model_list[0].reins.named_parameters():
    #         stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
    #         reins_named_params[name] = stacked.mean(dim=0)
            
    #     for name, param in global_model.reins.named_parameters():
    #         if name in reins_named_params:
    #             param.data.copy_(reins_named_params[name])
                
    #     weight_sum = sum(client.linear_rein.weight.data for client in client_model_list)
    #     bias_sum = sum(client.linear_rein.bias.data for client in client_model_list)
        
    #     global_model.linear_rein.weight.data.copy_(weight_sum / len(client_model_list))
    #     global_model.linear_rein.bias.data.copy_(bias_sum / len(client_model_list))
    
    # bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='norein')
    # logging.info(f"Global Model after Step 2 - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
    
    # import pandas as pd
    # df = pd.DataFrame(loss_records)
    # print(df.groupby("is_noisy")["loss"].describe())
    # print(df.groupby(["true_label", "is_noisy"])["loss"].describe())
    
    # plt.figure(figsize=(14,6))
    # sns.boxplot(data=df, x="true_label", y="loss", hue="is_noisy")
    # plt.xlabel("Class (True Label 기준)")
    # plt.ylabel("Loss")
    # plt.title("Per-class Loss Distribution (Clean vs Noisy)")
    # plt.legend(title="is_noisy", labels=["Clean","Noisy"])
    # plt.savefig(f"EP5_LPM_per_class_loss_box.png")
    # plt.close()

    # # -----------------------------
    # # 2. 클래스별 Clean vs Noisy KDE/히스토그램
    # g = sns.FacetGrid(df, col="true_label", hue="is_noisy", col_wrap=4, sharex=False, sharey=False)
    # g.map(sns.histplot, "loss", bins=30, stat="density", alpha=0.5)
    # g.add_legend()
    # plt.subplots_adjust(top=0.9)
    # g.fig.suptitle("Per-class Loss Histogram (Clean vs Noisy)")
    # g.savefig(f"EP5_LPM_per_class_loss_hist.png")
    # plt.close()
        
    # # Epoch별 평균 loss 추이
    # plt.figure(figsize=(14,6))
    # sns.lineplot(data=df, x="epoch", y="loss", hue="is_noisy", style="true_label", estimator="mean", errorbar="sd")
    # plt.xlabel("Epoch")
    # plt.ylabel("Average Loss")
    # plt.title("Epoch-wise Per-class Loss (Clean vs Noisy)")
    # plt.savefig(f"EP5_LPM_per_class_loss_epoch.png")
    # plt.close()
    
    # =============================================================================
    # Step 2: 클라이언트 모델들을 특정 idxs만 rein 어댑터를 추가해 학습.
    # =============================================================================
    # for adapter_idx in range(_small_variant['depth']):
    adapter_idx = [10]
    logging.info("="*50)
    logging.info(f"Step 2_{adapter_idx}: Training with Adapter Model with idx {adapter_idx}")
    logging.info("="*50)
    
    # Global model은 rein2만 활용
    # Local model은 local adapter가 rein1, global adapter가 rein2를 사용
    # global_model = FedDoubleModel(**_small_variant)
    global_model = ReinsDinoVisionTransformer(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False), strict=False)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.to(device)
    client_model_list = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    
    loss_records = []
    for epoch in range(5):
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
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # feats = model.forward_features(inputs)[:, 0, :]
                    feats = model.forward_features_widx(inputs, idxs=adapter_idx)[:, 0, :]
                    logits = model.linear_rein(feats)
                    
                    # logits = logits + 0.5*class_list
                    ce_losses = F.cross_entropy(logits, targets, reduction='none')
                
                    for i in range(len(batch_indices)):
                        is_noisy = (targets[i] != true_label[i]).item()
                        loss_records.append({
                            "epoch": epoch,
                            "client": client_idx,
                            "index": batch_indices[i].item(),
                            "loss": ce_losses[i].item(),
                            "is_noisy": is_noisy,
                            "label": targets[i].item(),       # noisy label 기준
                            "true_label": true_label[i].item() # clean 기준
                        })
                    loss = ce_losses.mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
    with torch.no_grad():
        reins_named_params = {}
        for name, _ in client_model_list[0].reins.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
            reins_named_params[name] = stacked.mean(dim=0)
            
        for name, param in global_model.reins.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])
                
        weight_sum = sum(client.linear_rein.weight.data for client in client_model_list)
        bias_sum = sum(client.linear_rein.bias.data for client in client_model_list)
        
        global_model.linear_rein.weight.data.copy_(weight_sum / len(client_model_list))
        global_model.linear_rein.bias.data.copy_(bias_sum / len(client_model_list))
    
    bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='index', index=adapter_idx)
    logging.info(f"Global Model after Step 2 - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
    
    import pandas as pd
    df = pd.DataFrame(loss_records)
    print(df.groupby("is_noisy")["loss"].describe())
    print(df.groupby(["true_label", "is_noisy"])["loss"].describe())
    plt.figure(figsize=(14,6))
    sns.boxplot(data=df, x="true_label", y="loss", hue="is_noisy")
    plt.xlabel("Class (True Label 기준)")
    plt.ylabel("Loss")
    plt.title("Per-class Loss Distribution (Clean vs Noisy)")
    plt.legend(title="is_noisy", labels=["Clean","Noisy"])
    plt.savefig(f"EP5_Index({adapter_idx})_per_class_loss_box.png")
    plt.close()

    # -----------------------------
    # 2. 클래스별 Clean vs Noisy KDE/히스토그램
    g = sns.FacetGrid(df, col="true_label", hue="is_noisy", col_wrap=4, sharex=False, sharey=False)
    g.map(sns.histplot, "loss", bins=30, stat="density", alpha=0.5)
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Per-class Loss Histogram (Clean vs Noisy)")
    g.savefig(f"EP5_Index({adapter_idx})_per_class_loss_hist.png")
    plt.close()
        
    # Epoch별 평균 loss 추이
    plt.figure(figsize=(14,6))
    sns.lineplot(data=df, x="epoch", y="loss", hue="is_noisy", style="true_label", estimator="mean", errorbar="sd")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Epoch-wise Per-class Loss (Clean vs Noisy)")
    plt.savefig(f"EP5_Index({adapter_idx})_per_class_loss_epoch.png")
    plt.close()
        
if __name__ == "__main__":   
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)