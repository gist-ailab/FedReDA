import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import uuid
import copy
import logging
import tempfile
import shutil
import errno

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

# --- FedNoRo 모듈 경로 추가 ---
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)

# --- 기존 FedLNL 의존성 ---
from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from update import average_reins
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# --- FedNoRo 데이터셋 관련 모듈 ---
from dataset.dataset import get_dataset
from utils.utils import add_noise

# ======================================================================
# Multiprocessing / tmpdir 설정
# ======================================================================
mp.set_start_method("spawn", force=True)   # fork 대신 spawn 강제

def worker_init_fn(worker_id):
    """각 worker마다 독립된 tmpdir을 NAS에 생성"""
    tmpdir = f"/home/work/DATA1/tmp/worker_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

# rmtree monkey patch → NAS busy 에러 무시
_old_rmtree = shutil.rmtree
def safe_rmtree(path, *a, **kw):
    try:
        return _old_rmtree(path, *a, **kw)
    except OSError as e:
        if e.errno == errno.EBUSY:
            # print(f"[WARN] Ignore busy tmpdir {path}")
            return
        raise
shutil.rmtree = safe_rmtree

# ======================================================================
# 로거 설정
# ======================================================================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler('Model_Adapter.txt', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

# ======================================================================
# Helper functions
# ======================================================================
def calculate_accuracy(model, dataloader, device, mode='rein', index=None):
    model.eval()
    all_preds, all_targets = [], []
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
                if isinstance(index, int):
                    index = [index]
                feats = model.forward_features_widx(inputs, idxs=index)[:, 0, :]
                logits = model.linear_rein(feats)
            else:
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    model.train()
    return bacc, acc


def ema_update(target_module, source_module, alpha=0.9):
    for (name_t, param_t), (name_s, param_s) in zip(
        target_module.named_parameters(), source_module.named_parameters()
    ):
        assert name_t == name_s
        param_t.data.mul_(alpha).add_(param_s.data * (1 - alpha))


# ======================================================================
# DatasetSplit
# ======================================================================
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


# ======================================================================
# FedDouble Model
# ======================================================================
class FedDoubleModel(ReinsDinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_fusion1(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x1 = blk(x)
            x = self.reins.forward(x1, idx, batch_first=True, has_cls_token=True)
            x = x + self.reins2.forward(x1, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_fusion2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
            x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    def forward_fusion3(self, x):
        return self.forward_features(x) + self.forward_features2(x)


# ======================================================================
# Main
# ======================================================================
def main(args):
    setup_logging()
    logging.info("=" * 50)
    logging.info("Starting FedDouble training process...")
    device = torch.device(f"cuda:{args.gpu}")

    args.num_users = args.num_clients
    args.n_clients = args.num_clients
    
    # 데이터셋
    dataset_train, dataset_test, dict_users = get_dataset(args)
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_train_noisy
    logging.info(f"Loaded dataset '{args.dataset}' with {len(dict_users)} clients.")
    logging.info(f"Noisy train data distribution: {Counter(dataset_train.targets)}")

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    clients_train_loader_list, clients_train_class_num_list = [], []
    for i in range(args.num_clients):
        client_indices = dict_users[i]
        client_dataset = DatasetSplit(dataset_train, client_indices)

        train_loader = DataLoader(
            client_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=True
        )

        class_num_list = [0 for _ in range(args.n_classes)]
        for idx in dict_users[i]:
            class_num_list[int(dataset_train.targets[idx])] += 1
        class_num_tensor = torch.cuda.FloatTensor(class_num_list)
        class_p_list = torch.log(class_num_tensor / class_num_tensor.sum()).view(1, -1)

        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p_list)

    # Client 수 줄이기
    clients_train_loader_list = clients_train_loader_list[:5]
    args.num_clients = 5

    # Global model 초기화
    adapter_idx = [i for i in range(12)]
    global_model = ReinsDinoVisionTransformer(**_small_variant)
    global_model.load_state_dict(
        torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth',
                   weights_only=False),
        strict=False
    )
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.to(device)
    client_model_list = [copy.deepcopy(global_model) for _ in range(args.num_clients)]

    loss_tracker = defaultdict(lambda: {
        "total_loss": 0.0,
        "count": 0,
        "is_noisy": None,
        "label": None,
        "true_label": None,
        "client": None
    })
    for epoch in range(5):
        for client_idx in range(args.num_clients):
            model = client_model_list[client_idx]
            model.train()
            for name, param in model.named_parameters():
                if 'reins.' in name or 'linear_rein.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
            )
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]

            for _ in range(args.local_ep):
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    feats = model.forward_features_widx(inputs, idxs=adapter_idx)[:, 0, :]
                    logits = model.linear_rein(feats)
                    
                    # logits = logits + 0.5*class_list
                    
                    ce_losses = F.cross_entropy(logits, targets, reduction='none')

                    for i in range(len(batch_indices)):
                        idx = batch_indices[i].item()
                        loss_tracker[idx]["total_loss"] += ce_losses[i].item()
                        loss_tracker[idx]["count"] += 1
                        loss_tracker[idx]["is_noisy"] = (targets[i] != true_label[i]).item()
                        loss_tracker[idx]["label"] = targets[i].item()
                        loss_tracker[idx]["true_label"] = true_label[i].item()
                        loss_tracker[idx]["client"] = client_idx

                    loss = ce_losses.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    loss_records = []
    for idx, v in loss_tracker.items():
        loss_records.append({
            "index": idx,
            "loss_sum": v["total_loss"],
            "loss_mean": v["total_loss"] / v["count"],
            "count": v["count"],
            "is_noisy": v["is_noisy"],
            "label": v["label"],
            "true_label": v["true_label"],
            "client": v["client"]
        })
    
    # Global update
    with torch.no_grad():
        reins_named_params = {}
        for name, _ in client_model_list[0].reins.named_parameters():
            stacked = torch.stack([
                dict(client.reins.named_parameters())[name].data
                for client in client_model_list
            ])
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

    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import confusion_matrix, classification_report

    # ----------------------------------------------------------
    # GMM을 이용한 clean/noisy 분류 시도
    # ----------------------------------------------------------
    # # feature로 누적합(loss_sum) 또는 평균(loss_mean) 중 선택
    df = pd.DataFrame(loss_records)
    # X = df[["loss_sum"]].values   # 누적합 기준
    # # X = df[["loss_mean"]].values   # 평균 기준으로 하고 싶으면 이 줄 사용

    # gmm = GaussianMixture(n_components=2, random_state=0)
    # df["gmm_cluster"] = gmm.fit_predict(X)

    # # ----------------------------------------------------------
    # # 클러스터 라벨을 clean/noisy로 매핑
    # #   - 어떤 클러스터가 noisy인지 알 수 없으니,
    # #     cluster별 평균 loss를 보고 noisy=평균 더 큰 쪽으로 매핑
    # # ----------------------------------------------------------
    # cluster_means = df.groupby("gmm_cluster")["loss_sum"].mean().sort_values()
    # clean_cluster = cluster_means.index[0]
    # noisy_cluster = cluster_means.index[1]

    # df["gmm_pred"] = df["gmm_cluster"].map(
    #     lambda c: 1 if c == noisy_cluster else 0
    # )  # 1=noisy, 0=clean
    df["gmm_pred_classwise"] = -1

    for label, group in df.groupby("label"):
        X = group[["loss_sum"]].values
        if len(np.unique(X)) < 2:
            # 데이터가 너무 적거나 분산이 없을 경우 clean으로 처리
            df.loc[group.index, "gmm_pred_classwise"] = 0
            continue
        
        gmm = GaussianMixture(n_components=2, random_state=0)
        clusters = gmm.fit_predict(X)
        
        # 평균 loss가 큰 cluster를 noisy로 지정
        cluster_means = group.groupby(clusters)["loss_sum"].mean().sort_values()
        clean_cluster, noisy_cluster = cluster_means.index[0], cluster_means.index[1]
        
        mapped = [1 if c == noisy_cluster else 0 for c in clusters]
        df.loc[group.index, "gmm_pred_classwise"] = mapped

    # ----------------------------------------------------------
    # Confusion Matrix 및 성능 지표
    # ----------------------------------------------------------
    # cm = confusion_matrix(df["is_noisy"], df["gmm_pred"])
    # print("Confusion Matrix (rows=true, cols=pred):\n", cm)

    # print("\nClassification Report:")
    # print(classification_report(df["is_noisy"], df["gmm_pred"], digits=4))
    cm = confusion_matrix(df["is_noisy"], df["gmm_pred_classwise"])
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)
    print("\nClassification Report:")
    print(classification_report(df["is_noisy"], df["gmm_pred_classwise"], digits=4))

    # 시각화
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Clean", "Pred Noisy"],
                yticklabels=["True Clean", "True Noisy"])
    plt.title("Confusion Matrix of GMM-based Noise Detection")
    plt.savefig("GMM_confusion_matrix.png")
    plt.close()
    
    # 로그 저장 및 시각화
    
    print(df.groupby("is_noisy")["loss_sum"].describe())
    print(df.groupby(["label", "is_noisy"])["loss_sum"].describe())

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="label", y="loss_sum", hue="is_noisy")
    plt.savefig(f"EP5_Index({adapter_idx})_per_class_loss_box.png")
    plt.close()

    g = sns.FacetGrid(df, col="label", hue="is_noisy", col_wrap=4, sharex=False, sharey=False)
    g.map(sns.histplot, "loss_sum", bins=30, stat="density", alpha=0.5)
    g.add_legend()
    g.savefig(f"EP5_Index({adapter_idx})_per_class_loss_hist.png")
    plt.close()

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
