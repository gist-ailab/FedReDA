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
mp.set_start_method("spawn", force=True)

def worker_init_fn(worker_id):
    tmpdir = f"/home/work/DATA1/tmp/worker_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

# rmtree monkey patch
_old_rmtree = shutil.rmtree
def safe_rmtree(path, *a, **kw):
    try:
        return _old_rmtree(path, *a, **kw)
    except OSError as e:
        if e.errno == errno.EBUSY:
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

    # clients
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

    # Client 수 줄이기 (데모용)
    clients_train_loader_list = clients_train_loader_list[:5]
    args.num_clients = 5

    # Global model
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

    # tracker
    loss_tracker = defaultdict(lambda: {
        "losses": [],
        "confidences": [],
        "margins": [],
        "preds": [],
        "agreements": [],
        "grad_ll": [],
        "is_noisy": None,
        "label": None,
        "true_label": None,
        "client": None,
    })

    # training
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
            for _ in range(args.local_ep):
                for inputs, targets, true_label, batch_indices in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    feats = model.forward_features_widx(inputs, idxs=adapter_idx)[:, 0, :]
                    logits = model.linear_rein(feats)
                    # logits = logits + 0.5*class_list

                    ce_losses = F.cross_entropy(logits, targets, reduction='none')
                    probs = F.softmax(logits, dim=1)
                    conf_top1, preds = probs.max(dim=1)
                    top2 = torch.topk(probs, 2, dim=1).values[:, 1]
                    margins = conf_top1 - top2

                    # global agreement
                    with torch.no_grad():
                        g_feats = global_model.forward_features_widx(inputs, idxs=adapter_idx)[:, 0, :]
                        g_logits = global_model.linear_rein(g_feats)
                        g_preds = torch.argmax(g_logits, dim=1)

                    # gradient closed form
                    y_onehot = torch.zeros_like(probs).scatter_(1, targets.view(-1,1), 1.0)
                    e = probs - y_onehot
                    e_sq = (e ** 2).sum(dim=1)
                    feat_sq = (feats ** 2).sum(dim=1)
                    grad_ll_norm = torch.sqrt(e_sq * (feat_sq + 1.0))

                    # record
                    for i in range(len(batch_indices)):
                        idx_i = batch_indices[i].item()
                        loss_tracker[idx_i]["losses"].append(ce_losses[i].item())
                        loss_tracker[idx_i]["confidences"].append(conf_top1[i].item())
                        loss_tracker[idx_i]["margins"].append(margins[i].item())
                        loss_tracker[idx_i]["preds"].append(preds[i].item())
                        loss_tracker[idx_i]["agreements"].append(int(preds[i].item() == g_preds[i].item()))
                        loss_tracker[idx_i]["grad_ll"].append(grad_ll_norm[i].item())

                        loss_tracker[idx_i]["is_noisy"] = (targets[i] != true_label[i]).item()
                        loss_tracker[idx_i]["label"] = targets[i].item()
                        loss_tracker[idx_i]["true_label"] = true_label[i].item()
                        loss_tracker[idx_i]["client"] = client_idx

                    # update
                    loss = ce_losses.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    # aggregate per-sample features
    loss_records = []
    for idx, v in loss_tracker.items():
        L = np.array(v["losses"]); C = np.array(v["confidences"])
        M = np.array(v["margins"]); P = np.array(v["preds"])
        A = np.array(v["agreements"]); G = np.array(v["grad_ll"])
        n = max(1, len(L))
        consistency = (np.bincount(P).max() / len(P)) if len(P) > 0 else 0
        loss_var = float(L.var()) if len(L) > 1 else 0.0
        if len(L) > 1:
            x = np.arange(len(L), dtype=np.float32)
            b1 = np.polyfit(x, L, 1)[0]
            loss_slope = float(b1)
        else:
            loss_slope = 0.0
        loss_records.append({
            "index": idx,
            "loss_sum": float(L.sum()),
            "loss_mean": float(L.mean()) if len(L) else 0.0,
            "loss_var": loss_var,
            "loss_slope": loss_slope,
            "conf_mean": float(C.mean()) if len(C) else 0.0,
            "margin_mean": float(M.mean()) if len(M) else 0.0,
            "consistency": float(consistency),
            "agree_rate": float(A.mean()) if len(A) else 0.0,
            "grad_ll_mean": float(G.mean()) if len(G) else 0.0,
            "count": int(n),
            "is_noisy": v["is_noisy"],
            "label": v["label"],
            "true_label": v["true_label"],
            "client": v["client"],
        })
    df = pd.DataFrame(loss_records)

    # GMM class-wise
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import confusion_matrix, classification_report

    df["gmm_pred_classwise"] = -1
    features = ["loss_mean","loss_var","loss_slope","conf_mean",
                "margin_mean","consistency","agree_rate","grad_ll_mean"]

    for label, group in df.groupby("label"):
        X = group[features].values
        if len(group) < 5:
            df.loc[group.index, "gmm_pred_classwise"] = 0
            continue
        gmm = GaussianMixture(n_components=2, random_state=0)
        clusters = gmm.fit_predict(X)
        cm = group.groupby(clusters)["loss_mean"].mean().sort_values()
        clean_cluster, noisy_cluster = cm.index[0], cm.index[1]
        df.loc[group.index, "gmm_pred_classwise"] = (clusters == noisy_cluster).astype(int)
        
    feats_raw = ["loss_mean", "grad_ll_mean"]
    for f in feats_raw:
        df[f + "_cz"] = df.groupby("label")[f].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    feats_cz = [f + "_cz" for f in feats_raw]
    
    # -----------------------------------------------
    # 2) 결과 컬럼 초기화
    df["gmm_pred_all3"]   = -1  # loss+grad 동시 사용
    df["gmm_pred_loss"]   = -1  # loss만
    df["gmm_pred_grad"]   = -1  # grad만

    # -----------------------------------------------
    # 3) 클래스별 GMM: (a) 3개 동시, (b) 단일 특성
    for label, group in df.groupby("label"):
        # (a) 3개 동시
        X = group[feats_cz].values
        if len(group) >= 5 and np.isfinite(X).all():
            gmm = GaussianMixture(n_components=2, random_state=0)
            clusters = gmm.fit_predict(X)
            # clean/noisy 매핑: "원시 loss_mean"의 클래스별 평균을 기준으로 더 낮은 쪽을 clean
            mean_loss_by_cluster = group.groupby(clusters)["loss_mean"].mean().sort_values()
            clean_c, noisy_c = mean_loss_by_cluster.index[0], mean_loss_by_cluster.index[1]
            df.loc[group.index, "gmm_pred_all3"] = (clusters == noisy_c).astype(int)
        else:
            df.loc[group.index, "gmm_pred_all3"] = 0  # 데이터 적음 → 보수적으로 clean 처리

        # (b) 단일 특성별
        for f_raw, f_cz, out_col in zip(feats_raw, feats_cz,
                                        ["gmm_pred_loss", "gmm_pred_grad"]):
            X1 = group[[f_cz]].values
            if len(group) >= 5 and np.isfinite(X1).all():
                gmm1 = GaussianMixture(n_components=2, random_state=0)
                c1 = gmm1.fit_predict(X1)
                # 해당 단일 특성의 "원시 값" 평균으로 매핑
                mean_feat = group.groupby(c1)[f_raw].mean().sort_values()
                clean_c, noisy_c = mean_feat.index[0], mean_feat.index[1]
                df.loc[group.index, out_col] = (c1 == noisy_c).astype(int)
            else:
                df.loc[group.index, out_col] = 0

    # -----------------------------------------------
    # 4) 교집합(세 단일 GMM이 모두 동일한 경우만 채택)
    df["gmm_pred_intersect"] = -1
    mask_clean = (df["gmm_pred_loss"]==0) & (df["gmm_pred_grad"]==0)
    mask_noisy = (df["gmm_pred_loss"]==1) & (df["gmm_pred_grad"]==1)
    df.loc[mask_clean, "gmm_pred_intersect"] = 0
    df.loc[mask_noisy, "gmm_pred_intersect"] = 1
    # (-1은 불일치 → 불확실)

    # -----------------------------------------------
    # 5) Confusion matrix 출력
    targets = ["gmm_pred_all3","gmm_pred_loss","gmm_pred_grad","gmm_pred_intersect"]
    for tgt in targets:
        valid = df[tgt] != -1  # 교집합의 불확실 샘플 제외
        print(f"\n=== Confusion Matrix for {tgt} ===")
        cm = confusion_matrix(df.loc[valid, "is_noisy"], df.loc[valid, tgt], labels=[0,1])
        print(cm)
        print(classification_report(df.loc[valid, "is_noisy"], df.loc[valid, tgt], labels=[0,1], digits=4))

    #######################################################################
    # cm = confusion_matrix(df["is_noisy"], df["gmm_pred_classwise"])
    # print("Confusion Matrix (rows=true, cols=pred):\n", cm)
    # print("\nClassification Report:")
    # print(classification_report(df["is_noisy"], df["gmm_pred_classwise"], digits=4))
    # print("\nCluster-wise feature means:")
    # print(df.groupby("gmm_pred_classwise")[features].mean())
    # print("\nTrue clean/noisy 별 feature 평균:")
    # print(df.groupby("is_noisy")[features].mean())

    # print("\nTrue clean/noisy 별 feature 분포 요약:")
    # # print(df.groupby("is_noisy")[features].describe())
    # for col in features:  # features = ["loss_mean","loss_var",...]
    #     print(f"\n=== {col} ===")
    #     print(df.groupby("is_noisy")[col].describe())
    
    # # 로그 저장 및 시각화
    
    # print(df.groupby("is_noisy")["loss_sum"].describe())
    # print(df.groupby(["label", "is_noisy"])["loss_sum"].describe())

    # plt.figure(figsize=(14, 6))
    # sns.boxplot(data=df, x="label", y="loss_sum", hue="is_noisy")
    # plt.savefig(f"EP5_Index({adapter_idx})_per_class_loss_box.png")
    # plt.close()

    # g = sns.FacetGrid(df, col="label", hue="is_noisy", col_wrap=4, sharex=False, sharey=False)
    # g.map(sns.histplot, "loss_sum", bins=30, stat="density", alpha=0.5)
    # g.add_legend()
    # g.savefig(f"EP5_Index({adapter_idx})_per_class_loss_hist.png")
    # plt.close()

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
