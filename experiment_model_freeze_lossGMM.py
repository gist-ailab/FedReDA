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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report

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

# ================================
# FedAvg 함수
# ================================
def fedavg_clients_to_global(global_model, client_model_list):
    global_state = global_model.state_dict()
    new_state = {}
    for k in global_state.keys():
        new_state[k] = sum([c.state_dict()[k] for c in client_model_list]) / len(client_model_list)
    # 필요 시 adapter/linear만 평균하려면 아래처럼 필터:
    # for k in global_state.keys():
    #     if ('reins.' in k) or ('linear_rein.' in k):
    #         new_state[k] = sum([c.state_dict()[k] for c in client_model_list]) / len(client_model_list)
    #     else:
    #         new_state[k] = global_state[k]
    global_model.load_state_dict(new_state)
    return global_model

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
# DatasetSplit (client 학습용)
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
# DatasetGlobal (global loss 추론용)
# ======================================================================
class DatasetGlobal(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, noisy_label = self.base[idx]
        true_label = self.base.true_labels[idx]
        return img, noisy_label, true_label, idx

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
                    # logits = logits + 0.5*class_list  # (옵션)

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

    # ------------------------
    # 1) FedAvg로 Global Model 생성
    # ------------------------
    print("\n[Step] Aggregating client models into global model...")
    global_model = fedavg_clients_to_global(global_model, client_model_list)
    global_model.eval()

    # ------------------------
    # 2) Global Model로 전체 학습 데이터 loss 계산
    # ------------------------
    print("[Step] Computing per-sample loss with global model...")
    global_loss_tracker = {
        "loss": np.zeros(len(dataset_train), dtype=np.float32),
        "is_noisy": np.zeros(len(dataset_train), dtype=np.int32),
        "label": np.zeros(len(dataset_train), dtype=np.int32)
    }

    global_loader = DataLoader(
        DatasetGlobal(dataset_train),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    with torch.no_grad():
        for inputs, noisy_label, true_label, idx in global_loader:
            inputs = inputs.to(device)
            feats = global_model.forward_features_widx(inputs, idxs=adapter_idx)[:, 0, :]
            logits = global_model.linear_rein(feats)
            ce = F.cross_entropy(logits, noisy_label.to(device), reduction='none').cpu().numpy()

            idx_np = idx.numpy()
            global_loss_tracker["loss"][idx_np] = ce
            global_loss_tracker["is_noisy"][idx_np] = (noisy_label != true_label).numpy()
            global_loss_tracker["label"][idx_np] = noisy_label.numpy()

    df_global = pd.DataFrame({
        "index": np.arange(len(dataset_train)),
        "loss_mean": global_loss_tracker["loss"],
        "is_noisy": global_loss_tracker["is_noisy"],
        "label": global_loss_tracker["label"]
    })

    # ------------------------
    # 3) 클래스별 z-score 계산
    # ------------------------
    df_global["loss_mean_cz"] = df_global.groupby("label")["loss_mean"] \
                                         .transform(lambda x: (x - x.mean())/(x.std()+1e-8))

    # ------------------------
    # 4) Multistage GMM (Global, stage1~3)
    # ------------------------
    def multistage_gmm(df, feat_cz, raw_loss="loss_mean",
                       max_stages=3, min_clean_size=50,
                       min_gap=0.15, min_noisy_frac=0.05, min_conf=0.80):
        pred = pd.Series(-1, index=df.index, dtype=int)
        for lbl, grp in df.groupby("label"):
            remain = grp.index.copy()
            noisy_all = []
            for s in range(1, max_stages+1):
                if len(remain) < max(min_clean_size, 5): break
                X = df.loc[remain, [feat_cz]].values
                if not np.isfinite(X).all(): break
                gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
                clusters = gmm.predict(X)
                resp = gmm.predict_proba(X)
                conf = resp.max(axis=1).mean()

                tmp = df.loc[remain].copy()
                tmp["cluster"] = clusters
                mean_by = tmp.groupby("cluster")[raw_loss].mean().sort_values()
                clean_c, noisy_c = mean_by.index[0], mean_by.index[1]

                gap = float(np.linalg.norm(gmm.means_[clean_c] - gmm.means_[noisy_c]))
                noisy_frac = (clusters==noisy_c).mean()

                if gap < min_gap or noisy_frac < min_noisy_frac or conf < min_conf:
                    break

                noisy_idx = remain[clusters==noisy_c]
                noisy_all.append(noisy_idx)
                remain = remain[clusters==clean_c]

            if len(noisy_all):
                noisy_all = np.concatenate(noisy_all)
                pred.loc[noisy_all] = 1
            pred.loc[remain] = 0
        return pred

    df_global["gmm_stage1"] = multistage_gmm(df_global,"loss_mean_cz",max_stages=1)
    df_global["gmm_stage2"] = multistage_gmm(df_global,"loss_mean_cz",max_stages=2)
    df_global["gmm_stage3"] = multistage_gmm(df_global,"loss_mean_cz",max_stages=3)

    # ------------------------
    # 5) Global 결과 비교
    # ------------------------
    for col in ["gmm_stage1","gmm_stage2","gmm_stage3"]:
        valid = df_global[col] != -1
        print(f"\n=== [Global Model] Confusion Matrix for {col} ===")
        cm = confusion_matrix(df_global.loc[valid,"is_noisy"], df_global.loc[valid,col], labels=[0,1])
        print(cm)
        print(classification_report(df_global.loc[valid,"is_noisy"], df_global.loc[valid,col], digits=4))

    print("\n[Info] Global Model 기반 GMM 결과가 위에 출력되었습니다.")

    # ------------------------
    # 6) Local 학습 기반 지표 집계 및 멀티스테이지(GMM, class-wise)
    # ------------------------
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

    # Local: loss_mean/var/slope 클래스별 z-score
    loss_feats_raw = ["loss_mean", "loss_var", "loss_slope"]
    for f in loss_feats_raw:
        df[f + "_cz"] = df.groupby("label")[f].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    loss_feats_cz = [f + "_cz" for f in loss_feats_raw]

    def multistage_gmm_classwise(df, feats_cz, raw_loss_col="loss_mean",
                                 max_stages=4, min_clean_size=50,
                                 min_mean_gap=0.15, min_noisy_frac=0.05, min_conf=0.80):
        pred = pd.Series(-1, index=df.index, dtype=int)
        for lbl, grp in df.groupby("label"):
            remain_idx = grp.index.copy()
            noisy_idx_total = []
            for stage in range(1, max_stages+1):
                if len(remain_idx) < max(min_clean_size, 5):
                    break
                X = df.loc[remain_idx, feats_cz].values
                if not np.isfinite(X).all():
                    break
                gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
                clusters = gmm.predict(X)
                resp = gmm.predict_proba(X)
                conf = resp.max(axis=1).mean()

                tmp = df.loc[remain_idx].copy()
                tmp["cluster"] = clusters
                mean_by_c = tmp.groupby("cluster")[raw_loss_col].mean().sort_values()
                clean_c, noisy_c = mean_by_c.index[0], mean_by_c.index[1]

                mu = gmm.means_
                gap = float(np.linalg.norm(mu[clean_c] - mu[noisy_c], ord=2))
                noisy_mask = (clusters == noisy_c)
                noisy_frac = noisy_mask.mean()

                if gap < min_mean_gap or noisy_frac < min_noisy_frac or conf < min_conf:
                    break

                noisy_idx = remain_idx[noisy_mask]
                noisy_idx_total.append(noisy_idx)
                remain_idx = remain_idx[~noisy_mask]

            if len(noisy_idx_total):
                noisy_idx_total = np.concatenate(noisy_idx_total)
                pred.loc[noisy_idx_total] = 1
            pred.loc[remain_idx] = 0
        return pred

    # Local 멀티스테이지: s2, s3, s4 (기존 유지)
    df["gmm_pred_multistage_s2"] = multistage_gmm_classwise(
        df, feats_cz=loss_feats_cz, raw_loss_col="loss_mean",
        max_stages=2, min_clean_size=50, min_mean_gap=0.15, min_noisy_frac=0.05, min_conf=0.80
    )
    df["gmm_pred_multistage_s3"] = multistage_gmm_classwise(
        df, feats_cz=loss_feats_cz, raw_loss_col="loss_mean",
        max_stages=3, min_clean_size=50, min_mean_gap=0.15, min_noisy_frac=0.05, min_conf=0.80
    )
    df["gmm_pred_multistage_s4"] = multistage_gmm_classwise(
        df, feats_cz=loss_feats_cz, raw_loss_col="loss_mean",
        max_stages=4, min_clean_size=50, min_mean_gap=0.15, min_noisy_frac=0.05, min_conf=0.80
    )

    # 평가(Local 멀티스테이지)
    for col in ["gmm_pred_multistage_s2","gmm_pred_multistage_s3","gmm_pred_multistage_s4"]:
        valid = df[col] != -1
        print(f"\n=== [Local Metrics] Confusion Matrix for {col} ===")
        cm = confusion_matrix(df.loc[valid,"is_noisy"], df.loc[valid,col], labels=[0,1])
        print(cm)
        print(classification_report(df.loc[valid,"is_noisy"], df.loc[valid,col], labels=[0,1], digits=4))

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
