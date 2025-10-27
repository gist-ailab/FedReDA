# experiment_dynamic_adapter_compare.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno

import sys, multiprocessing as mp
mp.set_executable(sys.executable)
mp.set_start_method("spawn", force=True)

# --- FedNoRo 모듈 경로 추가 ---
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)
    
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from dataset.dataset import get_dataset
from utils.utils import add_noise

mp.set_start_method("spawn", force=True)
def worker_init_fn(_): 
    tmpdir = f"/home/work/DATA1/tmp/worker_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

_old_rmtree = shutil.rmtree
def safe_rmtree(path,*a,**kw):
    try: return _old_rmtree(path,*a,**kw)
    except OSError as e:
        if e.errno == errno.EBUSY: return
        raise
shutil.rmtree = safe_rmtree

def setup_logging():
    lg = logging.getLogger(); lg.setLevel(logging.INFO)
    for h in list(lg.handlers): lg.removeHandler(h)
    fh = logging.FileHandler('experiment_dynamic_adapter_compare.log', mode='a')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(ch)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset; self.idxs = list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        true_label = self.dataset.true_labels[self.idxs[item]]
        index = self.idxs[item]
        return image, label, true_label, index

class DatasetGlobal(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, noisy = self.base[idx]
        true = self.base.true_labels[idx]
        return img, noisy, true, idx

def _make_loader(ds, bs, nw, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=nw, worker_init_fn=worker_init_fn,
                      persistent_workers=True if nw>0 else False)

def eval_model(model, loader, device):
    model.eval(); ap, at = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.forward_features(x)[:,0,:]
            logits = model.linear_rein(feats)
            ap.extend(torch.argmax(logits,1).cpu().numpy())
            at.extend(y.cpu().numpy())
    acc = accuracy_score(at, ap)
    bacc = balanced_accuracy_score(at, ap)
    model.train(); return bacc, acc

class RankScheduler:
    def __init__(self, r_min, r_max, mode="time_dec", gmm_clean_thr=0.6, r_clean=None, r_noisy=None):
        self.r_min, self.r_max = int(r_min), int(r_max)
        self.mode = mode
        self.gmm_thr = gmm_clean_thr
        self.r_clean = int(r_clean) if r_clean is not None else self.r_max
        self.r_noisy = int(r_noisy) if r_noisy is not None else self.r_min
    def epoch_rank(self, epoch, total_epochs):
        if self.mode=="time_inc":
            t = 0 if total_epochs<=1 else epoch/(total_epochs-1)
            return int(round(self.r_min + t*(self.r_max-self.r_min)))
        if self.mode=="time_dec":
            t = 0 if total_epochs<=1 else epoch/(total_epochs-1)
            return int(round(self.r_max - t*(self.r_max-self.r_min)))
        return self.r_max
    def gmm_rank(self, clean_ratio):
        return self.r_clean if clean_ratio>=self.gmm_thr else self.r_noisy

def classwise_loss_gmm(loss_vec, label_vec):
    df = pd.DataFrame({"loss": loss_vec, "label": label_vec})
    df["cz"] = df.groupby("label")["loss"].transform(lambda x:(x-x.mean())/(x.std()+1e-8))
    pred = np.full(len(df), -1, dtype=np.int32)
    for _, g in df.groupby("label"):
        idxs = g.index.values
        X = df.loc[idxs, ["cz"]].values
        if len(idxs)<5 or not np.isfinite(X).all(): 
            pred[idxs]=0; continue
        gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
        c = gmm.predict(X); prob = gmm.predict_proba(X)
        # noisy=평균 손실 큰 군집
        means = pd.DataFrame({"c":c,"loss":df.loc[idxs,"loss"].values}).groupby("c")["loss"].mean().sort_values()
        clean_c, noisy_c = means.index[0], means.index[1]
        pred[idxs[c==clean_c]]=0; pred[idxs[c==noisy_c]]=1
    return pred

def train_one(flavor_name,  # "BASELINE_FIXED" | "DYNAMIC"
              adapter_kind, rank_mode, args,
              dict_users, selected_clients,
              dataset_train, dataset_test, device):

    # 공통 로더
    test_loader = _make_loader(dataset_test, args.batch_size, args.num_workers, shuffle=False)
    adapter_idx = list(range(_small_variant['depth']))

    # 글로벌/클라이언트 초기화
    def build_model():
        m = ReinsDinoVisionTransformer(
            adapter_kind=adapter_kind,
            rein_r_max=args.rein_r_max, rein_alpha=1.0, rein_pre_norm=True,
            **_small_variant
        )
        m.load_state_dict(
            torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False),
            strict=False
        )
        m.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
        m.to(device)
        return m

    global_model = build_model()
    client_models = [copy.deepcopy(global_model) for _ in range(len(selected_clients))]
    for m in client_models: m.to(device)

    # 랭크 스케줄러
    sched = RankScheduler(args.rein_r_min, args.rein_r_max,
                          mode=rank_mode, gmm_clean_thr=args.gmm_clean_thr,
                          r_clean=args.r_clean, r_noisy=args.r_noisy)

    # 초기 rank
    for m in client_models:
        if hasattr(m, "set_rank_all"): m.set_rank_all(args.rein_r_init)
        if flavor_name=="BASELINE_FIXED":
            # r_max로 잠금
            if hasattr(m, "lock_rank"): m.lock_rank(args.rein_r_max)

    # 클라이언트 로더
    def make_client_loaders():
        L=[]
        for ci in selected_clients:
            ds = DatasetSplit(dataset_train, dict_users[ci])
            L.append(_make_loader(ds, args.batch_size, args.num_workers, shuffle=True))
        return L
    client_loaders = make_client_loaders()

    # 학습 루프
    E = args.epochs
    for epoch in range(E):
        logging.info(f"[{flavor_name}] epoch {epoch+1}/{E}")

        # 동적이면 epoch 기반 업데이트
        if flavor_name=="DYNAMIC" and rank_mode in ["time_inc","time_dec"]:
            r_e = sched.epoch_rank(epoch, E)
            for m in client_models:
                if hasattr(m, "set_rank_all"): m.set_rank_all(r_e)

        # 로컬 학습
        for li, (model, loader) in enumerate(zip(client_models, client_loaders)):
            model.train()
            for n,p in model.named_parameters():
                p.requires_grad = ('reins' in n) or ('linear_rein' in n)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

            for _ in range(args.local_ep):
                for x,y,_,_ in loader:
                    x=x.to(device); y=y.to(device)
                    feats = model.forward_features_widx(x, idxs=adapter_idx)[:,0,:]
                    logits = model.linear_rein(feats)
                    loss = F.cross_entropy(logits, y)
                    opt.zero_grad(); loss.backward(); opt.step()
                    # 동적만 비활성 봉인
                    if flavor_name=="DYNAMIC" and hasattr(model.reins, "freeze_inactive"):
                        model.reins.freeze_inactive()

        # FedAvg
        global_model = _fedavg(global_model, client_models)

        # 동적 + GMM 적응
        if flavor_name=="DYNAMIC" and rank_mode=="gmm_adapt":
            g_loss = np.zeros(len(dataset_train), dtype=np.float32)
            g_label = np.zeros(len(dataset_train), dtype=np.int32)
            with torch.no_grad():
                glob_loader = _make_loader(DatasetGlobal(dataset_train), args.batch_size, args.num_workers, shuffle=False)
                for x, noisy_y, _, idx in glob_loader:
                    x=x.to(device)
                    feats = global_model.forward_features_widx(x, idxs=adapter_idx)[:,0,:]
                    logits = global_model.linear_rein(feats)
                    ce = F.cross_entropy(logits, noisy_y.to(device), reduction='none').cpu().numpy()
                    g_loss[idx.numpy()] = ce; g_label[idx.numpy()] = noisy_y.numpy()
            pred = classwise_loss_gmm(g_loss, g_label)
            # 클린 비율로 rank 결정
            for li, ci in enumerate(selected_clients):
                idxs = list(dict_users[ci])
                idxs = [i for i in idxs if i < len(pred)]
                sel = pred[idxs]; sel = sel[sel!=-1]
                cr = float((sel==0).mean()) if len(sel) else 1.0
                r_new = sched.gmm_rank(cr)
                if hasattr(client_models[li], "set_rank_all"): client_models[li].set_rank_all(r_new)

        # 글로벌 -> 로컬 브로드캐스트
        for li in range(len(selected_clients)):
            client_models[li].load_state_dict(global_model.state_dict(), strict=True)

        # 평가
        bacc, acc = eval_model(global_model, test_loader, device)
        logging.info(f"[{flavor_name}] Acc={acc:.4f}  BalAcc={bacc:.4f}")

    return global_model

def _fedavg(gmodel, clist):
    gstate = gmodel.state_dict()
    new = {k: sum(c.state_dict()[k] for c in clist)/len(clist) for k in gstate}
    gmodel.load_state_dict(new); return gmodel

def main(args):
    setup_logging()
    device = torch.device(f"cuda:{args.gpu}")
    args.num_users = args.num_clients
    args.n_clients = args.num_clients

    # 데이터 분할: args.num_clients 전체, 학습 사용: round(frac*total)
    total_clients = int(args.num_clients)
    k_train = max(1, min(total_clients, int(round(args.frac*total_clients))))
    selected_clients = list(range(k_train))

    # 데이터
    args.num_users = total_clients
    args.n_clients = total_clients
    dataset_train, dataset_test, dict_users = get_dataset(args)
    y = np.array(dataset_train.targets)
    y_noisy, _, _ = add_noise(args, y, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_noisy
    logging.info(f"dataset={args.dataset}, total={len(dict_users)}, train_clients={k_train}")

    # A) 베이스라인: 고정 rank
    #   - 원본 Reins와 비교하려면 adapter_kind="rein"
    # baseline = train_one(
    #     flavor_name="BASELINE_FIXED",
    #     adapter_kind="dynamic",            # or "rein"
    #     rank_mode="none",                  # 동적 변경 없음
    #     args=args,
    #     dict_users=dict_users,
    #     selected_clients=selected_clients,
    #     dataset_train=dataset_train,
    #     dataset_test=dataset_test,
    #     device=device
    # )

    # B) 동적: epoch 기반 또는 GMM 기반
    dynamic = train_one(
        flavor_name="DYNAMIC",
        adapter_kind="dynamic",
        rank_mode=args.rank_mode,          # "time_inc" | "time_dec" | "gmm_adapt"
        args=args,
        dict_users=dict_users,
        selected_clients=selected_clients,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        device=device
    )
    
    # dynamic = train_one(
    #     flavor_name="DYNAMIC",
    #     adapter_kind="dynamic",
    #     rank_mode='time_inc',          # "time_inc" | "time_dec" | "gmm_adapt"
    #     args=args,
    #     dict_users=dict_users,
    #     selected_clients=selected_clients,
    #     dataset_train=dataset_train,
    #     dataset_test=dataset_test,
    #     device=device
    # )
    
    # dynamic = train_one(
    #     flavor_name="DYNAMIC",
    #     adapter_kind="dynamic",
    #     rank_mode='time_dec',          # "time_inc" | "time_dec" | "gmm_adapt"
    #     args=args,
    #     dict_users=dict_users,
    #     selected_clients=selected_clients,
    #     dataset_train=dataset_train,
    #     dataset_test=dataset_test,
    #     device=device
    # )

if __name__ == "__main__":
    args = args_parser()
    # 기본값 보강
    if not hasattr(args, "epochs"): args.epochs = 20
    if not hasattr(args, "local_ep"): args.local_ep = 5
    if not hasattr(args, "batch_size"): args.batch_size = 64
    if not hasattr(args, "num_workers"): args.num_workers = 4
    if not hasattr(args, "lr"): args.lr = 1e-3
    if not hasattr(args, "gpu"): args.gpu = 0
    if not hasattr(args, "num_clients"): args.num_clients = 20
    if not hasattr(args, "frac"): args.frac = 0.25
    if not hasattr(args, "num_classes"): args.num_classes = 5

    # rank 파라미터
    if not hasattr(args, "rein_r_min"): args.rein_r_min = 4
    if not hasattr(args, "rein_r_max"): args.rein_r_max = 48
    if not hasattr(args, "rein_r_init"): args.rein_r_init = 16
    if not hasattr(args, "rank_mode"): args.rank_mode = "time_dec"  # "time_inc"|"time_dec"|"gmm_adapt"

    if not hasattr(args, "gmm_clean_thr"): args.gmm_clean_thr = 0.6
    if not hasattr(args, "r_clean"): args.r_clean = 48
    if not hasattr(args, "r_noisy"): args.r_noisy = 8

    torch.cuda.set_device(args.gpu)
    main(args)
