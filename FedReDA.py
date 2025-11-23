# FedReDA.py
# - Adapter1 (reins): per-round student, trained on clients, re-init from global each round
# - Adapter2 (reins2): server / teacher, built from Adapter1 via FedAvg or LOO
# - Step1: per-client train adapter1 → avg → global.adapter2 (initial teacher)
# - Main loop:
#     * if USE_LOO_TEACHER: client-wise LOO teacher from clients' adapter1
#       else: shared FedAvg teacher from global.adapter2
#     * Init adapter1 from global.adapter2
#     * Train adapter1 with GMM+agreement-masked loss
#     * Avg adapter1 → update global.adapter2 (noise-aware weighted FedAvg)
# - Train: AMP; Eval: FP32

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno
import multiprocessing as mp

mp.set_executable(sys.executable)
mp.set_start_method("spawn", force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.mixture import GaussianMixture

# ---------- Paths ----------
fednoro_path = './other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)

from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from rein.models.backbones.reins import Reins
from dataset.dataset import get_dataset
from utils.utils import add_noise
import time

# =================== System ===================
USE_AMP = True
TORCH_COMPILE = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4)//2))
except Exception:
    pass
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
torch.backends.cudnn.benchmark = True

# =================== Hyper & Mask ===================
EMA_BETA = 0.9
GMM_MIN_SAMPLES_PER_CLASS = 20
GMM_RANDOM_SEED = 0
MASK_UPDATE_EVERY = 2
MASK_MOMENTUM     = 0.8

#Hyper Parameter
USE_LOO_TEACHER         = True   # True: LOO teacher, False: shared FedAvg teacher
ENABLE_NOISY_KD     = True
USE_AGREE_MASK = True
ENABLE_ADAPTER_PROX = True

# Teacher behavior
TEACHER_USE_REIN2_ONLY  = True   # 항상 adapter2(reins2)만 teacher
TEACHER_REIN2_SHARPEN   = False
TEACHER_SHARPEN_T       = 0.8    # <1이면 샤프닝

# Mask
USE_GMM_EMA = True
CLEAN_THRESHOLD = 0.6
TAU_G = 0.7
TAU_L = 0.7

# 초기 equal-average 집계
USE_EQUAL_AVG_EARLY = True
EQUAL_AVG_EPOCHS = 3

# Distill / CE
DISTILL_T       = 3.0
KD_WEIGHT       = 1.0
CE_LABEL_SMOOTH = 0.05

# Loss 토글
ENABLE_CLEAN_CE     = True
FEDPROX_MU          = 5e-4

# Adapter ranks (Reins 내부에서 사용, 여기서는 구조 통일용)
R2_RANK = 32
R1_RANK = 32
# ========================================================

def worker_init_fn(_):
    tmpdir = f"/home/work/DATA1/tmp/worker_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

_old_rmtree = shutil.rmtree
def safe_rmtree(path, *a, **kw):
    try:
        return _old_rmtree(path, *a, **kw)
    except OSError as e:
        if e.errno == errno.EBUSY:
            return
        raise
shutil.rmtree = safe_rmtree

# ---------- Logging ----------
def setup_logging(flag=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fname = 'results/FedDouble_KD2.txt' if flag is None else f'results/FedDouble_KD2_{flag}.txt'
    fh = logging.FileHandler(fname, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
class ComputeTracker:
    def __init__(self, num_gpus=1):
        self.num_gpus = num_gpus
        self.train_seconds = 0.0
        self.infer_seconds = 0.0
        self.train_samples = 0
        self.infer_samples = 0

    @property
    def train_gpu_hours(self):
        return self.train_seconds * self.num_gpus / 3600.0

    @property
    def infer_gpu_hours(self):
        return self.infer_seconds * self.num_gpus / 3600.0

COMPUTE = ComputeTracker(num_gpus=1) 

# ---------- Dataset ----------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, int(label), self.idxs[item]

def build_loader(ds, args, shuffle=True):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        persistent_workers=(args.num_workers > 0),
        pin_memory=True, drop_last=False
    )

# ---------- Model: 2-adapter ----------
class FedDoubleModel(ReinsDinoVisionTransformer):
    """
    Adapter1: self.reins   (Reins) - student
    Adapter2: self.reins2  (Reins) - teacher (global / LOO)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # teacher path using reins2
    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

# ---------- Eval ----------
@torch.no_grad()
def calculate_accuracy(model, dataloader, device, mode='student'):
    model = model.to(device)
    model.eval()
    preds, targets = [], []
    for inputs, t in dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
            if mode == 'student':
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'teacher':
                feats = model.forward_features2(inputs)[:, 0, :]
                logits = model.linear_rein2(feats)
            else:
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
        preds.append(torch.argmax(logits, dim=1).cpu())
        targets.append(torch.as_tensor(t))
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return balanced_accuracy_score(targets, preds), accuracy_score(targets, preds)

@torch.no_grad()
def evaluate_with_timing(model, dataloader, device, mode='teacher', track_compute=False):
    model = model.to(device)
    model.eval()
    preds, targets = [], []
    n_samples = 0
    t0 = time.time()

    for inputs, t in dataloader:
        bs = inputs.size(0)
        n_samples += bs
        inputs = inputs.to(device, non_blocking=True).float()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
            if mode == 'student':
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'teacher':
                feats = model.forward_features2(inputs)[:, 0, :]
                logits = model.linear_rein2(feats)
            else:
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
        preds.append(torch.argmax(logits, dim=1).cpu())
        targets.append(torch.as_tensor(t))

    t1 = time.time()
    if track_compute:
        COMPUTE.infer_seconds += (t1 - t0)
        COMPUTE.infer_samples += n_samples

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return balanced_accuracy_score(targets, preds), accuracy_score(targets, preds)

# ---------- GMM ----------
def gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean):
    idx = np.asarray(sample_indices)
    y = np.asarray(sample_labels)
    L = np.asarray(sample_loss_mean, dtype=np.float64)
    clean = {}
    for cls in np.unique(y):
        m = (y == cls)
        idxs = idx[m]
        Lc = L[m]
        if idxs.size < GMM_MIN_SAMPLES_PER_CLASS or not np.isfinite(Lc).all():
            for i in idxs:
                clean[int(i)] = True
            continue
        mu, std = Lc.mean(), Lc.std()
        Lz = (Lc - mu) / (std + 1e-8)
        gmm = GaussianMixture(
            n_components=2,
            random_state=GMM_RANDOM_SEED
        ).fit(Lz.reshape(-1, 1))
        lab = gmm.predict(Lz.reshape(-1, 1))
        means = [Lc[lab == k].mean() if np.any(lab == k) else np.inf for k in range(2)]
        clean_c = int(np.argmin(means))
        for i, z in zip(idxs, lab):
            clean[int(i)] = (z == clean_c)
    return clean

# ---------- Helpers ----------
def init_student_adapter_from_scratch(model, embed_dim, num_classes, device):
    model.reins = Reins(
        num_layers=_small_variant['depth'],
        embed_dims=embed_dim,
        patch_size=_small_variant['patch_size']
    ).to(device)
    model.linear_rein = nn.Linear(embed_dim, num_classes).to(device)

def copy_adapter(src_adapter: nn.Module, dst_adapter: nn.Module):
    with torch.no_grad():
        s = dict(src_adapter.named_parameters())
        d = dict(dst_adapter.named_parameters())
        for k in d.keys():
            if k in s and s[k].shape == d[k].shape:
                d[k].data.copy_(s[k].data)

def distribute_loo_teacher(
    client_model_list,
    target_prefix="reins2",
    target_head="linear_rein2",
    source_prefix="reins",
    source_head="linear_rein",
):
    """
    LOO teacher:
      teacher_i(target) = average_{j != i}( source_j )
    """
    assert len(client_model_list) >= 2, "LOO 평균을 위해 최소 2개 클라이언트 필요"

    ref_sd = client_model_list[0].state_dict()
    src_adapter_keys = [k for k in ref_sd.keys() if k.startswith(source_prefix + ".")]
    src_head_w = f"{source_head}.weight"
    src_head_b = f"{source_head}.bias"
    assert src_head_w in ref_sd and src_head_b in ref_sd, "source head 없음"

    sums = {k: torch.zeros_like(ref_sd[k]) for k in src_adapter_keys + [src_head_w, src_head_b]}
    for m in client_model_list:
        msd = m.state_dict()
        for k in src_adapter_keys:
            sums[k].add_(msd[k])
        sums[src_head_w].add_(msd[src_head_w])
        sums[src_head_b].add_(msd[src_head_b])

    N = float(len(client_model_list))
    denom = (N - 1.0)

    for m in client_model_list:
        msd = m.state_dict()
        for k in src_adapter_keys:
            loo = (sums[k] - msd[k]) / denom
            tgt_k = k.replace(source_prefix + ".", target_prefix + ".")
            msd[tgt_k].copy_(loo)
        loo_w = (sums[src_head_w] - msd[src_head_w]) / denom
        loo_b = (sums[src_head_b] - msd[src_head_b]) / denom
        msd[f"{target_head}.weight"].copy_(loo_w)
        msd[f"{target_head}.bias"].copy_(loo_b)
        m.load_state_dict(msd, strict=False)

def distribute_fedavg_teacher_from_global(client_model_list, global_model):
    """
    FedAvg teacher:
      모든 클라이언트가 동일한 global.adapter2(reins2/linear_rein2)를 teacher로 사용.
    """
    g_sd = global_model.state_dict()
    for m in client_model_list:
        msd = m.state_dict()
        for k, v in g_sd.items():
            if (k.startswith("reins2.") or k.startswith("linear_rein2.")) and k in msd:
                msd[k].copy_(v)
        m.load_state_dict(msd, strict=False)

# ---------- Main ----------
def main(args):
    global COMPUTE
    flag = str((args.level_n_upperb + args.level_n_lowerb) / 2) +\
            f"_USE_LOO_TEACHER={USE_LOO_TEACHER}"+\
            f'_KD={ENABLE_NOISY_KD}' +\
            f'_Confience_MASK={USE_AGREE_MASK}'
    setup_logging(flag=flag)
    logging.info("=" * 60)
    logging.info("FedDouble-2Adapter (Reins-only)")
    logging.info(f"USE_LOO_TEACHER={USE_LOO_TEACHER}")
    logging.info(f"[MASK] GMM_MASK={USE_GMM_EMA} | Confience_MASK={USE_AGREE_MASK} | CLEAN_THR={CLEAN_THRESHOLD}")
    logging.info(f"[AVG] use_equal_early={USE_EQUAL_AVG_EARLY}, n={EQUAL_AVG_EPOCHS}")
    logging.info(f"[LOSS] CE={ENABLE_CLEAN_CE}(smooth={CE_LABEL_SMOOTH}) | KD={ENABLE_NOISY_KD}(T={DISTILL_T}) | PROX={ENABLE_ADAPTER_PROX}(mu={FEDPROX_MU})")

    device = torch.device(f"cuda:{args.gpu}")
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    t_main_start = time.time()

    # ----- Data -----
    args.num_users = args.num_clients
    args.n_clients = args.num_clients
    dataset_train, dataset_test, dict_users = get_dataset(args)

    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_train_noisy
    args.num_classes = args.n_classes

    logging.info(f"Dataset '{args.dataset}' with {len(dict_users)} clients.")
    logging.info(f"Noisy train dist: {Counter(dataset_train.targets)}")
    # print(len(dataset_train))
    # print(len(dataset_test))
    # exit()

    NUM_SAMPLES = len(dataset_train)
    loss_ema  = torch.full((NUM_SAMPLES,), float('nan'), dtype=torch.float32)
    label_buf = torch.full((NUM_SAMPLES,), -1, dtype=torch.long)

    test_loader = build_loader(dataset_test, args, shuffle=False)

    clients_train_loader_list = []
    clients_train_class_num_list = []
    for cid in range(args.num_clients):
        client_indices = dict_users[cid]
        client_dataset = DatasetSplit(dataset_train, client_indices)
        train_loader = build_loader(client_dataset, args, shuffle=True)

        class_num = torch.bincount(
            torch.as_tensor([dataset_train.targets[idx] for idx in client_indices], dtype=torch.long),
            minlength=args.num_classes
        ).float().to(device)
        class_p = (class_num / class_num.sum()).clamp_min(1e-8).log().view(1, -1)

        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p)

    # ----- Global model -----
    global_model = FedDoubleModel(**_small_variant)
    global_model.load_state_dict(
        torch.load('./checkpoints/dinov2_vits14_pretrain.pth'),
        strict=False
    )
    embed_dim = _small_variant['embed_dim']

    # student adapter1 (scratch)
    init_student_adapter_from_scratch(global_model, embed_dim, args.num_classes, device)

    # teacher adapter2
    global_model.reins2 = Reins(
        num_layers=_small_variant['depth'],
        embed_dims=_small_variant['embed_dim'],
        patch_size=_small_variant['patch_size']
    ).to(device)
    global_model.linear_rein2 = nn.Linear(embed_dim, args.num_classes).to(device)

    global_model = global_model.to(device)
    
    # ----- Model size (parameters) -----
    total_params = sum(p.numel() for p in global_model.parameters())
    adapter_params = sum(
        p.numel() for n, p in global_model.named_parameters()
        if ('reins' in n) or ('linear_rein' in n)
    )
    backbone_params = total_params - adapter_params

    logging.info(f"Total params: {total_params:,} ({total_params/1e6:.3f} M)")
    logging.info(f"Adapter params (reins*/linear_rein*): {adapter_params:,} ({adapter_params/1e6:.3f} M)")
    logging.info(f"Backbone params: {backbone_params:,} ({backbone_params/1e6:.3f} M)")

    if TORCH_COMPILE:
        try:
            global_model = torch.compile(global_model)
        except Exception:
            pass

    # per-client models
    client_model_list = [copy.deepcopy(global_model).to(device) for _ in range(args.num_clients)]
    client_clean_prob = [defaultdict(lambda: 1.0) for _ in range(args.num_clients)]

    # ================== Step1: warmup adapter1 ==================
    logging.info("Step1: Per-client training of Adapter1 (reins) + linear_rein (CE)")
    for warmup_ep in range(args.round1):
        for cid in tqdm(range(args.num_clients), desc="Step1 Clients"):
            m = client_model_list[cid]
            # train only adapter1
            for n, p in m.named_parameters():
                p.requires_grad = ('reins.' in n) or ('linear_rein.' in n)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
            loader = clients_train_loader_list[cid]
            class_p = clients_train_class_num_list[cid]
            for _ in range(args.local_ep):
                for x, y, _ in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    COMPUTE.train_samples += x.size(0)
                    
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        f = m.forward_features(x)[:, 0, :]
                        z = m.linear_rein(f) + 0.5 * class_p
                        ce = F.cross_entropy(z, y)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(ce).backward()
                    scaler.step(opt)
                    scaler.update()

    # ================== Step2: init global teacher from adapter1 ==================
    logging.info("Step2: Averaging Adapter1 → init global Adapter2 (teacher)")
    with torch.no_grad():
        g2 = dict(global_model.reins2.named_parameters())
        sums = {k: torch.zeros_like(v) for k, v in g2.items()}
        W = torch.zeros_like(global_model.linear_rein2.weight)
        B = torch.zeros_like(global_model.linear_rein2.bias)

        for m in client_model_list:
            s = dict(m.reins.named_parameters())
            for k in g2.keys():
                sums[k].add_(s[k])
            W.add_(m.linear_rein.weight.data)
            B.add_(m.linear_rein.bias.data)

        for k in g2.keys():
            g2[k].data.copy_(sums[k] / len(client_model_list))
        global_model.linear_rein2.weight.data.copy_(W / len(client_model_list))
        global_model.linear_rein2.bias.data.copy_(B / len(client_model_list))

    # bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='teacher')
    bacc, acc = evaluate_with_timing(global_model, test_loader, device, mode='teacher', track_compute=True)
    logging.info(f"After Step2 (Teacher init)  BAcc: {bacc*100:.2f}  Acc: {acc*100:.2f}")

    # ================== Main Rounds ==================
    for epoch in tqdm(range(args.round3), desc="MainLoop-Rounds"):
        logging.info("=" * 60)
        logging.info(f"Round {epoch+1}/{args.round3}")

        # ----- Step3: build teachers -----
        if USE_LOO_TEACHER:
            logging.info("Step3: Distribute LOO teachers (Adapter2 from Adapter1)")
            distribute_loo_teacher(
                client_model_list,
                target_prefix="reins2", target_head="linear_rein2",
                source_prefix="reins",  source_head="linear_rein"
            )
        else:
            logging.info("Step3: Distribute shared FedAvg teacher (global Adapter2)")
            distribute_fedavg_teacher_from_global(client_model_list, global_model)

        # freeze teacher params
        for m in client_model_list:
            for n, p in m.named_parameters():
                if 'reins2.' in n or 'linear_rein2.' in n:
                    p.requires_grad = False

        # ----- Step4: init student adapter1 from global teacher each round -----
        logging.info("Step4: Init Adapter1 from global Adapter2 for all clients")
        with torch.no_grad():
            for m in client_model_list:
                copy_adapter(global_model.reins2, m.reins)
                m.linear_rein.weight.data.copy_(global_model.linear_rein2.weight.data)
                m.linear_rein.bias.data.copy_(global_model.linear_rein2.bias.data)

        # enable only adapter1 for training
        for m in client_model_list:
            for n, p in m.named_parameters():
                p.requires_grad = ('reins.' in n) or ('linear_rein.' in n)

        # ----- Step6: train Adapter1 with teacher + mask; update EMA -----
        logging.info("Step6: Train Adapter1 with teacher + mask; update EMA")
        global_model.eval()
        g_reins2_sd = {n: p.detach().clone() for n, p in global_model.reins2.named_parameters()}

        client_clean_counts = np.zeros(args.num_clients, dtype=np.float64)
        client_total_counts = np.zeros(args.num_clients, dtype=np.float64)

        for cid in tqdm(range(args.num_clients), desc="Step6 Clients", leave=False):
            m = client_model_list[cid]
            m.train()
            opt = torch.optim.AdamW(
                list(m.reins.parameters()) + list(m.linear_rein.parameters()),
                lr=args.lr
            )
            loader = clients_train_loader_list[cid]
            class_p = clients_train_class_num_list[cid]

            for _ in range(args.local_ep):
                for x, y, idxs in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    COMPUTE.train_samples += x.size(0)

                    # student
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        s_feat  = m.forward_features(x)[:, 0, :]
                        s_logit = m.linear_rein(s_feat) + 0.5 * class_p
                        s_prob  = s_logit.softmax(1)

                    # teacher
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                        x32 = x.float()
                        t_feat  = m.forward_features2(x32)[:, 0, :]
                        t_logit = m.linear_rein2(t_feat) + 0.5 * class_p.float()
                        if TEACHER_USE_REIN2_ONLY:
                            Tt = TEACHER_SHARPEN_T if TEACHER_REIN2_SHARPEN else 1.0
                            t_prob = F.softmax(t_logit / Tt, dim=1)
                        else:
                            t_prob = F.softmax(t_logit, dim=1)

                    # mask (prev round probs + agreement)
                    with torch.no_grad():
                        if USE_GMM_EMA:
                            probs = torch.tensor(
                                [client_clean_prob[cid][int(j)] for j in idxs],
                                device=x.device, dtype=torch.float32
                            )
                        if USE_AGREE_MASK:
                            t_pred = t_prob.argmax(1)
                            s_pred = s_prob.argmax(1)
                            t_conf = t_prob.max(1).values
                            s_conf = s_prob.max(1).values
                            agree = (t_pred == s_pred) & (t_conf >= TAU_G) & (s_conf >= TAU_L)

                            # is_clean = agree | (probs >= CLEAN_THRESHOLD)
                        if USE_GMM_EMA and USE_AGREE_MASK:
                            is_clean = agree | (probs >= CLEAN_THRESHOLD)
                        elif USE_GMM_EMA:
                            is_clean = (probs >= CLEAN_THRESHOLD)
                        else:
                            is_clean = agree

                        client_clean_counts[cid] += float(is_clean.sum().item())
                        client_total_counts[cid] += float(is_clean.numel())

                        # EMA update (student CE)
                        ce_student = F.cross_entropy(
                            s_logit.float(), y,
                            reduction='none'
                        ).detach().to(torch.float32)
                        idxs_cpu = torch.as_tensor(idxs, dtype=torch.long, device='cpu')
                        vals_cpu = ce_student.cpu()
                        old = loss_ema[idxs_cpu]
                        nanmask = torch.isnan(old)
                        old[nanmask]  = vals_cpu[nanmask]
                        old[~nanmask] = EMA_BETA * old[~nanmask] + (1.0 - EMA_BETA) * vals_cpu[~nanmask]
                        loss_ema[idxs_cpu] = old
                        label_buf[idxs_cpu] = y.detach().cpu()

                    # loss
                    s_logit = s_logit.float()
                    loss = torch.tensor(0.0)

                    if ENABLE_CLEAN_CE and is_clean.any():
                        loss = loss + F.cross_entropy(
                            s_logit[is_clean], y[is_clean],
                            label_smoothing=float(CE_LABEL_SMOOTH)
                        )

                    # nz = np.logical_not(is_clean)
                    nz = ~is_clean
                    if ENABLE_NOISY_KD and nz.any():
                        s_logpT = (s_logit[nz] / DISTILL_T).log_softmax(1)
                        with torch.no_grad():
                            t_pT = (t_logit[nz] / DISTILL_T).softmax(1)
                        loss = loss + KD_WEIGHT * F.kl_div(
                            s_logpT, t_pT,
                            reduction='batchmean'
                        ) * (DISTILL_T ** 2)

                    if ENABLE_ADAPTER_PROX and FEDPROX_MU > 0.0:
                        prox = 0.0
                        for (n, p) in m.reins.named_parameters():
                            gp = g_reins2_sd[n]
                            prox = prox + (p - gp).pow(2).sum()
                        loss = loss + FEDPROX_MU * prox

                    if loss.requires_grad==False:
                        continue
                    
                    opt.zero_grad(set_to_none=True)
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()

        # ----- Step5(A): GMM on EMA -----
        logging.info("Step5(A): Update clean probs via class-wise GMM on EMA")
        if (epoch % MASK_UPDATE_EVERY) == 0:
            for cid in tqdm(range(args.num_clients), desc="Step5 BuildMasks", leave=False):
                cidxs = np.fromiter(dict_users[cid], dtype=np.int64)
                vals = loss_ema[cidxs].numpy()
                valid = ~np.isnan(vals)
                if valid.any():
                    sample_indices   = cidxs[valid]
                    sample_labels    = label_buf[cidxs][valid].numpy()
                    sample_loss_mean = vals[valid]
                    clean_bool = gmm_split_classwise(
                        sample_indices,
                        sample_labels,
                        sample_loss_mean
                    )
                    for idx_i, is_clean in clean_bool.items():
                        p_old = client_clean_prob[cid][idx_i]
                        p_new = 1.0 if is_clean else 0.0
                        client_clean_prob[cid][idx_i] = (
                            MASK_MOMENTUM * p_old + (1.0 - MASK_MOMENTUM) * p_new
                        )
        else:
            logging.info("Step5 skipped (hysteresis cadence)")

        # ----- Step7: aggregate Adapter1 → global Adapter2 -----
        logging.info("Step7: Aggregate Adapter1 → global Adapter2")
        eps = 1e-8
        client_clean_pct = (client_clean_counts + eps) / (client_total_counts + eps)

        if USE_EQUAL_AVG_EARLY and epoch < EQUAL_AVG_EPOCHS:
            weights = np.ones(args.num_clients, dtype=np.float64) / args.num_clients
        else:
            weights = client_clean_pct / (client_clean_pct.sum() + eps)

        global_clean_pct = float(
            client_clean_counts.sum() / (client_total_counts.sum() + eps)
        )

        g_sd = global_model.state_dict()
        tgt_keys = [k for k in g_sd.keys() if k.startswith("reins2.")]
        sums = {
            k: torch.zeros_like(g_sd[k])
            for k in tgt_keys + ["linear_rein2.weight", "linear_rein2.bias"]
        }

        for wi, m in zip(weights, client_model_list):
            msd = m.state_dict()
            w = float(wi)
            for tk in tgt_keys:
                ck = tk.replace("reins2.", "reins.")
                sums[tk].add_(msd[ck] * w)
            sums["linear_rein2.weight"].add_(msd["linear_rein.weight"] * w)
            sums["linear_rein2.bias"].add_(msd["linear_rein.bias"] * w)

        for k in tgt_keys:
            g_sd[k].copy_(sums[k])
        g_sd["linear_rein2.weight"].copy_(sums["linear_rein2.weight"])
        g_sd["linear_rein2.bias"].copy_(sums["linear_rein2.bias"])
        global_model.load_state_dict(g_sd, strict=True)

        # monitor teacher
        # bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='teacher')
        bacc, acc = evaluate_with_timing(global_model, test_loader, device, mode='teacher', track_compute=True)
        logging.info(
            f"[Round {epoch+1}] "
            f"BAcc {bacc*100:.2f}  Acc {acc*100:.2f}  | "
            f"global_clean_pct={global_clean_pct:.3f}  | "
            f"client_clean_pct={[float(x) for x in client_clean_pct]}  | "
            f"weights={[float(w) for w in weights]}"
        )

    logging.info("FedDouble-2Adapter (Reins-only) training finished.")
    
    # ----- Compute summary -----
    total_wall = time.time() - t_main_start

    # 추론 시간 제외한 "approx. training" 시간
    train_only = max(total_wall - COMPUTE.infer_seconds, 0.0)
    COMPUTE.train_seconds = train_only  # 한 번의 run 기준이면 += 보다 = 이 더 안전

    logging.info(f"[Compute] Total wall-clock (train + eval + overhead): {total_wall:.2f} s")
    logging.info(f"[Compute] Approx. training wall-clock (excluding eval): "
                 f"{COMPUTE.train_seconds:.2f} s "
                 f"({COMPUTE.train_gpu_hours:.3f} GPU+CPU hours)")
    
    logging.info(f"[Compute] Total train samples processed (with local_ep & rounds): "
                f"{COMPUTE.train_samples}")
    
    if COMPUTE.infer_samples > 0:
        infer_time_per_1000 = COMPUTE.infer_seconds / COMPUTE.infer_samples * 1000.0
        infer_gpu_hours_per_1000 = infer_time_per_1000 * COMPUTE.num_gpus / 3600.0
        logging.info(f"[Compute] Inference: total {COMPUTE.infer_samples} samples, "
                     f"total wall-clock {COMPUTE.infer_seconds:.2f} s "
                     f"({COMPUTE.infer_gpu_hours:.3f} GPU+CPU hours)")
        logging.info(f"[Compute] Inference per 1000 instances: "
                     f"{infer_time_per_1000:.4f} s "
                     f"({infer_gpu_hours_per_1000:.6f} GPU+CPU hours)")

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
