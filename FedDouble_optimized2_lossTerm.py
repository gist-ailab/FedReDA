# FedDouble_ablate.py — Agreement-masked Dual-Adapter FL (rein1=Dynamic, rein2/3=Fixed Reins)
# - Step1: per-client train rein2 (fixed) → avg to global.reins2 (teacher)
# - Step4: rein2(frozen) + rein1(Dynamic) residual fusion warmup; rein1 re-init every epoch
# - Step5: GMM mask using per-sample loss EMA (first K epochs seeded by teacher CE)
# - Step6: rein3(fixed) init from global.reins2 each round; train rein3 only
# - Step7: (weighted or equal-avg) avg rein3 → global.reins2
# - Train: AMP; Eval: FP32
# - Step6 Loss(3-term; togglable): Clean CE (+label_smooth) + Noisy KD(T) + Adapter Prox

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
from rein.models.backbones.reins import Reins, DynamicReins
from dataset.dataset import get_dataset
from utils.utils import add_noise

# =================== System ===================
USE_AMP = True
TORCH_COMPILE = False
EVAL_PER_CLIENT_STEP4 = False

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

# =================== Ablation & Hyper ===================
# --- EMA / GMM ---
EMA_BETA = 0.9
SEED_EMA_EPOCHS = 2
GMM_MIN_SAMPLES_PER_CLASS = 20
GMM_RANDOM_SEED = 0
MASK_UPDATE_EVERY = 2
MASK_MOMENTUM     = 0.8

# --- (A) Teacher 경로 선택 ---
#   'rein2' | 'fusion2_head2' | 'fusion2_head_rein'
TEACHER_MODE = 'rein2'
FUSION_TEACHER_GAMMA = 0.4   # fusion2에서 residual 강도(고정 게이트)

# ---- Teacher path control (SAFE) ----
TEACHER_USE_REIN2_ONLY   = True    # 항상 rein2만 사용 (권장)
TEACHER_REIN2_SHARPEN    = False   # False: 완전 순수 rein2 / True: rein2 logits 소폭 샤프닝
TEACHER_SHARPEN_T        = 0.8     # < 1.0 이면 샤프닝(온도 낮춤); TEACHER_REIN2_SHARPEN=True일 때만 사용
USE_LOO_TEACHER = True

# --- (B) 마스크 완화/강화 ---
USE_STRICT_MASK = True       # False: GMM 확률만 / True: (agreement&confidence) ∪ GMM
CLEAN_THRESHOLD = 0.6
TAU_G = 0.7                   # strict 모드용 (teacher conf)
TAU_L = 0.7                   # strict 모드용 (student conf)

# --- (C) 초기 equal-average 집계 ---
USE_EQUAL_AVG_EARLY = True
EQUAL_AVG_EPOCHS = 3          # 앞 n라운드는 균등 평균

# --- (D) KD/CE 스무딩 ---
DISTILL_T       = 3.0         # 2.0~4.0 권장
KD_WEIGHT       = 1.0
CE_LABEL_SMOOTH = 0.05        # 0.0 이면 off

# --- Step6 loss 토글 ---
ENABLE_CLEAN_CE     = True
ENABLE_NOISY_KD     = False
ENABLE_ADAPTER_PROX = True
FEDPROX_MU          = 5e-4

# --- Adapter ranks ---
R_MAX_ALL = 32
R1_START  = 16
R2_RANK   = 32  # (Reins 내부 정의 사용)
R3_RANK   = 32  # (Reins 내부 정의 사용)
# ========================================================

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

# ---------- Logging ----------
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler('FedDouble_exteacher.txt', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

# ---------- Dataset ----------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, int(label), self.idxs[item]

def build_loader(ds, args, shuffle=True):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers>0 else None,
        prefetch_factor=(2 if args.num_workers>0 else None),
        persistent_workers=(args.num_workers>0),
        pin_memory=True, drop_last=False
    )

# ---------- Model ----------
class FedDoubleModel(ReinsDinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fusion_gate = nn.Parameter(torch.tensor(0.2))  # 학습가능 게이트(학생용)
    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x
    def forward_features3(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins3.forward(x, idx, batch_first=True, has_cls_token=True)
        return x
    def forward_fusion2(self, x, masks=None, gamma=None):
        # teacher: reins2 (frozen), student: reins (Dynamic residual)
        # gamma=None → 학습가능 self.fusion_gate 사용; gamma!=None → 고정 잔차 강도
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            # with torch.no_grad():
            t = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
            s = self.reins.forward(t.detach(), idx, batch_first=True, has_cls_token=True)
            g = (self.fusion_gate if gamma is None else torch.tensor(gamma, device=t.device, dtype=t.dtype))
            x = t + g * s
        # return s
        return x

# ---------- Eval ----------
@torch.no_grad()
def calculate_accuracy(model, dataloader, device, mode='rein'):
    model = model.to(device); model.eval()
    preds, targets = [], []
    for inputs, t in dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
            if mode == 'rein':
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'rein2':
                feats = model.forward_features2(inputs)[:, 0, :]
                logits = model.linear_rein2(feats)
            elif mode == 'rein3':
                feats = model.forward_features3(inputs)[:, 0, :]
                logits = model.linear_rein3(feats)
            elif mode == 'fusion_teacher_probe':
                feats = model.forward_fusion2(inputs, gamma=FUSION_TEACHER_GAMMA)[:,0,:]
                logits = model.linear_rein2(feats)
            else:
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
        preds.append(torch.argmax(logits, dim=1).cpu())
        targets.append(torch.as_tensor(t))
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return balanced_accuracy_score(targets, preds), accuracy_score(targets, preds)

# ---------- GMM ----------
def gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean):
    idx = np.asarray(sample_indices); y = np.asarray(sample_labels)
    L = np.asarray(sample_loss_mean, dtype=np.float64)
    clean = {}
    for cls in np.unique(y):
        m = (y == cls); idxs = idx[m]; Lc = L[m]
        if idxs.size < GMM_MIN_SAMPLES_PER_CLASS or not np.isfinite(Lc).all():
            for i in idxs: clean[int(i)] = True
            continue
        mu, std = Lc.mean(), Lc.std()
        Lz = (Lc - mu) / (std + 1e-8)
        gmm = GaussianMixture(n_components=2, random_state=GMM_RANDOM_SEED).fit(Lz.reshape(-1,1))
        lab = gmm.predict(Lz.reshape(-1,1))
        means = [Lc[lab==k].mean() if np.any(lab==k) else np.inf for k in range(2)]
        clean_c = int(np.argmin(means))
        for i, z in zip(idxs, lab): clean[int(i)] = (z == clean_c)
    return clean

# ---------- Helpers ----------
def reinit_rein1_and_head(model, embed_dim, num_classes, device):
    model.reins = DynamicReins(dim=embed_dim, r_max=R_MAX_ALL, alpha=1.0, dropout=0.0, pre_norm=True).to(device)
    model.reins.set_rank(R1_START)
    model.linear_rein = nn.Linear(embed_dim, num_classes).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = ('reins.' in n) or ('linear_rein.' in n)

def copy_adapter(src_adapter: nn.Module, dst_adapter: nn.Module):
    with torch.no_grad():
        s = dict(src_adapter.named_parameters()); d = dict(dst_adapter.named_parameters())
        for k in d.keys():
            if k in s and s[k].shape == d[k].shape:
                d[k].data.copy_(s[k].data)

def distribute_loo_teacher(
    client_model_list,
    target_prefix="reins2",            # 각 클라이언트에 덮어쓸 teacher 타깃(서버에서 주는 자리)
    target_head="linear_rein2",
    source_prefix="reins2",            # LOO 평균을 계산할 소스 어댑터(초기 라운드=rein2, 이후 라운드=rein3)
    source_head="linear_rein2",
):
    """
    각 클라이언트 i에 대해: teacher_i  =  average_over_{j != i}( source_j )
    를 계산하여, 클라이언트의 {target_prefix, target_head}에 복사한다.
    글로벌 모델은 건드리지 않음(평가/로깅용으로 full-avg 유지).

    가정: 모든 클라이언트 모델의 source 모듈/헤드는 동일 shape.
    """
    assert len(client_model_list) >= 2, "LOO 평균을 위해 최소 2개 클라이언트가 필요합니다."

    # 1) 키 집합 고정 (첫 번째 클라이언트 기준)
    ref_sd = client_model_list[0].state_dict()
    src_adapter_keys = [k for k in ref_sd.keys() if k.startswith(source_prefix + ".")]
    src_head_w = f"{source_head}.weight"
    src_head_b = f"{source_head}.bias"
    assert src_head_w in ref_sd and src_head_b in ref_sd, "source head가 없습니다."

    # 2) 전체 합계 계산
    sums = {k: torch.zeros_like(ref_sd[k]) for k in src_adapter_keys + [src_head_w, src_head_b]}
    for m in client_model_list:
        msd = m.state_dict()
        for k in src_adapter_keys:
            sums[k].add_(msd[k])
        sums[src_head_w].add_(msd[src_head_w])
        sums[src_head_b].add_(msd[src_head_b])

    N = float(len(client_model_list))
    denom = (N - 1.0)

    # 3) 클라이언트별 LOO = (sum - self) / (N-1) 를 target으로 복사
    for m in client_model_list:
        msd = m.state_dict()
        # adapter
        for k in src_adapter_keys:
            loo = (sums[k] - msd[k]) / denom
            tgt_k = k.replace(source_prefix + ".", target_prefix + ".")
            msd[tgt_k].copy_(loo)
        # head
        loo_w = (sums[src_head_w] - msd[src_head_w]) / denom
        loo_b = (sums[src_head_b] - msd[src_head_b]) / denom
        msd[f"{target_head}.weight"].copy_(loo_w)
        msd[f"{target_head}.bias"].copy_(loo_b)
        m.load_state_dict(msd, strict=False)

# ---------- Main ----------
def main(args):
    setup_logging()
    logging.info("="*60)
    logging.info("FedDouble (ablation-ready)")
    logging.info(f"[TEACHER] mode={TEACHER_MODE} | gamma={FUSION_TEACHER_GAMMA}")
    logging.info(f"[MASK] strict={USE_STRICT_MASK} | CLEAN_THR={CLEAN_THRESHOLD}")
    logging.info(f"[AVG] use_equal_early={USE_EQUAL_AVG_EARLY}, n={EQUAL_AVG_EPOCHS}")
    logging.info(f"[LOSS] CE={ENABLE_CLEAN_CE}(smooth={CE_LABEL_SMOOTH}) | KD={ENABLE_NOISY_KD}(T={DISTILL_T}) | PROX={ENABLE_ADAPTER_PROX}(mu={FEDPROX_MU})")

    device = torch.device(f"cuda:{args.gpu}")
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # Data & clients
    args.num_users = args.num_clients
    args.n_clients = args.num_clients
    dataset_train, dataset_test, dict_users = get_dataset(args)

    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_train_noisy
    args.num_classes = args.n_classes

    logging.info(f"Dataset '{args.dataset}' with {len(dict_users)} clients.")
    logging.info(f"Noisy train dist: {Counter(dataset_train.targets)}")

    NUM_SAMPLES = len(dataset_train)
    loss_ema  = torch.full((NUM_SAMPLES,), float('nan'), dtype=torch.float32)  # CPU buffer
    label_buf = torch.full((NUM_SAMPLES,), -1, dtype=torch.long)

    test_loader = build_loader(dataset_test, args, shuffle=False)

    clients_train_loader_list = []
    clients_train_class_num_list = []
    for i in range(args.num_clients):
        client_indices = dict_users[i]
        client_dataset = DatasetSplit(dataset_train, client_indices)
        train_loader = build_loader(client_dataset, args, shuffle=True)

        class_num = torch.bincount(
            torch.as_tensor([dataset_train.targets[idx] for idx in client_indices], dtype=torch.long),
            minlength=args.num_classes
        ).float().to(device)
        class_p = (class_num / class_num.sum()).clamp_min(1e-8).log().view(1, -1)

        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p)

    # Global model
    global_model = FedDoubleModel(**_small_variant)
    global_model.load_state_dict(torch.load('./checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    embed_dim = _small_variant['embed_dim']

    global_model.reins  = DynamicReins(dim=embed_dim, r_max=R_MAX_ALL, alpha=1.0, dropout=0.0, pre_norm=True).to(device)
    global_model.reins.set_rank(R1_START)
    global_model.reins2 = Reins(num_layers=_small_variant['depth'],
                                embed_dims=_small_variant['embed_dim'],
                                patch_size=_small_variant['patch_size']).to(device)
    global_model.reins3 = Reins(num_layers=_small_variant['depth'],
                                embed_dims=_small_variant['embed_dim'],
                                patch_size=_small_variant['patch_size']).to(device)
    global_model.linear_rein  = nn.Linear(embed_dim, args.num_classes).to(device)
    global_model.linear_rein2 = nn.Linear(embed_dim, args.num_classes).to(device)
    global_model.linear_rein3 = nn.Linear(embed_dim, args.num_classes).to(device)
    global_model = global_model.to(device)

    if TORCH_COMPILE:
        try: global_model = torch.compile(global_model)
        except Exception: pass

    client_model_list = [copy.deepcopy(global_model).to(device) for _ in range(args.num_clients)]
    client_clean_prob = [defaultdict(lambda: 1.0) for _ in range(args.num_clients)]

    # ---------------- Step1: Per-client train rein2 (fixed) ----------------
    logging.info("Step1: Per-client training of rein2 (fixed Reins) + linear_rein2 (CE)")
    for cid in tqdm(range(args.num_clients), desc="Step1 Clients"):
        m = client_model_list[cid]
        for n, p in m.named_parameters():
            p.requires_grad = ('reins2.' in n) or ('linear_rein2.' in n)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
        loader = clients_train_loader_list[cid]; class_p = clients_train_class_num_list[cid]
        for _ in range(args.local_ep):
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    f2 = m.forward_features2(x)[:, 0, :]
                    z2 = m.linear_rein2(f2) + 0.5 * class_p
                    ce = F.cross_entropy(z2, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(ce).backward(); scaler.step(opt); scaler.update()

    # ---------------- Step2: Average rein2 → global.reins2 ----------------
    logging.info("Step2: Averaging clients' rein2 into global.reins2 (teacher init)")
    with torch.no_grad():
        g_params = dict(global_model.reins2.named_parameters())
        sums = {k: torch.zeros_like(v) for k, v in g_params.items()}
        W = torch.zeros_like(global_model.linear_rein2.weight)
        B = torch.zeros_like(global_model.linear_rein2.bias)
        for m in client_model_list:
            s = dict(m.reins2.named_parameters())
            for k in g_params.keys(): sums[k].add_(s[k].data)
            W.add_(m.linear_rein2.weight.data); B.add_(m.linear_rein2.bias.data)
        for k in g_params.keys(): g_params[k].data.copy_(sums[k] / len(client_model_list))
        global_model.linear_rein2.weight.data.copy_(W / len(client_model_list))
        global_model.linear_rein2.bias.data.copy_(B / len(client_model_list))

    if USE_LOO_TEACHER:
        distribute_loo_teacher(client_model_list,
                            target_prefix="reins2", target_head="linear_rein2",
                            source_prefix="reins2", source_head="linear_rein2")

    bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
    logging.info(f"After Step2  BAcc: {bacc*100:.2f}  Acc: {acc*100:.2f}")

    # ---------------- Main Rounds ----------------
    for epoch in tqdm(range(args.round3), desc="MainLoop-Epoch"):
        logging.info("="*60); logging.info(f"Round {epoch+1}/{args.round3}")

        # Step3: sync teacher to clients (LOO) ; prepare rein3
        if USE_LOO_TEACHER:
            # LOO source 선택: 0라운드=rein2, 이후 라운드=rein3
            src_prefix = "reins2" if epoch == 0 else "reins3"
            src_head   = "linear_rein2" if epoch == 0 else "linear_rein3"

            # 각 클라의 target teacher 자리에 LOO 평균을 배포
            distribute_loo_teacher(
                client_model_list,
                target_prefix="reins2", target_head="linear_rein2",
                source_prefix=src_prefix, source_head=src_head
            )
        else:
            # 기존: 글로벌 teacher를 그대로 복사
            g_sd = global_model.state_dict()
            for cid in range(args.num_clients):
                m = client_model_list[cid]; msd = m.state_dict()
                for k in g_sd.keys():
                    if (k.startswith("reins2.") or k.startswith("linear_rein2.")) and k in msd:
                        msd[k].copy_(g_sd[k])
                m.load_state_dict(msd, strict=False)

        # 공통: teacher 고정, rein3는 매 라운드 글로벌.reins2에서 초기화
        for cid in range(args.num_clients):
            m = client_model_list[cid]
            for n, p in m.named_parameters():
                if 'reins2.' in n or 'linear_rein2.' in n: p.requires_grad = False

            # 스펙: rein3는 항상 글로벌 teacher에서 초기화
            copy_adapter(global_model.reins2, m.reins3)
            m.linear_rein3.weight.data.copy_(global_model.linear_rein2.weight.data)
            m.linear_rein3.bias.data.copy_(global_model.linear_rein2.bias.data)

            # 기본적으로 모두 freeze. 이후 단계에서 필요한 모듈만 requires_grad=True로 켬
            for n, p in m.named_parameters():
                p.requires_grad = False

        # Step4: fusion warmup; rein1 reinit each epoch
        logging.info("Step4: Fusion warmup (rein2 frozen) + rein1 reinit each epoch")
        for cid in tqdm(range(args.num_clients), desc="Step4 Clients", leave=False):
            m = client_model_list[cid]
            reinit_rein1_and_head(m, embed_dim, args.num_classes, device)
            for n, p in m.named_parameters():
                if 'reins2.' in n or 'linear_rein2.' in n or 'reins3.' in n or 'linear_rein3.' in n:
                    p.requires_grad = False
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
            loader = clients_train_loader_list[cid]; class_p = clients_train_class_num_list[cid]
            for _ in range(args.local_ep):
                for x, y, idxs in loader:
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        f  = m.forward_fusion2(x)[:,0,:]
                        z  = m.linear_rein(f) + 0.5 * class_p
                        ce_student = F.cross_entropy(z, y, reduction='none')
                        loss = ce_student.mean()
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                        x32 = x.float()
                        ft  = m.forward_features2(x32)[:,0,:]
                        zt  = m.linear_rein2(ft) + 0.5 * class_p.float()
                        ce_teacher = F.cross_entropy(zt, y, reduction='none')
                    with torch.no_grad():
                        ce_for_ema = ce_teacher if (epoch < SEED_EMA_EPOCHS) else ce_student
                        idxs_cpu = torch.as_tensor(idxs, dtype=torch.long, device='cpu')
                        vals     = ce_for_ema.detach().to(torch.float32).cpu()
                        old      = loss_ema[idxs_cpu]; nanmask = torch.isnan(old)
                        old[nanmask]  = vals[nanmask]
                        old[~nanmask] = EMA_BETA * old[~nanmask] + (1.0 - EMA_BETA) * vals[~nanmask]
                        loss_ema[idxs_cpu] = old
                        label_buf[idxs_cpu] = y.detach().cpu()
                    opt.zero_grad(set_to_none=True)
                    if USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else: loss.backward(); opt.step()

            if EVAL_PER_CLIENT_STEP4:
                bacc_c, acc_c = calculate_accuracy(m, test_loader, device, mode='rein2')
                logging.info(f"Client {cid} Step4 sanity — BAcc {bacc_c*100:.2f} Acc {acc_c*100:.2f}")

        # Step5: build/update masks
        logging.info("Step5: Build masks via class-wise GMM + EMA (hysteresis)")
        do_update_mask = (epoch % MASK_UPDATE_EVERY == 0)
        if do_update_mask:
            for cid in tqdm(range(args.num_clients), desc="Step5 BuildMasks", leave=False):
                cidxs = np.fromiter(dict_users[cid], dtype=np.int64)
                vals  = loss_ema[cidxs].numpy(); valid = ~np.isnan(vals)
                if valid.any():
                    sample_indices   = cidxs[valid]
                    sample_labels    = label_buf[cidxs][valid].numpy()
                    sample_loss_mean = vals[valid]
                    clean_bool = gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean)
                    for idx_i, is_clean in clean_bool.items():
                        p_old = client_clean_prob[cid][idx_i]; p_new = 1.0 if is_clean else 0.0
                        client_clean_prob[cid][idx_i] = MASK_MOMENTUM * p_old + (1.0 - MASK_MOMENTUM) * p_new
        else:
            logging.info("Step5 skipped (hysteresis cadence)")

        # Step5b: enable rein3 only
        for cid in range(args.num_clients):
            m = client_model_list[cid]
            for n, p in m.named_parameters():
                p.requires_grad = ('reins3.' in n) or ('linear_rein3.' in n)

        # Step6: train rein3 (compact 3-term)
        logging.info("Step6: Train rein3 (compact 3-term; ablation-ready)")
        global_model.eval()
        g_reins2_sd = {n: p.detach().clone() for n, p in global_model.reins2.named_parameters()}

        client_clean_counts = np.zeros(args.num_clients, dtype=np.float64)
        client_total_counts = np.zeros(args.num_clients, dtype=np.float64)

        for cid in tqdm(range(args.num_clients), desc="Step6 Clients", leave=False):
            m = client_model_list[cid]; m.train()
            opt = torch.optim.AdamW(list(m.reins3.parameters()) + list(m.linear_rein3.parameters()), lr=args.lr)
            loader = clients_train_loader_list[cid]; class_p = clients_train_class_num_list[cid]

            for _ in range(args.local_ep):
                for x, y, idxs in loader:
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

                    # student
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        l_feat  = m.forward_features3(x)[:,0,:]
                        l_logit = m.linear_rein3(l_feat) + 0.5 * class_p
                        l_prob  = l_logit.softmax(1)

                    # teacher (choose route)
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                        x32 = x.float()

                        if TEACHER_USE_REIN2_ONLY:
                            # 강제 rein2 경로
                            g_feat  = m.forward_features2(x32)[:,0,:]
                            g_logit = m.linear_rein2(g_feat) + 0.5 * class_p.float()
                            Tt = TEACHER_SHARPEN_T if TEACHER_REIN2_SHARPEN else 1.0
                            g_prob = F.softmax(g_logit / Tt, dim=1)
                        else:
                            if TEACHER_MODE == 'rein2':
                                g_feat  = m.forward_features2(x32)[:,0,:]
                                g_logit = m.linear_rein2(g_feat) + 0.5 * class_p.float()
                            elif TEACHER_MODE == 'fusion2_head2':
                                g_feat  = m.forward_fusion2(x32, gamma=FUSION_TEACHER_GAMMA)[:,0,:]
                                g_logit = m.linear_rein2(g_feat) + 0.5 * class_p.float()
                            elif TEACHER_MODE == 'fusion2_head_rein':
                                g_feat  = m.forward_fusion2(x32, gamma=FUSION_TEACHER_GAMMA)[:,0,:]
                                g_logit = m.linear_rein(g_feat) + 0.5 * class_p.float()
                            else:
                                raise ValueError(f"Unknown TEACHER_MODE: {TEACHER_MODE}")
                            g_prob = F.softmax(g_logit, dim=1)

                    # mask
                    with torch.no_grad():
                        probs = torch.tensor([client_clean_prob[cid][int(j)] for j in idxs],
                                             device=x.device, dtype=torch.float32)
                        if USE_STRICT_MASK:
                            g_pred = g_prob.argmax(1); l_pred = l_prob.argmax(1)
                            g_conf = g_prob.max(1).values; l_conf = l_prob.max(1).values
                            agree = (g_pred == l_pred) & (g_conf >= TAU_G) & (l_conf >= TAU_L)
                            is_clean = agree | (probs >= CLEAN_THRESHOLD)
                        else:
                            is_clean = (probs >= CLEAN_THRESHOLD)

                        client_clean_counts[cid] += float(is_clean.sum().item())
                        client_total_counts[cid] += float(is_clean.numel())

                    # loss
                    l_logit = l_logit.float(); loss = 0.0

                    # (1) Clean CE (+label smoothing)
                    if ENABLE_CLEAN_CE and is_clean.any():
                        loss = loss + F.cross_entropy(
                            l_logit[is_clean], y[is_clean],
                            label_smoothing=float(CE_LABEL_SMOOTH)
                        )

                    # (2) Noisy KD
                    nz = ~is_clean
                    if ENABLE_NOISY_KD and nz.any():
                        s_logpT = (l_logit[nz] / DISTILL_T).log_softmax(1)
                        with torch.no_grad():
                            # KD용 teacher 분포는 KD-온도(DISTILL_T)로 맞춥니다.
                            t_pT = (g_logit[nz] / DISTILL_T).softmax(1)
                        loss = loss + KD_WEIGHT * F.kl_div(s_logpT, t_pT, reduction='batchmean') * (DISTILL_T**2)

                    # (3) Adapter Prox
                    if ENABLE_ADAPTER_PROX and FEDPROX_MU > 0.0:
                        prox = 0.0
                        for (n, p) in m.reins3.named_parameters():
                            gp = g_reins2_sd[n]
                            prox = prox + (p - gp).pow(2).sum()
                        loss = loss + FEDPROX_MU * prox

                    opt.zero_grad(set_to_none=True)
                    if USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else: loss.backward(); opt.step()

        # Step7: average back to global
        logging.info("Step7: Average rein3 → global.reins2")
        eps = 1e-8
        client_clean_pct = (client_clean_counts + eps) / (client_total_counts + eps)

        if USE_EQUAL_AVG_EARLY and epoch < EQUAL_AVG_EPOCHS:
            weights = np.ones(args.num_clients, dtype=np.float64) / args.num_clients
        else:
            weights = client_clean_pct / (client_clean_pct.sum() + eps)

        global_clean_pct = float(client_clean_counts.sum() / (client_total_counts.sum() + eps))

        g_sd = global_model.state_dict()
        tgt_keys = [k for k in g_sd.keys() if k.startswith("reins2.")]
        sums = {k: torch.zeros_like(g_sd[k]) for k in tgt_keys + ["linear_rein2.weight","linear_rein2.bias"]}

        for wi, m in zip(weights, client_model_list):
            msd = m.state_dict(); w = float(wi)
            for tk in tgt_keys:
                ck = tk.replace("reins2.", "reins3.")
                sums[tk].add_(msd[ck] * w)
            sums["linear_rein2.weight"].add_(msd["linear_rein3.weight"] * w)
            sums["linear_rein2.bias"  ].add_(msd["linear_rein3.bias"]   * w)

        for k in tgt_keys: g_sd[k].copy_(sums[k])
        g_sd["linear_rein2.weight"].copy_(sums["linear_rein2.weight"])
        g_sd["linear_rein2.bias"  ].copy_(sums["linear_rein2.bias"])
        global_model.load_state_dict(g_sd, strict=True)

        bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
        logging.info(
            f"[Round {epoch+1}] "
            f"BAcc {bacc*100:.2f}  Acc {acc*100:.2f}  | "
            f"global_clean_pct={global_clean_pct:.3f}  | "
            f"client_clean_pct={[float(x) for x in client_clean_pct]}  | "
            f"weights={[float(w) for w in weights]}"
        )

    logging.info("FedDouble training finished.")

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
