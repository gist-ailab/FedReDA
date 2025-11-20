# FedDouble.py — Agreement-masked Dual-Adapter FL (rein1=Dynamic, rein2/3=Fixed Reins)
# - Step1: per-client train rein2 (fixed) → avg to global.reins2 (teacher)
# - Step4: rein2(frozen) + rein1(Dynamic) residual fusion warmup; rein1 re-init every epoch
# - Step5: GMM mask using per-sample loss EMA (first K epochs seeded by teacher CE)
# - Step6: rein3(fixed) init from global.reins2 each round; train rein3 only
# - Step7: clean-weighted avg rein3 → global.reins2
# - Train: AMP; Eval: FP32
# - Step6 compact 3-term loss (toggle): L = Clean CE + Noisy KD + Adapter Prox
# - (Optional) Conservative label refinement uses agreement+confidence to update per-sample hard labels.

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

# =================== Switches/Hyper ===================
USE_AMP = True
TORCH_COMPILE = False
EVAL_PER_CLIENT_STEP4 = False

EMA_BETA = 0.9
SEED_EMA_EPOCHS = 2

GMM_MIN_SAMPLES_PER_CLASS = 20
GMM_RANDOM_SEED = 0

# ---- Step6 compact loss toggles ----
ENABLE_CLEAN_CE     = True   # (1) clean CE
ENABLE_NOISY_KD     = True   # (2) noisy KD
ENABLE_ADAPTER_PROX = True   # (3) adapter proximal penalty

DISTILL_T   = 2.0
KD_WEIGHT   = 1.0
FEDPROX_MU  = 5e-4

MASK_UPDATE_EVERY = 2
MASK_MOMENTUM     = 0.8
CLEAN_THRESHOLD   = 0.6

TAU_G = 0.7
TAU_L = 0.7
TAU_G_REFINE = 0.95
TAU_L_REFINE = 0.95

# ---- Conservative label refinement (hard-label) ----
ENABLE_REFINEMENT        = True
REFINE_START_EPOCH       = 1       # start from this main round (0-indexed)
REFINE_REQUIRE_AGREE     = True    # teacher & student must agree
REFINE_CONF_THRESH_G     = 0.8     # unused in final logic; kept for config print
REFINE_CONF_THRESH_L     = 0.8
REFINE_USE_TEACHER_ONLY  = True    # else: avg(teacher,student) logits to pick argmax
REFINE_STICKY_BETTER     = True    # overwrite only if new conf > old

# Adapter ranks
R_MAX_ALL = 64   # rein1(Dynamic) r_max
R1_START  = 8    # rein1 active rank at (re)init
R2_RANK   = 32   # rein2 fixed rank (teacher)
R3_RANK   = 32   # rein3 fixed rank (student)
# ======================================================

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

def worker_init_fn(_):
    tmpdir = f"/home/work/DATA1/tmp/worker_{uuid.uuid4().hex}"
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    tempfile.tempdir = tmpdir

_old_rmtree = shutil.rmtree
def safe_rmtree(path,*a,**kw):
    try:
        return _old_rmtree(path,*a,**kw)
    except OSError as e:
        if e.errno == errno.EBUSY:
            return
        raise
shutil.rmtree = safe_rmtree

# ---------- Logging ----------
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler('FedDouble_optimized_refine.txt', mode='a')
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
    def __len__(self):
        return len(self.idxs)
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
        self.fusion_gate = nn.Parameter(torch.tensor(0.2))  # small residual gate

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

    def forward_fusion2(self, x, masks=None):
        # teacher: rein2 (frozen); student: rein1 (Dynamic) → residual on top of teacher tokens
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            with torch.no_grad():
                t = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
            s = self.reins.forward(t.detach(), idx, batch_first=True, has_cls_token=True)
            x = t + self.fusion_gate * s
        return x

# ---------- Eval (FP32) ----------
@torch.no_grad()
def calculate_accuracy(model, dataloader, device, mode='rein'):
    model = model.to(device)
    model.eval()
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
    idx = np.asarray(sample_indices)
    y   = np.asarray(sample_labels)
    L   = np.asarray(sample_loss_mean, dtype=np.float64)
    clean = {}
    for cls in np.unique(y):
        m = (y == cls)
        idxs = idx[m]; Lc = L[m]
        if idxs.size < GMM_MIN_SAMPLES_PER_CLASS or not np.isfinite(Lc).all():
            for i in idxs: clean[int(i)] = True
            continue
        mu, std = Lc.mean(), Lc.std()
        Lz = (Lc - mu) / (std + 1e-8)
        gmm = GaussianMixture(n_components=2, random_state=GMM_RANDOM_SEED).fit(Lz.reshape(-1,1))
        lab = gmm.predict(Lz.reshape(-1,1))
        means = [Lc[lab==k].mean() if np.any(lab==k) else np.inf for k in range(2)]
        clean_c = int(np.argmin(means))
        for i, z in zip(idxs, lab):
            clean[int(i)] = (z == clean_c)
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
        s = dict(src_adapter.named_parameters())
        d = dict(dst_adapter.named_parameters())
        for k in d.keys():
            if k in s and s[k].shape == d[k].shape:
                d[k].data.copy_(s[k].data)

# ---------- Main ----------
def main(args):
    setup_logging()
    logging.info("="*60)
    logging.info("FedDouble: rein1(Dynamic) only; rein2/3 fixed Reins; teacher from rein2 averaging")
    logging.info(f"[LOSS] CLEAN_CE={ENABLE_CLEAN_CE} | NOISY_KD={ENABLE_NOISY_KD} | ADAPTER_PROX={ENABLE_ADAPTER_PROX}")
    logging.info(f"[REFINE] ENABLE={ENABLE_REFINEMENT} | START={REFINE_START_EPOCH} | AGREE={REFINE_REQUIRE_AGREE} | TAU_G={TAU_G_REFINE} | TAU_L={TAU_L_REFINE} | TEACHER_ONLY={REFINE_USE_TEACHER_ONLY}")

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
    loss_ema   = torch.full((NUM_SAMPLES,), float('nan'), dtype=torch.float32)   # CPU
    label_buf  = torch.full((NUM_SAMPLES,), -1, dtype=torch.long)               # CPU (original/noisy)
    # refinement buffers (CPU)
    refined_label = np.full(NUM_SAMPLES, -1, dtype=np.int64)
    refined_conf  = np.full(NUM_SAMPLES, -np.inf, dtype=np.float32)  # confidence max-keep

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
    global_model = global_model.to(device)

    embed_dim = _small_variant['embed_dim']
    # Adapters: rein1=Dynamic, rein2/3=Fixed
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

    if TORCH_COMPILE:
        try:
            global_model = torch.compile(global_model)
        except Exception:
            pass

    client_model_list = [copy.deepcopy(global_model).to(device) for _ in range(args.num_clients)]
    client_clean_prob = [defaultdict(lambda: 1.0) for _ in range(args.num_clients)]

    # ---------------- Step1: Per-client train rein2 (fixed) ----------------
    logging.info("Step1: Per-client training of rein2 (fixed Reins) + linear_rein2 (CE)")
    for cid in tqdm(range(args.num_clients), desc="Step1 Clients"):
        m = client_model_list[cid]
        for n, p in m.named_parameters():
            p.requires_grad = ('reins2.' in n) or ('linear_rein2.' in n)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
        loader = clients_train_loader_list[cid]
        class_p = clients_train_class_num_list[cid]
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
            for k in g_params.keys():
                sums[k].add_(s[k].data)
            W.add_(m.linear_rein2.weight.data)
            B.add_(m.linear_rein2.bias.data)
        for k in g_params.keys():
            g_params[k].data.copy_(sums[k] / len(client_model_list))
        global_model.linear_rein2.weight.data.copy_(W / len(client_model_list))
        global_model.linear_rein2.bias.data.copy_(B / len(client_model_list))

    bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
    logging.info(f"After Step2  BAcc: {bacc*100:.2f}  Acc: {acc*100:.2f}")

    # ---------------- Main Rounds ----------------
    for epoch in tqdm(range(args.round3), desc="MainLoop-Epoch"):
        logging.info("="*60); logging.info(f"Round {epoch+1}/{args.round3}")

        # Step3: sync teacher to clients; prepare rein3; freeze default
        g_sd = global_model.state_dict()
        for cid in range(args.num_clients):
            m = client_model_list[cid]
            msd = m.state_dict()
            for k in g_sd.keys():
                if (k.startswith("reins2.") or k.startswith("linear_rein2.")) and k in msd:
                    msd[k].copy_(g_sd[k])
            m.load_state_dict(msd, strict=False)
            for n, p in m.named_parameters():
                if 'reins2.' in n or 'linear_rein2.' in n:
                    p.requires_grad = False

            copy_adapter(m.reins2, m.reins3)
            m.linear_rein3.weight.data.copy_(m.linear_rein2.weight.data)
            m.linear_rein3.bias.data.copy_(m.linear_rein2.bias.data)
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
            loader = clients_train_loader_list[cid]
            class_p = clients_train_class_num_list[cid]

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
                        old      = loss_ema[idxs_cpu]
                        nanmask  = torch.isnan(old)
                        old[nanmask]  = vals[nanmask]
                        old[~nanmask] = EMA_BETA * old[~nanmask] + (1.0 - EMA_BETA) * vals[~nanmask]
                        loss_ema[idxs_cpu] = old
                        label_buf[idxs_cpu] = y.detach().cpu()

                    opt.zero_grad(set_to_none=True)
                    if USE_AMP:
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss.backward(); opt.step()

            if EVAL_PER_CLIENT_STEP4:
                bacc_c, acc_c = calculate_accuracy(m, test_loader, device, mode='rein2')
                logging.info(f"Client {cid} Step4 sanity — BAcc {bacc_c*100:.2f} Acc {acc_c*100:.2f}")

        # Step5: build/update masks (hysteresis)
        msg = "Step5: Build masks via class-wise GMM + EMA (hysteresis)"
        if epoch == 0: msg += " [first round may be sparse]"
        logging.info(msg)

        do_update_mask = (epoch % MASK_UPDATE_EVERY == 0)
        if do_update_mask:
            for cid in tqdm(range(args.num_clients), desc="Step5 BuildMasks", leave=False):
                cidxs = np.fromiter(dict_users[cid], dtype=np.int64)
                vals  = loss_ema[cidxs].numpy()
                valid = ~np.isnan(vals)
                if valid.any():
                    sample_indices   = cidxs[valid]
                    sample_labels    = label_buf[cidxs][valid].numpy()
                    sample_loss_mean = vals[valid]
                    clean_bool = gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean)
                    for idx_i, is_clean in clean_bool.items():
                        p_old = client_clean_prob[cid][idx_i]
                        p_new = 1.0 if is_clean else 0.0
                        client_clean_prob[cid][idx_i] = MASK_MOMENTUM * p_old + (1.0 - MASK_MOMENTUM) * p_new
        else:
            logging.info("Step5 skipped (hysteresis cadence)")

        # Step5b: enable rein3 only
        for cid in range(args.num_clients):
            m = client_model_list[cid]
            for n, p in m.named_parameters():
                p.requires_grad = ('reins3.' in n) or ('linear_rein3.' in n)

        # Step6: train rein3 — compact loss (+ optional conservative label refinement)
        logging.info("Step6: Train rein3 (compact 3-term; optional conservative refinement)")
        global_model.eval()
        g_reins2_sd = {n: p.detach().clone() for n, p in global_model.reins2.named_parameters()}

        client_clean_counts = np.zeros(args.num_clients, dtype=np.float64)
        client_total_counts = np.zeros(args.num_clients, dtype=np.float64)

        for cid in tqdm(range(args.num_clients), desc="Step6 Clients", leave=False):
            m = client_model_list[cid]
            m.train()

            opt = torch.optim.AdamW(list(m.reins3.parameters()) + list(m.linear_rein3.parameters()), lr=args.lr)
            loader = clients_train_loader_list[cid]
            class_p = clients_train_class_num_list[cid]

            for _ in range(args.local_ep):
                for x, y, idxs in loader:
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

                    # student
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        l_feat  = m.forward_features3(x)[:,0,:]
                        l_logit = m.linear_rein3(l_feat) + 0.5 * class_p
                        l_prob  = l_logit.softmax(1)

                    # teacher
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                        x32 = x.float()
                        g_feat  = global_model.forward_features2(x32)[:,0,:]
                        g_logit = global_model.linear_rein2(g_feat) + 0.5 * class_p.float()
                        g_prob  = g_logit.softmax(1)

                    # masks
                    with torch.no_grad():
                        g_pred = g_prob.argmax(1); l_pred = l_prob.argmax(1)
                        g_conf = g_prob.max(1).values; l_conf = l_prob.max(1).values
                        agree_mask = (g_pred==l_pred)
                        conf_ok = (g_conf>=TAU_G) & (l_conf>=TAU_L)

                        probs = torch.tensor([client_clean_prob[cid][int(j)] for j in idxs],
                                             device=x.device, dtype=torch.float32)
                        is_clean = (probs >= CLEAN_THRESHOLD)

                        # ---- (Optional) Conservative label refinement ----
                        if ENABLE_REFINEMENT and epoch >= REFINE_START_EPOCH:
                            # 후보 마스크(배치 크기와 동일한 길이)
                            cand = (agree_mask & (g_conf >= TAU_G_REFINE) & (l_conf >= TAU_L_REFINE))

                            # 새 라벨/신뢰도 (배치 차원)
                            if REFINE_USE_TEACHER_ONLY:
                                new_label_tensor = g_pred
                                new_conf_tensor  = g_conf
                            else:
                                avg_prob = 0.5*(g_prob + l_prob)
                                new_conf_tensor, new_label_tensor = avg_prob.max(1)

                            # ### FIX 1: 전역 인덱스와 배치 인덱스를 명확히 분리
                            idxs_np  = idxs.detach().cpu().numpy()            # 전역 인덱스 (N_i)
                            cand_np  = cand.detach().cpu().numpy().astype(bool)
                            nl_np    = new_label_tensor.detach().cpu().numpy() # 길이 = batch_size
                            nc_np    = new_conf_tensor.detach().cpu().numpy()

                            for b, j in enumerate(idxs_np):
                                if not cand_np[b]:
                                    continue
                                if REFINE_STICKY_BETTER:
                                    if nc_np[b] > refined_conf[j]:
                                        refined_label[j] = int(nl_np[b])
                                        refined_conf[j]  = float(nc_np[b])
                                else:
                                    refined_label[j] = int(nl_np[b])
                                    refined_conf[j]  = float(nc_np[b])

                    # ----- compact loss -----
                    l_logit = l_logit.float()
                    loss = 0.0

                    # ### FIX 2: batch 전역 인덱스로 refined label을 안전하게 수집
                    if ENABLE_CLEAN_CE:
                        idxs_np = idxs.detach().cpu().numpy()
                        rlab_cpu = refined_label[idxs_np]                 # numpy (len=batch)
                        has_ref_np = (rlab_cpu != -1)
                        has_ref = torch.from_numpy(has_ref_np).to(y.device, dtype=torch.bool)
                        rlab = torch.from_numpy(rlab_cpu).to(y.device, dtype=torch.long)
                        y_eff = torch.where(has_ref, rlab, y)
                    else:
                        y_eff = y  # unused if CE disabled

                    # (1) Clean CE
                    if ENABLE_CLEAN_CE and is_clean.any():
                        loss = loss + F.cross_entropy(l_logit[is_clean], y_eff[is_clean])

                    # (2) Noisy KD
                    nz = ~is_clean
                    if ENABLE_NOISY_KD and nz.any():
                        s_logpT = (l_logit[nz] / DISTILL_T).log_softmax(1)
                        with torch.no_grad():
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
                    if USE_AMP:
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss.backward(); opt.step()

                    # ### FIX 3: 마스크 통계는 한 번만 집계 (중복 제거)
                    client_clean_counts[cid] += float(is_clean.sum().item())
                    client_total_counts[cid] += float(is_clean.numel())

        # Step7: clean-weighted average (rein3 -> global.reins2)
        logging.info("Step7: Clean-weighted averaging rein3 → global.reins2")
        eps = 1e-8
        client_clean_pct = (client_clean_counts + eps) / (client_total_counts + eps)
        ratios = (client_clean_counts + eps) / (client_total_counts + eps)
        # weights = ratios / (ratios.sum() + eps)
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

        # bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
        # logging.info(f"[Round {epoch+1}] BAcc {bacc*100:.2f}  Acc {acc*100:.2f}  | "
        #              f"mask_use={(epoch % MASK_UPDATE_EVERY == 0)}  "
        #              f"avg_clean_ratio={weights.mean():.3f}")
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
