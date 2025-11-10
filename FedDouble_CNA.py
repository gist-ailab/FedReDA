# FedDouble_split_adapter_dual_infer.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno, random
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

# --- paths ---
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)

from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from dataset.dataset import get_dataset
from utils.utils import add_noise

# =================== Switches ===================
STEP4_USE_FUSION = True
EVAL_PER_CLIENT_STEP4 = False
EMA_BETA = 0.9
TORCH_COMPILE = False
USE_AMP = True
# ================================================

GMM_MIN_SAMPLES_PER_CLASS = 20
GMM_RANDOM_SEED = 0
DISTILL_T = 2.0
KL_WEIGHT = 1.0
FEAT_MSE_WEIGHT = 0.0
FEDPROX_MU = 5e-4
MASK_UPDATE_EVERY = 2
MASK_MOMENTUM     = 0.8
CLEAN_THRESHOLD   = 0.6

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
    fh = logging.FileHandler('FedDouble_split_adapter_dual_infer.txt', mode='a')
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

    # teacher (clean)
    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    # (원본 학생 경로; 필요시 사용)
    def forward_features3(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins3.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    # fusion teacher (원본 유지)
    def forward_fusion2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
            x = x+self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    # split student branches
    def forward_features3_clean(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins3_clean.forward(x, idx, batch_first=True, has_cls_token=True)
        return x
    def forward_features3_noisy(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins3_noisy.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

    # global noisy (for inference)
    def forward_features2_noisy(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2_noisy.forward(x, idx, batch_first=True, has_cls_token=True)
        return x

# ---------- Eval helpers ----------
@torch.no_grad()
def calculate_accuracy(model, dataloader, device, mode='rein'):
    model.eval()
    preds, targets = [], []
    with torch.inference_mode(), torch.amp.autocast('cuda',enabled=USE_AMP):
        for inputs, t in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            if mode == 'rein':
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'rein2':
                feats = model.forward_features2(inputs)[:, 0, :]
                logits = model.linear_rein2(feats)
            elif mode == 'rein3':
                feats = model.forward_features3(inputs)[:, 0, :]
                logits = model.linear_rein23(feats)
            elif mode == 'fusion':
                feats = model.forward_fusion2(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'split_clean':
                feats = model.forward_features3_clean(inputs)[:,0,:]
                logits = model.linear_rein3_clean(feats)
            elif mode == 'split_noisy':
                feats = model.forward_features3_noisy(inputs)[:,0,:]
                logits = model.linear_rein3_noisy(feats)
            else:
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            preds.append(torch.argmax(logits, dim=1).cpu())
            targets.append(torch.as_tensor(t))
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return balanced_accuracy_score(targets, preds), accuracy_score(targets, preds)

@torch.no_grad()
def evaluate_inference(model, dataloader, device, args):
    """
    Inference modes:
      - clean:  z = z_clean
      - merged: z = z_clean + merge_eps * z_noisy
      - dual:   z = z_clean + lambda * z_noisy
                (if dual_entropy_gate>0, use entropy gate to set lambda in {0,1})
    """
    model.eval()
    mode = getattr(args, "inference_mode", "clean")
    merge_eps = float(getattr(args, "merge_eps", 0.1))
    dual_lambda = float(getattr(args, "dual_lambda", 0.2))
    ent_gate = float(getattr(args, "dual_entropy_gate", -1.0))

    preds, targets = [], []
    with torch.inference_mode():
        for inputs, t in dataloader:
            x = inputs.to(device, non_blocking=True)
            t_cpu = torch.as_tensor(t)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                # clean branch logits
                f_c = model.forward_features2(x)[:,0,:]
                z_c = model.linear_rein2(f_c)

                if mode == "clean":
                    z = z_c

                else:
                    # noisy branch logits (global)
                    f_n = model.forward_features2_noisy(x)[:,0,:]
                    z_n = model.linear_rein2_noisy(f_n)

                    if mode == "merged":
                        z = z_c + merge_eps * z_n

                    elif mode == "dual":
                        if ent_gate is not None and ent_gate > 0:
                            # entropy of clean probs
                            p_c = z_c.softmax(1).clamp_min(1e-8)
                            ent = (-p_c * p_c.log()).sum(1)  # [B]
                            lam = (ent > ent_gate).float().view(-1,1)  # {0,1}
                            z = z_c + lam * z_n  # gate: if high-entropy → rely on noisy
                        else:
                            z = z_c + dual_lambda * z_n
                    else:
                        z = z_c  # fallback

                y_hat = torch.argmax(z, dim=1).cpu()
            preds.append(y_hat)
            targets.append(t_cpu)
    preds = torch.cat(preds).numpy(); targets = torch.cat(targets).numpy()
    return balanced_accuracy_score(targets, preds), accuracy_score(targets, preds)

# ---------- GMM Split (class-wise) ----------
def gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean):
    idx = np.asarray(sample_indices)
    y   = np.asarray(sample_labels)
    L   = np.asarray(sample_loss_mean, dtype=np.float64)
    clean = {}
    for cls in np.unique(y):
        mask = (y == cls)
        idxs = idx[mask]
        Lc   = L[mask]
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

# ---------- Averaging helpers ----------
@torch.no_grad()
def average_adapter_to_global(
    global_model,
    client_models,
    client_prefix="reins3",
    client_head="linear_rein3",
    target_prefix="reins2",
    target_head="linear_rein2",
    strict_shape=True,
):
    g_sd = global_model.state_dict()
    tgt_adapter_keys = [k for k in g_sd.keys() if k.startswith(target_prefix + ".")]
    tgt_head_w = f"{target_head}.weight"
    tgt_head_b = f"{target_head}.bias"
    all_tgt_keys = tgt_adapter_keys + [tgt_head_w, tgt_head_b]

    sums = {k: torch.zeros_like(g_sd[k]) for k in all_tgt_keys}
    n = float(len(client_models))
    for m in client_models:
        msd = m.state_dict()
        for tgt_k in tgt_adapter_keys:
            cli_k = tgt_k.replace(target_prefix + ".", client_prefix + ".")
            if cli_k not in msd:
                raise KeyError(f"Client key not found: {cli_k}")
            if strict_shape and (msd[cli_k].shape != g_sd[tgt_k].shape):
                raise ValueError(f"Shape mismatch: {cli_k} vs {tgt_k} -> {msd[cli_k].shape} vs {g_sd[tgt_k].shape}")
            sums[tgt_k].add_(msd[cli_k])
        for tgt_k, cli_k in [(tgt_head_w, f"{client_head}.weight"),
                             (tgt_head_b, f"{client_head}.bias")]:
            if cli_k not in msd:
                raise KeyError(f"Client head key not found: {cli_k}")
            if strict_shape and (msd[cli_k].shape != g_sd[tgt_k].shape):
                raise ValueError(f"Shape mismatch: {cli_k} vs {tgt_k} -> {msd[cli_k].shape} vs {g_sd[tgt_k].shape}")
            sums[tgt_k].add_(msd[cli_k])
    for k in all_tgt_keys:
        g_sd[k].copy_(sums[k] / n)
    global_model.load_state_dict(g_sd, strict=True)

# ---------- Main ----------
def main(args):
    setup_logging()
    logging.info("="*50)
    logging.info("Starting FedDouble (split-adapter + dual-branch inference) ...")

    # inject new flags if missing
    if not hasattr(args, "inference_mode"):      args.inference_mode = "clean"   # clean|merged|dual
    if not hasattr(args, "merge_eps"):           args.merge_eps = 0.1
    if not hasattr(args, "dual_lambda"):         args.dual_lambda = 0.2
    if not hasattr(args, "dual_entropy_gate"):   args.dual_entropy_gate = -1.0

    device = torch.device(f"cuda:{args.gpu}")
    scaler = torch.amp.GradScaler('cuda',enabled=USE_AMP)

    # ====== Step 0: data & clients ======
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
    loss_ema  = torch.full((NUM_SAMPLES,), float('nan'), dtype=torch.float32)  # CPU
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

    # ====== Step 0: model ======
    global_model = FedDoubleModel(**_small_variant)
    global_model.load_state_dict(
        torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'),
        strict=False
    )
    embed_dim = _small_variant['embed_dim']

    # 원본 구조 유지 + teacher(head)
    global_model.linear_rein  = nn.Linear(embed_dim, args.num_classes)
    global_model.reins2       = copy.deepcopy(global_model.reins)
    global_model.linear_rein2 = nn.Linear(embed_dim, args.num_classes)

    # ---------- split student adapters ----------
    global_model.reins3_clean = copy.deepcopy(global_model.reins)
    global_model.reins3_noisy = copy.deepcopy(global_model.reins)
    global_model.linear_rein3_clean = nn.Linear(embed_dim, args.num_classes)
    global_model.linear_rein3_noisy = nn.Linear(embed_dim, args.num_classes)

    # ---------- global noisy branch for inference ----------
    global_model.reins2_noisy = copy.deepcopy(global_model.reins2)
    global_model.linear_rein2_noisy = nn.Linear(embed_dim, args.num_classes)
    with torch.no_grad():
        global_model.linear_rein2_noisy.weight.copy_(global_model.linear_rein2.weight)
        global_model.linear_rein2_noisy.bias.copy_(global_model.linear_rein2.bias)

    global_model.to(device)

    if TORCH_COMPILE:
        try:
            global_model = torch.compile(global_model)
        except Exception:
            pass

    # 클라이언트 모델 초기화
    client_model_list = [copy.deepcopy(global_model).to(device) for _ in range(args.num_clients)]

    # clean 확률 캐시
    client_clean_prob = [defaultdict(lambda: 1.0) for _ in range(args.num_clients)]

    # ====== Step 1: Pre-train rein ======
    logging.info("Step 1: Pre-training 'rein'")
    for epoch in tqdm(range(args.round1), desc="Step1: Pretrain-Epoch", position=0):
        for client_idx in tqdm(range(args.num_clients), desc="Clients", leave=False, position=1):
            model = client_model_list[client_idx]
            for n, p in model.named_parameters():
                p.requires_grad = ('reins.' in n) or ('linear_rein.' in n)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_p = clients_train_class_num_list[client_idx]

            for _ in range(args.local_ep):
                for inputs, targets, _ in loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda',enabled=USE_AMP):
                        feats  = model.forward_features(inputs)[:,0,:]
                        logits = model.linear_rein(feats) + 0.5*class_p
                        loss   = F.cross_entropy(logits, targets)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
        logging.info(f"epoch {epoch} pretrain done.")

    # ====== Step 2: Average 'rein' -> global.reins2 ======
    logging.info("Step 2: Averaging 'rein' -> global rein2")
    with torch.no_grad():
        for name, _ in global_model.reins2.named_parameters():
            stacked = torch.stack([dict(cm.reins.named_parameters())[name].data for cm in client_model_list])
            dict(global_model.reins2.named_parameters())[name].data.copy_(stacked.mean(dim=0))
        w_sum = sum(cm.linear_rein.weight.data for cm in client_model_list)
        b_sum = sum(cm.linear_rein.bias.data   for cm in client_model_list)
        global_model.linear_rein2.weight.data.copy_(w_sum/len(client_model_list))
        global_model.linear_rein2.bias.data.copy_(b_sum/len(client_model_list))

        # 초기 noisy-global은 clean-global을 복제
        for name, p in global_model.reins2_noisy.named_parameters():
            p.data.copy_(dict(global_model.reins2.named_parameters())[name].data)
        global_model.linear_rein2_noisy.weight.data.copy_(global_model.linear_rein2.weight.data)
        global_model.linear_rein2_noisy.bias.data.copy_(global_model.linear_rein2.bias.data)

    bacc, acc = evaluate_inference(global_model, test_loader, device, args)
    logging.info(f"After Step2  [infer={args.inference_mode}]  BAcc: {bacc*100:.2f}% | Acc: {acc*100:.2f}%")

    # ====== Main Training Loop ======
    for epoch in tqdm(range(args.round3), desc="MainLoop-Epoch", position=0):
        logging.info("="*80)
        logging.info(f"Epoch {epoch+1}/{args.round3}  [infer={args.inference_mode}]")

        # Step 3: init/freeze
        logging.info("Step 3: init/freeze")
        g_sd = global_model.state_dict()
        for client_idx in range(args.num_clients):
            m = client_model_list[client_idx]
            msd = m.state_dict()

            # reins <- global.reins
            for k in g_sd.keys():
                if k.startswith("reins.") and k in msd:
                    msd[k].copy_(g_sd[k])

            # linear_rein 재초기화
            m.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)

            # reins2, linear_rein2 <- global (clean teacher)
            for k in g_sd.keys():
                if (k.startswith("reins2.") or k.startswith("linear_rein2.")) and k in msd:
                    msd[k].copy_(g_sd[k])

            # split adapters init from teacher
            for k, v in g_sd.items():
                if k.startswith("reins2."):
                    k_clean = k.replace("reins2.", "reins3_clean.")
                    k_noisy = k.replace("reins2.", "reins3_noisy.")
                    if k_clean in msd and msd[k_clean].shape == v.shape:
                        msd[k_clean].copy_(v)
                    if k_noisy in msd and msd[k_noisy].shape == v.shape:
                        msd[k_noisy].copy_(v)
            for suf in ("weight","bias"):
                k2  = f"linear_rein2.{suf}"
                k3c = f"linear_rein3_clean.{suf}"
                k3n = f"linear_rein3_noisy.{suf}"
                if (k2 in g_sd and k3c in msd and msd[k3c].shape == g_sd[k2].shape):
                    msd[k3c].copy_(g_sd[k2])
                if (k2 in g_sd and k3n in msd and msd[k3n].shape == g_sd[k2].shape):
                    msd[k3n].copy_(g_sd[k2])

            m.load_state_dict(msd, strict=True)

            # grad 설정
            for n,p in m.named_parameters():
                if 'reins.' in n or 'linear_rein.' in n:
                    p.requires_grad = True
                elif 'reins2.' in n or 'linear_rein2.' in n:
                    p.requires_grad = False
                elif ('reins3_clean.' in n) or ('linear_rein3_clean.' in n) \
                     or ('reins3_noisy.' in n) or ('linear_rein3_noisy.' in n):
                    p.requires_grad = False
                else:
                    p.requires_grad = False

        # Step 4: train rein (원본 유지)
        logging.info(f"Step 4: train 'rein' ({'fusion' if STEP4_USE_FUSION else 'single'})")
        for client_idx in tqdm(range(args.num_clients), desc="Step4 Clients", leave=False, position=1):
            m = client_model_list[client_idx]
            m.train()
            for n,p in m.named_parameters():
                p.requires_grad = ('reins.' in n) or ('linear_rein.' in n)

            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_p = clients_train_class_num_list[client_idx]

            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda',enabled=USE_AMP):
                        feats = m.forward_fusion2(inputs)[:, 0, :] if STEP4_USE_FUSION else m.forward_features(inputs)[:, 0, :]
                        logits = m.linear_rein(feats) + 0.5 * class_p
                        ce_losses = F.cross_entropy(logits, targets, reduction='none')
                        loss = ce_losses.mean()

                    with torch.no_grad():
                        idxs = torch.as_tensor(batch_indices, dtype=torch.long, device='cpu')
                        vals = ce_losses.detach().to(torch.float32).cpu()
                        old  = loss_ema[idxs]
                        is_nan = torch.isnan(old)
                        old[is_nan] = vals[is_nan]
                        old[~is_nan] = EMA_BETA*old[~is_nan] + (1.0-EMA_BETA)*vals[~is_nan]
                        loss_ema[idxs] = old
                        label_buf[idxs] = targets.detach().cpu()

                    opt.zero_grad(set_to_none=True)
                    if USE_AMP:
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss.backward(); opt.step()

            if EVAL_PER_CLIENT_STEP4:
                bacc_c, acc_c = evaluate_inference(m, test_loader, device, args)
                logging.info(f"Client {client_idx} Step4 - BAcc {bacc_c*100:.2f} | Acc {acc_c*100:.2f}")

        # Step 5: GMM split (hysteresis)
        logging.info("Step 5: GMM split + update clean prob (EMA)")
        do_update_mask = (epoch % MASK_UPDATE_EVERY == 0)
        if do_update_mask:
            for client_idx in tqdm(range(args.num_clients), desc="Step5 BuildMasks", leave=False, position=1):
                client_idxs = np.fromiter(dict_users[client_idx], dtype=np.int64)
                vals = loss_ema[client_idxs].numpy()
                valid = ~np.isnan(vals)
                if valid.any():
                    sample_indices   = client_idxs[valid]
                    sample_labels    = label_buf[client_idxs][valid].numpy()
                    sample_loss_mean = vals[valid]
                    clean_bool = gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean)
                    for idx_i, is_clean in clean_bool.items():
                        p_old = client_clean_prob[client_idx][idx_i]
                        p_new = 1.0 if is_clean else 0.0
                        client_clean_prob[client_idx][idx_i] = MASK_MOMENTUM*p_old + (1.0-MASK_MOMENTUM)*p_new
        else:
            logging.info("Step5 skipped (hysteresis)")

        # Step 5b: split-branch만 학습 가능
        for client_idx in range(args.num_clients):
            m = client_model_list[client_idx]
            for n,p in m.named_parameters():
                p.requires_grad = (
                    ('reins3_clean.' in n) or ('linear_rein3_clean.' in n) or
                    ('reins3_noisy.' in n) or ('linear_rein3_noisy.' in n)
                )

        # Step 6: Split-Adapter Distillation
        logging.info("Step 6: Split-Adapter Distillation (Clean→CE on reins3_clean, Noisy→KD on reins3_noisy)")
        global_model.eval()
        g_reins2_sd = {n: p.detach().clone() for n,p in global_model.reins2.named_parameters()}

        branch_use_clean = 0.0
        branch_use_noisy = 0.0
        branch_total     = 0.0

        for client_idx in tqdm(range(args.num_clients), desc="Step6 Clients", leave=False, position=1):
            m = client_model_list[client_idx]
            m.train()

            params = list(m.reins3_clean.parameters()) + list(m.linear_rein3_clean.parameters()) + \
                     list(m.reins3_noisy.parameters()) + list(m.linear_rein3_noisy.parameters())
            opt = torch.optim.AdamW(params, lr=args.lr)

            loader = clients_train_loader_list[client_idx]
            class_p = clients_train_class_num_list[client_idx]

            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    # teacher (global clean)
                    with torch.no_grad(), torch.amp.autocast('cuda',enabled=USE_AMP):
                        t_feats  = global_model.forward_features2(inputs)[:, 0, :]
                        t_logits = global_model.linear_rein2(t_feats) + 0.5 * class_p
                        t_probsT = F.softmax(t_logits / DISTILL_T, dim=1)

                    # routing
                    with torch.no_grad():
                        probs = torch.tensor(
                            [client_clean_prob[client_idx][int(j)] for j in batch_indices],
                            device=inputs.device, dtype=torch.float32
                        )
                        is_clean = probs >= CLEAN_THRESHOLD
                        is_noisy = ~is_clean
                        bc = float(is_clean.sum().item()); bn = float(is_noisy.sum().item()); bt = float(is_clean.numel())
                        branch_use_clean += bc; branch_use_noisy += bn; branch_total += bt

                    loss = 0.0

                    # clean branch
                    if is_clean.any():
                        with torch.amp.autocast('cuda',enabled=USE_AMP):
                            c_feats  = m.forward_features3_clean(inputs[is_clean])[:, 0, :]
                            c_logits = m.linear_rein3_clean(c_feats) + 0.5 * class_p
                            ce = F.cross_entropy(c_logits, targets[is_clean])
                        loss = loss + ce
                        if FEDPROX_MU > 0.0:
                            prox_c = 0.0
                            for (n,p) in m.reins3_clean.named_parameters():
                                gp = g_reins2_sd[n.replace("reins3_clean","reins2")]
                                prox_c = prox_c + (p - gp).pow(2).sum()
                            loss = loss + FEDPROX_MU * prox_c

                    # noisy branch (KD)
                    if is_noisy.any():
                        with torch.amp.autocast('cuda',enabled=USE_AMP):
                            n_feats  = m.forward_features3_noisy(inputs[is_noisy])[:, 0, :]
                            n_logits = m.linear_rein3_noisy(n_feats) + 0.5 * class_p
                            s_logpT  = F.log_softmax(n_logits / DISTILL_T, dim=1)
                            kl = F.kl_div(s_logpT, t_probsT[is_noisy], reduction='batchmean') * (DISTILL_T ** 2)
                            if FEAT_MSE_WEIGHT > 0.0:
                                kl = kl + FEAT_MSE_WEIGHT * F.mse_loss(n_feats, t_feats[is_noisy])
                        loss = loss + KL_WEIGHT * kl
                        if FEDPROX_MU > 0.0:
                            prox_n = 0.0
                            for (n,p) in m.reins3_noisy.named_parameters():
                                gp = g_reins2_sd[n.replace("reins3_noisy","reins2")]
                                prox_n = prox_n + (p - gp).pow(2).sum()
                            loss = loss + FEDPROX_MU * prox_n

                    opt.zero_grad(set_to_none=True)
                    if USE_AMP:
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss.backward(); opt.step()

        # Step 7: Average back to global — BOTH BRANCHES (clean & noisy)
        logging.info("Step 7: Average clean→global.reins2  and  noisy→global.reins2_noisy")
        logging.info(f"[Routing usage] clean={branch_use_clean/max(branch_total,1.0):.3f}, noisy={branch_use_noisy/max(branch_total,1.0):.3f}")

        # clean aggregate
        average_adapter_to_global(
            global_model, client_model_list,
            client_prefix="reins3_clean", client_head="linear_rein3_clean",
            target_prefix="reins2", target_head="linear_rein2", strict_shape=True
        )
        # noisy aggregate
        average_adapter_to_global(
            global_model, client_model_list,
            client_prefix="reins3_noisy", client_head="linear_rein3_noisy",
            target_prefix="reins2_noisy", target_head="linear_rein2_noisy", strict_shape=True
        )

        bacc, acc = evaluate_inference(global_model, test_loader, device, args)
        logging.info(f"[Epoch {epoch+1}]  [infer={args.inference_mode}]  BAcc: {bacc*100:.2f}% | Acc: {acc*100:.2f}%")

    logging.info("FedDouble (split-adapter + dual-branch inference) training finished.")

if __name__ == "__main__":
    args = args_parser()
    # 신규 CLI 옵션이 기존 parser에 없을 수 있으므로 환경변수/기본값으로 덮어씀
    # (util.options에서 커스텀 인자 추가 못하는 경우를 대비)
    import argparse as _ap
    ap = _ap.ArgumentParser(add_help=False)
    ap.add_argument("--inference_mode", type=str, default="clean", choices=["clean","merged","dual"])
    ap.add_argument("--merge_eps", type=float, default=0.1)
    ap.add_argument("--dual_lambda", type=float, default=0.2)
    ap.add_argument("--dual_entropy_gate", type=float, default=-1.0)
    extra, _ = ap.parse_known_args()
    for k,v in vars(extra).items():
        setattr(args, k, v)

    torch.cuda.set_device(args.gpu)
    main(args)
