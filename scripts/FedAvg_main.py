# FedAvg.py
# Plain Federated Averaging baseline (same model/dataset/setup as FedDouble)
# - Backbone frozen, train only single adapter (reins) + linear head
# - No dual adapters, no GMM/distill, no proximal; pure FedAvg with CE + class-prior logit adjustment

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno, random
import multiprocessing as mp
mp.set_executable(sys.executable)
mp.set_start_method("spawn", force=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# --- FedNoRo path ---
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)

# --- Project deps ---
from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from dataset.dataset import get_dataset
from utils.utils import add_noise

# =================== Behavior Knobs ===================
TORCH_COMPILE   = False
EVAL_EVERY      = 1        # evaluate every n global rounds
CLIENT_FRAC_FALLBACK = 1.0 # if args.frac not in options
USE_AMP = True
# ======================================================

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

# --- Logging ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler('FedAvg.txt', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

# --- Dataset split wrapper (compatible with FedDouble) ---
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

def build_loader(ds, args):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers>0 else None,
        prefetch_factor=(2 if args.num_workers>0 else None),
        persistent_workers=(args.num_workers>0),
        pin_memory=True
    )

# --- Model (single adapter path) ---
class FedAvgModel(ReinsDinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # linear head for adapter path
        self.linear_rein = nn.Linear(kwargs['embed_dim'], kwargs.get('num_classes', 1000))

    # convenience
    def extract_cls_rein(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(x, idx, batch_first=True, has_cls_token=True)
        return x[:, 0, :]

# --- Eval ---
def calculate_accuracy(model, dataloader, device):
    model.eval()
    preds_all, t_all = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            feats  = model.extract_cls_rein(inputs)
            logits = model.linear_rein(feats)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            t_all.extend(targets.numpy())
    acc  = accuracy_score(t_all, preds_all)
    bacc = balanced_accuracy_score(t_all, preds_all)
    model.train()
    return bacc, acc

# --- FedAvg aggregator (average only adapter + head) ---
def fedavg_to_global(global_model, client_models, client_sizes):
    weights = np.array(client_sizes, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    with torch.no_grad():
        # collect param dicts
        client_param_dicts = [dict(m.named_parameters()) for m in client_models]
        for name, p_g in global_model.named_parameters():
            if ('reins.' in name) or ('linear_rein.' in name):  # only trainable subset
                p_g.data.zero_()
                for w, cp in zip(weights, client_param_dicts):
                    p_g.data.add_(cp[name].data * float(w))
    return global_model

def broadcast_from_global(global_model, template_model):
    # return a fresh client model copied from global
    m = copy.deepcopy(template_model)
    m.load_state_dict(global_model.state_dict(), strict=True)
    return m

def main(args):
    setup_logging()
    logging.info("="*50)
    logging.info("Starting FedAvg baseline...")
    device = torch.device(f"cuda:{args.gpu}")

    # speed knobs
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass

    # ===================== Step 0: data =====================
    args.num_users  = args.num_clients
    args.n_clients  = args.num_clients
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    args.num_classes = args.n_classes
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    dataset_train.targets = y_train_noisy
    logging.info(f"Dataset '{args.dataset}' | clients={len(dict_users)}")
    logging.info(f"Noisy train dist: {Counter(dataset_train.targets)}")

    # per-client loaders + class-prior (logit adjustment)
    clients_train_loader_list = []
    clients_train_class_num_list = []
    client_sizes = []
    for i in range(args.num_clients):
        client_indices = dict_users[i]
        client_sizes.append(len(client_indices))
        client_dataset = DatasetSplit(dataset_train, client_indices)
        loader = build_loader(client_dataset, args)
        class_num = [0 for _ in range(args.num_classes)]
        for idx in dict_users[i]:
            class_num[int(dataset_train.targets[idx])] += 1
        class_num = torch.cuda.FloatTensor(class_num)
        class_p   = (class_num / class_num.sum()).clamp_min(1e-8).log().view(1, -1)
        clients_train_loader_list.append(loader)
        clients_train_class_num_list.append(class_p)

    test_loader = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers>0 else None,
        prefetch_factor=(2 if args.num_workers>0 else None),
        persistent_workers=(args.num_workers>0),
        pin_memory=True
    )

    # ===================== Step 1: global & template =====================
    # Build global model (backbone frozen; train only reins + linear_rein)
    global_model = FedAvgModel(**_small_variant)
    # attach num_classes info for head init
    if hasattr(args, "num_classes"):
        global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.load_state_dict(
        torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'),
        strict=False
    )
    global_model.to(device)

    # freeze backbone, train only adapter + head
    for n, p in global_model.named_parameters():
        if ('reins.' in n) or ('linear_rein.' in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    if TORCH_COMPILE:
        try: global_model = torch.compile(global_model)
        except Exception: pass

    template_client_model = copy.deepcopy(global_model).to(device)

    # optional warm-up (like FedDouble step1/2): local train once and average -> better init
    WARMUP_EPOCHS = getattr(args, 'round1', 0)
    if WARMUP_EPOCHS > 0:
        logging.info(f"Warm-up: {WARMUP_EPOCHS} epoch(s) of local training then averaging once.")
        scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
        client_models = []
        for ci in range(args.num_clients):
            m = broadcast_from_global(global_model, template_client_model)
            m.train(True)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
            loader = clients_train_loader_list[ci]
            class_p = clients_train_class_num_list[ci]
            for _ in range(WARMUP_EPOCHS):
                for x, y, _, _ in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    if USE_AMP:
                        with torch.amp.autocast('cuda'):
                            f = m.extract_cls_rein(x)
                            logits = m.linear_rein(f) + 0.5*class_p
                            loss = F.cross_entropy(logits, y)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt); scaler.update()
                    else:
                        f = m.extract_cls_rein(x)
                        logits = m.linear_rein(f) + 0.5*class_p
                        loss = F.cross_entropy(logits, y)
                        opt.zero_grad(set_to_none=True)
                        loss.backward(); opt.step()
            client_models.append(m)
        global_model = fedavg_to_global(global_model, client_models, client_sizes)

    bacc, acc = calculate_accuracy(global_model, test_loader, device)
    logging.info(f"[Init] Global BAcc={bacc*100:.2f}% | Acc={acc*100:.2f}%")

    # ===================== Step 2: main FedAvg rounds =====================
    R = getattr(args, 'round3', 50)  # use same naming as FedDouble if present
    frac = getattr(args, 'frac', CLIENT_FRAC_FALLBACK)
    m_clients = max(int(frac * args.num_clients), 1)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    for rnd in tqdm(range(R), desc="FedAvg-Rounds"):
        # sample clients
        all_ids = list(range(args.num_clients))
        random.shuffle(all_ids)
        selected = all_ids[:m_clients]

        # local updates
        local_models, local_sizes = [], []
        for ci in selected:
            m = broadcast_from_global(global_model, template_client_model)
            m.train(True)
            # ensure only adapter+head train
            for n, p in m.named_parameters():
                if ('reins.' in n) or ('linear_rein.' in n):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=args.lr)
            loader = clients_train_loader_list[ci]
            class_p = clients_train_class_num_list[ci]
            for _ in range(args.local_ep):
                for x, y, _, _ in loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    if USE_AMP:
                        with torch.amp.autocast('cuda'):
                            f = m.extract_cls_rein(x)
                            logits = m.linear_rein(f) + 0.5*class_p
                            loss = F.cross_entropy(logits, y)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt); scaler.update()
                    else:
                        f = m.extract_cls_rein(x)
                        logits = m.linear_rein(f) + 0.5*class_p
                        loss = F.cross_entropy(logits, y)
                        opt.zero_grad(set_to_none=True)
                        loss.backward(); opt.step()
            local_models.append(m)
            local_sizes.append(client_sizes[ci])

        # aggregate to global
        global_model = fedavg_to_global(global_model, local_models, local_sizes)

        # eval
        if ((rnd+1) % EVAL_EVERY) == 0:
            bacc, acc = calculate_accuracy(global_model, test_loader, device)
            logging.info(f"[Round {rnd+1:03d}] BAcc={bacc*100:.2f}% | Acc={acc*100:.2f}%")

    logging.info("="*50)
    logging.info("FedAvg training finished.")
    logging.info("="*50)

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
