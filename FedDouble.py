import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno
import sys, multiprocessing as mp
mp.set_executable(sys.executable)
mp.set_start_method("spawn", force=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
from tqdm.auto import tqdm

# --- FedNoRo 모듈 경로 ---
fednoro_path = '/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo'
if fednoro_path not in sys.path:
    sys.path.append(fednoro_path)

# --- 기존 의존성 ---
from util.options import args_parser
from dino_variant import _small_variant
from rein.models.backbones.reins_dinov2 import ReinsDinoVisionTransformer
from update import average_reins
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.mixture import GaussianMixture

# --- 데이터셋 ---
from dataset.dataset import get_dataset
from utils.utils import add_noise

# =================== Speed / Behavior Switches ===================
STEP4_USE_FUSION = True     # True면 forward_fusion2 사용, False면 forward_features 사용
EVAL_PER_CLIENT_STEP4 = False # True면 Step4 끝에 각 클라이언트 평가
EMA_BETA = 0.9               # loss EMA 계수
TORCH_COMPILE = False         # PyTorch 2.x에서 compile 사용(환경 따라 다름)
# ================================================================

GMM_MIN_SAMPLES_PER_CLASS = 20
GMM_RANDOM_SEED = 0
DISTILL_T = 2.0
KL_WEIGHT = 1.0
FEAT_MSE_WEIGHT = 0.0   # 글로벌 feat 정렬 쓰려면 >0로 설정

# [ADD-3] FedProx 강도
FEDPROX_MU = 5e-4

# [ADD-4] 마스크 히스테리시스/저빈도 갱신 설정
MASK_UPDATE_EVERY = 2        # 마스크는 2 에폭마다 갱신
MASK_MOMENTUM     = 0.8      # 마스크 EMA 모멘텀
CLEAN_THRESHOLD   = 0.6 

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

def gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"idx": sample_indices, "y": sample_labels, "L": sample_loss_mean})
    df["L_cz"] = df.groupby("y")["L"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    clean_mask_dict = {}
    for _, grp in df.groupby("y"):
        idxs = grp["idx"].values
        X = grp[["L_cz"]].values
        if len(idxs) < GMM_MIN_SAMPLES_PER_CLASS or not np.isfinite(X).all():
            for i in idxs: clean_mask_dict[int(i)] = True
            continue
        gmm = GaussianMixture(n_components=2, random_state=GMM_RANDOM_SEED).fit(X)
        lab = gmm.predict(X)
        means = grp.assign(cluster=lab).groupby("cluster")["L"].mean().sort_values()
        clean_c = means.index[0]
        for i, z in zip(idxs, lab):
            clean_mask_dict[int(i)] = (z == clean_c)
    return clean_mask_dict

# --- 로거 ---
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler('FedDouble_isic_0.2.txt', mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

def calculate_accuracy(model, dataloader, device, mode='rein'):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:  # 배치 tqdm 제거로 출력 오버헤드 최소화
            inputs = inputs.to(device, non_blocking=True)
            if mode == 'rein':
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            elif mode == 'rein2':
                feats = model.forward_features2(inputs)[:, 0, :]
                logits = model.linear_rein2(feats)
            elif mode == 'fusion':
                feats = model.forward_fusion2(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            else:
                feats = model.forward_features(inputs)[:, 0, :]
                logits = model.linear_rein(feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    model.train()
    return bacc, acc

def ema_update(target_module, source_module, alpha=0.9):
    for (name_t, param_t), (name_s, param_s) in zip(
        target_module.named_parameters(), source_module.named_parameters()
    ):
        assert name_t == name_s, "Parameter name mismatch!"
        param_t.data.mul_(alpha).add_(param_s.data * (1 - alpha))

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # true_label = self.dataset.true_labels[self.idxs[item]]
        index = self.idxs[item]
        return image, label.squeeze(), index

class FedDoubleModel(ReinsDinoVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward_features2(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins2.forward(x, idx, batch_first=True, has_cls_token=True)
        return x
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
        return (self.forward_features(x) + self.forward_features2(x))/2

def build_loader(ds, args):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers>0 else None,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True
    )

def main(args):
    setup_logging()
    logging.info("="*50)
    logging.info("="*50)
    logging.info("Starting FedDouble training process...")
    device = torch.device(f"cuda:{args.gpu}")

    # Speed knobs
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # =============================================================================
    # Step 0
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

    args.num_classes = args.n_classes

    # 전역 per-sample loss EMA/label 버퍼(벡터화)
    NUM_SAMPLES = len(dataset_train)
    loss_ema = torch.full((NUM_SAMPLES,), float('nan'), dtype=torch.float32)  # CPU 텐서
    label_buf = torch.full((NUM_SAMPLES,), -1, dtype=torch.long)

    test_loader = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers>0 else None,
        prefetch_factor=2,
        persistent_workers=True, pin_memory=True
    )
    
    clients_train_loader_list = []
    clients_train_class_num_list = []
    for i in range(args.num_clients):
        client_indices = dict_users[i]
        client_dataset = DatasetSplit(dataset_train, client_indices)
        train_loader = build_loader(client_dataset, args)
        class_num_list = [0 for _ in range(args.num_classes)]
        for idx in dict_users[i]:
            class_num_list[int(dataset_train.targets[idx])] += 1
        class_num_tensor = torch.cuda.FloatTensor(class_num_list)
        class_p_list = (class_num_tensor / class_num_tensor.sum()).clamp_min(1e-8).log().view(1, -1)
        clients_train_loader_list.append(train_loader)
        clients_train_class_num_list.append(class_p_list)

    # Global model
    global_model = FedDoubleModel(**_small_variant)
    global_model.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    global_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.reins2 = copy.deepcopy(global_model.reins)
    global_model.linear_rein2 = nn.Linear(_small_variant['embed_dim'], args.num_classes)
    global_model.to(device)

    if TORCH_COMPILE:
        try:
            global_model = torch.compile(global_model)
        except Exception:
            pass

    client_model_list = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    logging.info("Step 0 Finished: Initialization complete.")
    
    # [ADD-4] 클라이언트별 clean 확률 캐시(초기 1.0=clean)
    client_clean_prob = [defaultdict(lambda: 1.0) for _ in range(args.num_clients)]

    # =============================================================================
    # Step 1: Pre-train
    # =============================================================================
    logging.info("="*50)
    logging.info("Step 1: Pre-training client 'rein' adapters for epoch1")
    logging.info("="*50)
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in tqdm(range(args.round1), desc="Step1: Pretrain-Epoch", position=0):
        for client_idx in tqdm(range(args.num_clients), desc="Clients", leave=False, position=1):
            model = client_model_list[client_idx]
            model.train()
            for name, param in model.named_parameters():
                if 'reins.' in name or 'linear_rein.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]

            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        feats  = model.forward_features(inputs)[:,0,:]
                        logits = model.linear_rein(feats) + 0.5*class_list
                        loss   = F.cross_entropy(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        logging.info(f"epoch {epoch} pre-training finished.")

    # =============================================================================
    # Step 2: Average to global rein2
    # =============================================================================
    logging.info("="*50)
    logging.info("Step 2: Averaging pre-trained 'rein' adapters to global model")
    logging.info("="*50)
    with torch.no_grad():
        reins_named_params = {}
        for name, _ in client_model_list[0].reins.named_parameters():
            stacked = torch.stack([dict(client.reins.named_parameters())[name].data for client in client_model_list])
            reins_named_params[name] = stacked.mean(dim=0)
        for name, param in global_model.reins2.named_parameters():
            if name in reins_named_params:
                param.data.copy_(reins_named_params[name])
        weight_sum = sum(client.linear_rein.weight.data for client in client_model_list)
        bias_sum = sum(client.linear_rein.bias.data for client in client_model_list)
        global_model.linear_rein2.weight.data.copy_(weight_sum / len(client_model_list))
        global_model.linear_rein2.bias.data.copy_(bias_sum / len(client_model_list))
    
    bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
    logging.info(f"Global Model after Step 2 - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")

    # =============================================================================
    # Main Training Loop
    # =============================================================================
    for epoch in tqdm(range(args.round3), desc="MainLoop-Epoch", position=0):
        logging.info("="*80)
        logging.info(f"Main Training Loop: Starting Epoch {epoch + 1}/{args.round3}")
        logging.info("="*80)

        # Step 3: init/freeze
        logging.info(f"--- [Epoch {epoch+1}] Step 3: Initializing/Freezing Adapters ---")
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            client_model.reins = copy.deepcopy(global_model.reins).to(device) # 필요 시 deepcopy로 교체 가능
            client_model.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
            client_model.reins2.load_state_dict(global_model.reins2.state_dict())
            client_model.linear_rein2.load_state_dict(global_model.linear_rein2.state_dict())
            for name, param in client_model.named_parameters():
                if 'reins.' in name or 'linear_rein.' in name:
                    param.requires_grad = True
                elif 'reins2.' in name or 'linear_rein2.' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = False
        logging.info(f"--- [Epoch {epoch+1}] Step 3 Finished ---")

        # Step 4: train rein (fusion 옵션)
        logging.info(f"--- [Epoch {epoch+1}] Step 4: Training 'rein' adapter ({'fusion' if STEP4_USE_FUSION else 'single'}) ---")

        for client_idx in tqdm(range(args.num_clients), desc="Step4 Clients", leave=False, position=1):
            client_model = client_model_list[client_idx]
            client_model.train()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, client_model.parameters()), lr=args.lr)
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]

            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    if STEP4_USE_FUSION:
                        with torch.amp.autocast('cuda'):
                            feats = client_model.forward_fusion2(inputs)[:, 0, :]
                            # feats = client_model.forward_fusion3(inputs)[:, 0, :]
                            logits = client_model.linear_rein(feats) + 0.5 * class_list
                            ce_losses = F.cross_entropy(logits, targets, reduction='none')
                            loss = ce_losses.mean()
                    else:
                        with torch.amp.autocast('cuda'):
                            feats = client_model.forward_features(inputs)[:, 0, :]
                            logits = client_model.linear_rein(feats) + 0.5 * class_list
                            ce_losses = F.cross_entropy(logits, targets, reduction='none')
                            loss = ce_losses.mean()

                    # --- per-sample loss EMA(벡터화, GPU->CPU 일괄 전송) ---
                    with torch.no_grad():
                        idxs = torch.as_tensor(batch_indices, dtype=torch.long, device='cpu')
                        vals = ce_losses.detach().to(torch.float32).cpu()
                        old = loss_ema[idxs]
                        is_nan = torch.isnan(old)
                        old[is_nan] = vals[is_nan]
                        keep = ~is_nan
                        old[keep] = EMA_BETA * old[keep] + (1.0 - EMA_BETA) * vals[keep]
                        loss_ema[idxs] = old
                        label_buf[idxs] = targets.detach().cpu()

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            if EVAL_PER_CLIENT_STEP4:
                bacc_c, acc_c = calculate_accuracy(client_model, test_loader, device, mode='rein' if not STEP4_USE_FUSION else 'fusion')
                logging.info(f"Client {client_idx} after Step4 - BAcc: {bacc_c*100:.2f}%, Acc: {acc_c*100:.2f}%")
        logging.info(f"--- [Epoch {epoch+1}] Step 4 Finished ---")

        # Step 5: GMM split + unfreeze rein2
        logging.info(f"--- [Epoch {epoch+1}] Step 5: GMM split per class ---")
        # client_clean_masks = [{} for _ in range(args.num_clients)]
        # for client_idx in tqdm(range(args.num_clients), desc="Step5 BuildMasks", leave=False, position=1):
        #     client_idxs = torch.as_tensor(list(dict_users[client_idx]), dtype=torch.long)
        #     vals = loss_ema[client_idxs]
        #     valid = ~torch.isnan(vals)
        #     if valid.any():
        #         sample_indices = client_idxs[valid].tolist()
        #         sample_labels  = label_buf[client_idxs][valid].tolist()
        #         sample_loss_mean = vals[valid].tolist()
        #         client_clean_masks[client_idx] = gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean)
        #     else:
        #         client_clean_masks[client_idx] = {}
        # [ADD-4] 저빈도 갱신: 특정 에폭에만 GMM 수행
        do_update_mask = (epoch % MASK_UPDATE_EVERY == 0)
        if do_update_mask:
            for client_idx in tqdm(range(args.num_clients), desc="Step5 BuildMasks", leave=False, position=1):
                client_idxs = torch.as_tensor(list(dict_users[client_idx]), dtype=torch.long)
                vals = loss_ema[client_idxs]
                valid = ~torch.isnan(vals)
                if valid.any():
                    sample_indices = client_idxs[valid].tolist()
                    sample_labels  = label_buf[client_idxs][valid].tolist()
                    sample_loss_mean = vals[valid].tolist()
                    clean_bool = gmm_split_classwise(sample_indices, sample_labels, sample_loss_mean)  # idx->bool
                    # EMA로 확률 갱신
                    for idx_i, is_clean in clean_bool.items():
                        p_old = client_clean_prob[client_idx][idx_i]  # default 1.0
                        p_new = 1.0 if is_clean else 0.0
                        client_clean_prob[client_idx][idx_i] = MASK_MOMENTUM * p_old + (1.0 - MASK_MOMENTUM) * p_new
                # valid 없으면 기존 확률 유지
        else:
            logging.info("Step5: skip GMM update this epoch (hysteresis)")

        logging.info(f"--- [Epoch {epoch+1}] Step 5: Unfreezing 'rein2' adapter ---")
        for client_idx in range(args.num_clients):
            client_model = client_model_list[client_idx]
            for name, param in client_model.named_parameters():
                if 'reins.' in name or 'linear_rein.' in name:
                    param.requires_grad = False
                elif 'reins2.' in name or 'linear_rein2.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        logging.info(f"--- [Epoch {epoch+1}] Step 5 Finished ---")

        # Step 6: Distill to global (rein2 path) with AMP
        logging.info(f"--- [Epoch {epoch+1}] Step 6: Clean CE and Noisy KL-to-Global ---")
        global_model.eval()
        for client_idx in tqdm(range(args.num_clients), desc="Step6 Clients", leave=False, position=1):
            client_model = client_model_list[client_idx]
            client_model.train()
            optimizer = torch.optim.AdamW(
                list(client_model.reins2.parameters()) + list(client_model.linear_rein2.parameters()),
                lr=args.lr
            )
            loader = clients_train_loader_list[client_idx]
            class_list = clients_train_class_num_list[client_idx]
            # clean_mask_dict = client_clean_masks[client_idx]
            
            # [ADD-3] 글로벌 파라미터 스냅샷(프로시말 항 계산용)
            g_reins2 = {n: p.detach() for n, p in global_model.reins2.named_parameters()}

            for _ in range(args.local_ep):
                for inputs, targets, batch_indices in loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        # Student
                        s_feats = client_model.forward_features2(inputs)[:, 0, :]
                        s_logits = client_model.linear_rein2(s_feats) + 0.5 * class_list
                        # Teacher
                        with torch.no_grad():
                            t_feats = global_model.forward_fusion2(inputs)[:, 0, :]
                            # t_feats = global_model.forward_fusion3(inputs)[:, 0, :]
                            t_logits = global_model.linear_rein2(t_feats) + 0.5 * class_list
                            t_probsT = F.softmax(t_logits / DISTILL_T, dim=1)
                            
                        # [ADD-4] 확률 마스크(히스테리시스)
                        with torch.no_grad():
                            probs = []
                            for j in range(len(batch_indices)):
                                idx_j = int(batch_indices[j])
                                probs.append(client_clean_prob[client_idx][idx_j])  # default 1.0
                            probs = torch.tensor(probs, device=inputs.device, dtype=torch.float32)
                            is_clean = probs >= CLEAN_THRESHOLD

                        # mask
                        # is_clean = []
                        # for j in range(len(batch_indices)):
                        #     idx_j = int(batch_indices[j])
                        #     flag = clean_mask_dict.get(idx_j, True)
                        #     is_clean.append(flag)
                        # is_clean = torch.tensor(is_clean, device=inputs.device, dtype=torch.bool)

                        loss = 0.0
                        if is_clean.any():
                            ce_loss = F.cross_entropy(s_logits[is_clean], targets[is_clean])
                            loss = loss + ce_loss
                        if (~is_clean).any():
                            s_log_probsT = F.log_softmax(s_logits[~is_clean] / DISTILL_T, dim=1)
                            kl = F.kl_div(s_log_probsT, t_probsT[~is_clean], reduction='batchmean') * (DISTILL_T ** 2)
                            loss = loss + KL_WEIGHT * kl
                            if FEAT_MSE_WEIGHT > 0.0:
                                mse = F.mse_loss(s_feats[~is_clean], t_feats[~is_clean])
                                loss = loss + FEAT_MSE_WEIGHT * mse

                        # [ADD-3] FedProx(Proximal) 항 추가: reins2만 대상
                        prox = 0.0
                        for n, p in client_model.reins2.named_parameters():
                            gp = g_reins2[n]
                            prox = prox + (p - gp).pow(2).sum()
                        loss = loss + FEDPROX_MU * prox
                        
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        logging.info(f"--- [Epoch {epoch+1}] Step 6 Finished ---")

        # Step 7: Average back to global
        logging.info(f"--- [Epoch {epoch+1}] Step 7: Averaging 'rein' adapters to global model ---")
        with torch.no_grad():
            reins_named_params = {}
            for name, _ in client_model_list[0].reins2.named_parameters():
                stacked = torch.stack([dict(client.reins2.named_parameters())[name].data for client in client_model_list])
                reins_named_params[name] = stacked.mean(dim=0)
            for name, param in global_model.reins2.named_parameters():
                if name in reins_named_params:
                    param.data.copy_(reins_named_params[name])
            weight_sum = sum(client.linear_rein2.weight.data for client in client_model_list)
            bias_sum = sum(client.linear_rein2.bias.data for client in client_model_list)
            global_model.linear_rein2.weight.data.copy_(weight_sum / len(client_model_list))
            global_model.linear_rein2.bias.data.copy_(bias_sum / len(client_model_list))
        
        bacc, acc = calculate_accuracy(global_model, test_loader, device, mode='rein2')
        logging.info(f"Global Model after Epoch {epoch+1} - BAcc: {bacc*100:.2f}%, Acc: {acc*100:.2f}%")
        logging.info(f"--- [Epoch {epoch+1}] Step 7 Finished ---")

    logging.info("="*50)            
    logging.info("FedDouble training process finished.")
    logging.info("="*50)

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
