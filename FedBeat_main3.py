import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno, random, glob, json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader, Dataset

# project paths
sys.path.append('/home/work/Workspaces/yunjae_heo/FedLNL/')
sys.path.append('/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo/')

from utils.utils import add_noise, set_seed
from dataset.dataset import get_dataset
from model_dino import ReinDinov2, ReinDinov2_trans
from dino_variant import _small_variant
from update import train, evaluate, train_forward
from ensemble import compute_var, compute_mean_sq
import dataset.dataset as dataset

# ---------- System knobs ----------
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

def shutdown_loader(loader):
    it = getattr(loader, "_iterator", None)
    if it is not None:
        it._shutdown_workers()
    del loader

# ---------- FedAvg ----------
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = w_avg[k] / float(len(w))
    return w_avg

def average_weights_weighted(w, dataset_list):
    w_avg = copy.deepcopy(w[0])
    sizes = [len(ds) for ds in dataset_list]
    total = float(sum(sizes)) if len(sizes)>0 else 1.0
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * sizes[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * sizes[i]
        w_avg[k] = w_avg[k] / total
    return w_avg

# ---------- Args ----------
def args_parser():
    p = argparse.ArgumentParser()
    # system
    p.add_argument('--deterministic', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=str, default='1')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--result_dir', type=str, default='./results/')
    p.add_argument('--noise_rate', type=float, default=1.0)
    # basic
    p.add_argument('--exp', type=str, default='Fed')
    p.add_argument('--dataset', type=str, default='isic2019')
    p.add_argument('--model', type=str, default='ReinDinov2')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--base_lr', type=float, default=3e-4)
    p.add_argument('--pretrained', type=int, default=1)
    p.add_argument('--n_classes', type=int, default=8)
    # FL
    p.add_argument('--n_clients', type=int, default=20)
    p.add_argument('--iid', type=int, default=0)
    p.add_argument('--non_iid_prob_class', type=float, default=0.9)
    p.add_argument('--alpha_dirichlet', type=float, default=2.0)
    p.add_argument('--local_ep', type=int, default=5)
    p.add_argument('--round1', type=int, default=10)
    p.add_argument('--round2', type=int, default=10)
    p.add_argument('--round3', type=int, default=40)
    # FedBeat-ish
    p.add_argument('--s1', type=int, default=10)
    p.add_argument('--begin', type=int, default=10)
    p.add_argument('--end', type=int, default=49)
    p.add_argument('--a', type=float, default=0.8)
    p.add_argument('--warm', type=int, default=1)
    # noise
    p.add_argument('--level_n_system', type=float, default=1.0)
    p.add_argument('--level_n_lowerb', type=float, default=0.3)
    p.add_argument('--level_n_upperb', type=float, default=0.5)
    p.add_argument('--n_type', type=str, default="instance")
    # exp
    p.add_argument('--num_exp', type=int, default=1)
    p.add_argument('--log_to_file', default=True)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr_f', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--tau', type=float, default=0.4)
    p.add_argument('--local_ep2', type=int, default=5)
    p.add_argument('--trans_rounds', type=int, default=50)
    # speed/IO
    p.add_argument('--teachers_k', type=int, default=-1)
    p.add_argument('--save_distilled', action='store_true')
    p.add_argument('--use_amp', type=int, default=1)
    # resume
    p.add_argument('--resume', action='store_true')
    return p.parse_args()

# ---------- DatasetSplit ----------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, clean_labels):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.clean_labels = clean_labels
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        image, label = self.dataset[self.idxs[i]]
        clean_label = self.clean_labels[self.idxs[i]]
        return image, label, clean_label, self.idxs[i]

# ---------- Builders ----------
def build_loader(ds, batch, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
        drop_last=False
    )

def build_model(n_classes, variant, dino_state_dict, device):
    m = ReinDinov2(variant=variant, dino_state_dict=dino_state_dict, num_classes=n_classes)
    m.eval()
    return m.to(device)

def build_trans_model(n_classes, variant, dino_state_dict, warmup_state_dict, device):
    classifier_trans = ReinDinov2_trans(variant, dino_state_dict, n_classes)
    classifier_trans.eval()
    temp = OrderedDict()
    params = classifier_trans.state_dict()
    for name, parameter in warmup_state_dict.items():
        if name in params:
            temp[name] = parameter
    params.update(temp)
    classifier_trans.load_state_dict(params)
    return classifier_trans.to(device)

# ---------- Teacher SDs ----------
def make_teacher_state_dicts(base_model_sd, w_avg, w_var, w_norm, k, var_scale=0.1):
    teachers = []
    keys_to_perturb = set(w_avg.keys()) & set(base_model_sd.keys())
    with torch.no_grad():
        for _ in range(k):
            t_sd = OrderedDict((n, p.clone()) for n, p in base_model_sd.items())
            for name in keys_to_perturb:
                mean = w_avg[name]
                var  = torch.clamp(w_var[name], 1e-6, 1e2)
                eps  = torch.randn_like(mean)
                delta = (mean + torch.sqrt(var) * eps * var_scale) * w_norm[name]
                t_sd[name] = delta + base_model_sd[name]
            teachers.append(t_sd)
    return teachers

# ---------- Checkpoint utils (NEW) ----------
def _ckpt_id(args):
    return f"{args.dataset}_nl{args.level_n_lowerb}_nu{args.level_n_upperb}_nr{args.noise_rate}_tau{args.tau}_seed{args.seed}"

def _ckpt_dir(base_dir, args):
    d = os.path.join(base_dir, "ckpt", _ckpt_id(args))
    os.makedirs(d, exist_ok=True)
    return d

def save_ckpt(path, payload:dict):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

def latest_round(path_pattern):
    files = glob.glob(path_pattern)
    if not files: return None, None
    # expect ..._rd{num}.pth
    best = None; max_rd = -1
    for f in files:
        try:
            rd = int(os.path.splitext(f)[0].split("_rd")[-1])
            if rd > max_rd: max_rd, best = rd, f
        except Exception:
            continue
    return best, max_rd

# ---------- Main ----------
def main(args):
    # logging
    if args.log_to_file:
        log_path = os.path.join(args.result_dir, args.dataset,
                                f'log_{args.noise_rate}_{args.tau}_{(args.level_n_upperb+args.level_n_lowerb)/2:.3f}.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("fed")

    # seed & device
    if args.deterministic:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = False; cudnn.deterministic = True
    else:
        cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.use_amp)

    # dirs
    save_dir = os.path.join(args.result_dir, args.dataset, f'iid_{args.iid}')
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, f'{args.seed}_rate_{args.noise_rate}_threshold_{args.tau}')
    os.makedirs(model_dir, exist_ok=True)
    ckpt_dir = _ckpt_dir(model_dir, args)

    # dataset
    dataset_train, dataset_test, dict_users = get_dataset(args)
    log.info(f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    log.info(f"test:  {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # noisy labels
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    original_targets = dataset_train.targets
    dataset_train.targets = y_train_noisy

    # test loader
    test_loader = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=True,
        persistent_workers=True, prefetch_factor=2, worker_init_fn=worker_init_fn
    )

    # model
    log.info('constructing model...')
    variant = _small_variant
    dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False)
    classifier = build_model(args.n_classes, variant, dino_state_dict, device)

    # -------------------- Warm Up --------------------
    log.info('----------Starting Warm Up Classifier Model----------')
    best_acc = 0.0; best_round = 0; best_model_weights_list = []
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    global_sd = copy.deepcopy(classifier.state_dict())

    # resume warmup
    warm_done_path = os.path.join(ckpt_dir, "warmup_done.pth")
    if args.resume and os.path.exists(warm_done_path):
        ckpt = torch.load(warm_done_path, map_location='cpu')
        classifier.load_state_dict(ckpt["classifier"])
        best_acc = ckpt["best_acc"]; best_round = ckpt["best_round"]
        best_model_weights_list = ckpt["best_model_weights_list"]
        torch.save(classifier.state_dict(), os.path.join(model_dir, 'warmup_model.pth'))
        log.info(f"[RESUME] WarmUp already done at round {best_round}.")
    else:
        # maybe resume mid-warmup
        latest_path, latest_rd = latest_round(os.path.join(ckpt_dir, "warmup_rd*.pth"))
        start_rd = (latest_rd + 1) if (args.resume and latest_rd is not None) else 0
        if latest_rd is not None and args.resume:
            ckpt = torch.load(latest_path, map_location='cpu')
            classifier.load_state_dict(ckpt["classifier"])
            global_sd = copy.deepcopy(classifier.state_dict())
            best_acc = ckpt["best_acc"]; best_round = ckpt["best_round"]
            best_model_weights_list = ckpt["best_model_weights_list"]
            log.info(f"[RESUME] WarmUp from round {start_rd}/{args.round1}")

        for rd in range(start_rd, args.round1):
            local_weights_list = []
            selected_id = list(range(args.n_clients))
            random.shuffle(selected_id)

            for client_id in selected_id:
                ds_split = DatasetSplit(dataset_train, dict_users[client_id], original_targets)
                train_loader = build_loader(ds_split, args.batch_size, True, args.num_workers)

                class_num = [0 for _ in range(args.n_classes)]
                for idx in dict_users[client_id]:
                    class_num[int(dataset_train.targets[idx])] += 1
                class_p = torch.tensor(class_num, device=device, dtype=torch.float32)
                class_p = (class_p / class_p.sum()).clamp_min(1e-8).log().view(1, -1)

                model_local = build_model(args.n_classes, variant, dino_state_dict, device)
                model_local.load_state_dict(global_sd, strict=True)
                model_local.train()
                optimizer_w = torch.optim.AdamW(model_local.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                for epoch in range(args.local_ep):
                    _ = train(train_loader, epoch, model_local, optimizer_w, criterion, args, class_p)

                local_weights_list.append(copy.deepcopy(model_local.state_dict()))
                shutdown_loader(train_loader)

            classifier.load_state_dict(average_weights(local_weights_list))
            global_sd = copy.deepcopy(classifier.state_dict())
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_amp):
                val_acc = evaluate(test_loader, classifier)
            log.info(f'Warm up Round [{rd+1}] Val Acc: {val_acc:.4f} %')

            if val_acc > best_acc:
                best_acc = val_acc; best_round = rd + 1
                best_model_weights_list = copy.deepcopy(local_weights_list)
                torch.save(classifier.state_dict(), os.path.join(model_dir, 'warmup_model.pth'))

            # save warmup round ckpt
            save_ckpt(os.path.join(ckpt_dir, f"warmup_rd{rd+1}.pth"), {
                "classifier": classifier.state_dict(),
                "best_acc": best_acc,
                "best_round": best_round,
                "best_model_weights_list": best_model_weights_list
            })

        # mark done
        save_ckpt(warm_done_path, {
            "classifier": classifier.state_dict(),
            "best_acc": best_acc,
            "best_round": best_round,
            "best_model_weights_list": best_model_weights_list
            })
    log.info(f'Best WarmUp Round [{best_round}]')
    log.info('----------Finishing Warm Up Classifier Model----------')

    # -------------------- Distill --------------------
    log.info('----------Start Distilling----------')
    # if resuming and distill already done, skip
    distill_flag = os.path.join(ckpt_dir, "distill_done.flag")
    warmup_sd = torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location='cpu')
    # classifier.load_state_dict(warmup_sd, strict=True)
    # classifier.to(device)

    base_model = copy.deepcopy(classifier.state_dict())
    if not (args.resume and os.path.exists(distill_flag)):
        # recompute teacher stats
        if not best_model_weights_list:
            log.info("[WARN] best_model_weights_list empty; using current state only for teachers.")
            best_model_weights_list = [copy.deepcopy(base_model)]
        w_avg, w_sq_avg, w_norm = compute_mean_sq(best_model_weights_list, base_model)
        w_var = compute_var(w_avg, w_sq_avg)

        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_amp):
            test_acc = evaluate(test_loader, classifier)
        log.info(f'Test Accuracy after WarmUp: {test_acc:.4f} %')

        distilled_dataset_clients_list = []
        TEACHERS_K = args.n_clients if args.teachers_k == -1 else max(1, args.teachers_k)
        teachers_sd_global = make_teacher_state_dicts(base_model, w_avg, w_var, w_norm, TEACHERS_K, var_scale=0.1)

        for client_id in range(args.n_clients):
            ds_split = DatasetSplit(dataset_train, dict_users[client_id], original_targets)
            eval_loader = build_loader(ds_split, args.batch_size, False, min(args.num_workers, 2))

            distilled_idx_list = []
            distilled_pseudo_list = []

            for data, noisy_label, clean_label, indexes in eval_loader:
                data = data.to(device, non_blocking=True)
                logits_sum = None
                for t_sd in teachers_sd_global:
                    classifier.load_state_dict(t_sd, strict=False)
                    classifier.eval()
                    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_amp):
                        probs = classifier(data).softmax(dim=1)
                    logits_sum = probs if logits_sum is None else (logits_sum + probs)

                logits_avg = logits_sum / float(len(teachers_sd_global))
                conf, pseudo = torch.max(logits_avg, dim=1)
                mask = conf > args.tau

                distilled_idx_list.extend(indexes[mask.cpu()].tolist())
                distilled_pseudo_list.extend(pseudo[mask].cpu().tolist())

            shutdown_loader(eval_loader)

            distilled_example_index = np.array(distilled_idx_list, dtype=np.int64)
            distilled_pseudo_labels = np.array(distilled_pseudo_list, dtype=np.int64)

            if len(distilled_pseudo_labels) > 0:
                src_imgs = []
                for i in distilled_example_index:
                    img_tensor = dataset_train[i][0]
                    img = (img_tensor.clamp(0,1).mul(255).byte().cpu().numpy()
                           if img_tensor.dtype.is_floating_point
                           else img_tensor.cpu().numpy())
                    src_imgs.append(img)
                distilled_images = np.stack(src_imgs, axis=0)
                distilled_noisy_labels = np.array(dataset_train.targets)[distilled_example_index]
                distilled_clean_labels = np.array(original_targets)[distilled_example_index]
                distilled_acc = (distilled_pseudo_labels == distilled_clean_labels).mean() if len(distilled_clean_labels)>0 else 0.0
                log.info(f"[Client {client_id}] distilled={len(distilled_pseudo_labels)} acc={distilled_acc:.4f}")
            else:
                distilled_images = np.empty((0,), dtype=np.uint8)
                distilled_noisy_labels = np.empty((0,), dtype=np.int64)
                distilled_clean_labels = np.empty((0,), dtype=np.int64)
                log.info(f"[Client {client_id}] distilled=0")

            # always store distilled arrays for resume of Trans
            np.save(os.path.join(ckpt_dir, f'{client_id}_distilled_images.npy'), distilled_images)
            np.save(os.path.join(ckpt_dir, f'{client_id}_distilled_pseudo_labels.npy'), distilled_pseudo_labels)
            np.save(os.path.join(ckpt_dir, f'{client_id}_distilled_noisy_labels.npy'), distilled_noisy_labels)
            np.save(os.path.join(ckpt_dir, f'{client_id}_distilled_clean_labels.npy'), distilled_clean_labels)

        # flag
        with open(distill_flag, "w") as f:
            f.write("ok")
    else:
        log.info("[RESUME] Distill already done. Loading distilled arrays from ckpt dir.")

    # build distilled datasets from disk (resume-friendly)
    distilled_dataset_clients_list = []
    for client_id in range(args.n_clients):
        imgs = np.load(os.path.join(ckpt_dir, f'{client_id}_distilled_images.npy'), allow_pickle=True)
        noisy = np.load(os.path.join(ckpt_dir, f'{client_id}_distilled_noisy_labels.npy'), allow_pickle=True)
        pseudo = np.load(os.path.join(ckpt_dir, f'{client_id}_distilled_pseudo_labels.npy'), allow_pickle=True)
        distilled_dataset_ = dataset.distilled_dataset(
            imgs, noisy, pseudo,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        )
        distilled_dataset_clients_list.append(distilled_dataset_)
    log.info('----------Finishing Distilling----------')

    # -------------------- Train Transition Matrix --------------------
    log.info('----------Starting Training Trans Matrix Estimation Model----------')
    classifier.load_state_dict(warmup_sd, strict=True)
    classifier.to(device)
    classifier_trans = build_trans_model(args.n_classes, variant, dino_state_dict, warmup_sd, device)
    loss_function = nn.NLLLoss()
    eps = 1e-6

    # resume trans
    trans_done_path = os.path.join(ckpt_dir, "trans_done.pth")
    if args.resume and os.path.exists(trans_done_path):
        ckpt = torch.load(trans_done_path, map_location='cpu')
        classifier_trans.load_state_dict(ckpt["classifier_trans"])
        torch.save(classifier_trans.state_dict(), os.path.join(model_dir, 'trans_model.pth'))
        log.info("[RESUME] Trans already done.")
    else:
        latest_path, latest_rd = latest_round(os.path.join(ckpt_dir, "trans_rd*.pth"))
        start_rd = (latest_rd + 1) if (args.resume and latest_rd is not None) else 0
        if latest_rd is not None and args.resume:
            ckpt = torch.load(latest_path, map_location='cpu')
            classifier_trans.load_state_dict(ckpt["classifier_trans"])
            log.info(f"[RESUME] Trans from round {start_rd}/{args.round2}")

        for rd in range(start_rd, args.round2):
            local_weights_list = []
            for client_id in range(args.n_clients):
                distilled_ds = distilled_dataset_clients_list[client_id]
                if len(distilled_ds) == 0:
                    continue

                client_loader = build_loader(distilled_ds, args.batch_size, True, min(args.num_workers, 2))
                model_local_trans = ReinDinov2_trans(variant, dino_state_dict, args.n_classes).to(device)
                model_local_trans.load_state_dict(classifier_trans.state_dict(), strict=True)
                model_local_trans.train()
                optimizer_trans = torch.optim.AdamW(model_local_trans.parameters(), lr=args.lr)

                for epoch in range(args.local_ep):
                    epoch_loss = 0.0
                    for data, noisy_labels, pseudo_labels, index in client_loader:
                        data = data.to(device, non_blocking=True)
                        noisy_labels = noisy_labels.to(device, non_blocking=True).long()
                        pseudo_labels = pseudo_labels.to(device, non_blocking=True).long()

                        with torch.amp.autocast('cuda', enabled=use_amp):
                            logits_T = model_local_trans(data)  # (B,C,C)

                        T = F.softmax(logits_T.float(), dim=-1)
                        T = T.clamp_min(eps)
                        T = T / T.sum(dim=-1, keepdim=True)

                        one_hot = F.one_hot(pseudo_labels, num_classes=args.n_classes).float()
                        noisy_class_post = torch.bmm(one_hot.unsqueeze(1), T).squeeze(1)

                        log_probs = (noisy_class_post.clamp_min(eps)).log()
                        loss = loss_function(log_probs, noisy_labels)

                        optimizer_trans.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_local_trans.parameters(), 1.0)
                        optimizer_trans.step()

                        epoch_loss += float(loss.detach().cpu())

                    print(f"[Trans RD {rd+1}/{args.round2}] Epoch {epoch+1}/{args.local_ep} "
                          f"Loss {epoch_loss / max(1, len(client_loader)):.4f}")

                local_weights_list.append(copy.deepcopy(model_local_trans.state_dict()))
                shutdown_loader(client_loader)

            if local_weights_list:
                classifier_trans.load_state_dict(average_weights_weighted(local_weights_list, distilled_dataset_clients_list))

            # save round ckpt
            save_ckpt(os.path.join(ckpt_dir, f"trans_rd{rd+1}.pth"), {
                "classifier_trans": classifier_trans.state_dict()
            })
        # save done + export for finetune
        save_ckpt(trans_done_path, {"classifier_trans": classifier_trans.state_dict()})
        torch.save(classifier_trans.state_dict(), os.path.join(model_dir, 'trans_model.pth'))

    # -------------------- Finetuning --------------------
    log.info('----------Starting Finetuning Classifier Model----------')
    classifier.load_state_dict(warmup_sd, strict=True)
    classifier_trans.load_state_dict(torch.load(os.path.join(model_dir, 'trans_model.pth'), map_location='cpu'), strict=True)
    classifier.to(device); classifier_trans.to(device)

    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_amp):
        base_test = evaluate(test_loader, classifier)
    log.info(f'WarmUp model Test Acc: {base_test:.4f} %')

    best_acc = 0.0; best_round = 0; test_acc_list = []

    # resume finetune
    fin_done_path = os.path.join(ckpt_dir, "finetune_done.pth")
    if args.resume and os.path.exists(fin_done_path):
        ckpt = torch.load(fin_done_path, map_location='cpu')
        classifier.load_state_dict(ckpt["classifier"])
        log.info("[RESUME] Finetune already done.")
    else:
        latest_path, latest_rd = latest_round(os.path.join(ckpt_dir, "finetune_rd*.pth"))
        start_rd = (latest_rd + 1) if (args.resume and latest_rd is not None) else 0
        if latest_rd is not None and args.resume:
            ckpt = torch.load(latest_path, map_location='cpu')
            classifier.load_state_dict(ckpt["classifier"])
            best_acc = ckpt.get("best_acc", 0.0)
            best_round = ckpt.get("best_round", 0)
            test_acc_list = ckpt.get("test_acc_list", [])
            log.info(f"[RESUME] Finetune from round {start_rd}/{args.round3}")

        for rd in range(start_rd, args.round3):
            local_weights_list = []
            selected_ids = list(range(args.n_clients))
            random.shuffle(selected_ids)

            for client_id in selected_ids:
                ds_split = DatasetSplit(dataset_train, dict_users[client_id], original_targets)
                train_loader = build_loader(ds_split, args.batch_size, True, args.num_workers)

                model_local = ReinDinov2(variant=variant, dino_state_dict=dino_state_dict, num_classes=args.n_classes).to(device)
                model_local.load_state_dict(classifier.state_dict(), strict=True)
                model_local.train()
                optimizer_f = torch.optim.Adam(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)

                for epoch in range(args.local_ep2):
                    classifier_trans.eval()
                    train_forward(model_local, train_loader, optimizer_f, classifier_trans)
                    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_amp):
                        cur_acc = evaluate(test_loader, model_local)
                    log.info(f'[Fine RD {rd+1}/{args.round3}] Ep {epoch+1}/{args.local_ep2} Cl {client_id+1} Test {cur_acc:.4f}%')

                local_weights_list.append(copy.deepcopy(model_local.state_dict()))
                shutdown_loader(train_loader)

            classifier.load_state_dict(average_weights(local_weights_list))
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=use_amp):
                test_acc = evaluate(test_loader, classifier)
            test_acc_list.append(test_acc)
            log.info(f'Round [{rd+1}/{args.round3}] Test Acc: {test_acc:.4f} %')

            if test_acc > best_acc:
                best_acc = test_acc; best_round = rd + 1
                torch.save(classifier.state_dict(), os.path.join(model_dir, 'final_model.pth'))

            # save finetune round ckpt
            save_ckpt(os.path.join(ckpt_dir, f"finetune_rd{rd+1}.pth"), {
                "classifier": classifier.state_dict(),
                "best_acc": best_acc,
                "best_round": best_round,
                "test_acc_list": test_acc_list
            })

        save_ckpt(fin_done_path, {"classifier": classifier.state_dict()})

    log.info(f'Best Round [{best_round}]')
    if len(test_acc_list) > 0:
        log.info('Test Acc Max: %.4f', max(test_acc_list))

    return (max(test_acc_list) if test_acc_list else best_acc)

# ---------- Entrypoint ----------
if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import multiprocessing as mp
    try:
        mp.set_executable(sys.executable)
        # mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    acc_list = []
    for index_exp in range(args.num_exp):
        print(f"--- Running Experiment {index_exp+1}/{args.num_exp} ---")
        args.seed = index_exp + 1
        if args.deterministic:
            set_seed(args.seed)
        acc = main(args)
        acc_list.append(acc)

    print(f"All Accuracies: {acc_list}")
    if len(acc_list) > 0:
        arr = np.array(acc_list, dtype=np.float32)
        print(f"Average Accuracy: {arr.mean():.4f}")
        if len(arr) > 1:
            print(f"Std (ddof=1): {arr.std(ddof=1):.4f}")
