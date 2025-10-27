import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, uuid, copy, logging, tempfile, shutil, errno, random
import multiprocessing as mp
mp.set_executable(sys.executable)
mp.set_start_method("spawn", force=True)   # default 'fork' on Linux is fine; don't respawn unnecessarily

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

# project paths
sys.path.append('/home/work/Workspaces/yunjae_heo/FedLNL/')
sys.path.append('/home/work/Workspaces/yunjae_heo/FedLNL/other_repos/FedNoRo/')

from collections import OrderedDict, Counter
from torch.utils.data import DataLoader, Dataset

from utils.utils import add_noise, set_seed
from dataset.dataset import get_dataset
from model_dino import ReinDinov2, ReinDinov2_trans
from dino_variant import _small_variant
from update import train, evaluate, train_forward
from ensemble import compute_var, compute_mean_sq
import dataset.dataset as dataset

import argparse

# ---------- System knobs to reduce CPU thrash ----------
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

# ---------- FedAvg helpers ----------
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = w_avg[key] / float(len(w))
    return w_avg

def average_weights_weighted(w, dataset_list):
    w_avg = copy.deepcopy(w[0])
    sizes = [len(ds) for ds in dataset_list]
    total = float(sum(sizes)) if len(sizes)>0 else 1.0
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * sizes[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * sizes[i]
        w_avg[key] = w_avg[key] / total
    return w_avg

# ---------- Arguments ----------
def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1)
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--gpu', type=str,  default='1')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--noise_rate', type=float, default=1.0)

    # basic setting
    parser.add_argument('--exp', type=str, default='Fed')
    parser.add_argument('--dataset', type=str, default='ich')
    parser.add_argument('--model', type=str, default='ReinDinov2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_lr', type=float,  default=3e-4)
    parser.add_argument('--pretrained', type=int,  default=1)
    parser.add_argument('--n_classes', type=int, default=5)

    # FL
    parser.add_argument('--n_clients', type=int,  default=20)
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--non_iid_prob_class', type=float, default=0.9)
    parser.add_argument('--alpha_dirichlet', type=float, default=2.0)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--round1', type=int,  default=10)  # warm-up
    parser.add_argument('--round2', type=int,  default=10)  # trans matrix
    parser.add_argument('--round3', type=int,  default=40)  # finetune

    # FedBeat-ish knobs
    parser.add_argument('--s1', type=int,  default=10)
    parser.add_argument('--begin', type=int,  default=10)
    parser.add_argument('--end', type=int,  default=49)
    parser.add_argument('--a', type=float,  default=0.8)
    parser.add_argument('--warm', type=int,  default=1)

    # noise
    parser.add_argument('--level_n_system', type=float, default=1.0)
    parser.add_argument('--level_n_lowerb', type=float, default=0.3)
    parser.add_argument('--level_n_upperb', type=float, default=0.5)
    parser.add_argument('--n_type', type=str, default="instance")

    # experiments
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--print_txt', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_f', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--local_ep2', type=int, default=5)
    parser.add_argument('--trans_rounds', type=int, default=50)

    # speed/IO knobs (new)
    parser.add_argument('--teachers_k', type=int, default=-1, help='-1: use all clients as teachers; else cap')
    parser.add_argument('--save_distilled', action='store_true', help='save distilled arrays to disk')
    parser.add_argument('--use_amp', type=int, default=1)

    args = parser.parse_args()
    return args

# ---------- Dataset Wrapper ----------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, clean_labels):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.clean_labels = clean_labels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        clean_label = self.clean_labels[self.idxs[item]]
        return image, label, clean_label, self.idxs[item]

# ---------- Builders ----------
def build_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        worker_init_fn=(worker_init_fn if num_workers > 0 else None),
        drop_last=False
    )

def build_model(n_classes, variant, dino_state_dict, device):
    m = ReinDinov2(variant=variant, dino_state_dict=dino_state_dict, num_classes=n_classes)
    m.eval()
    return m.to(device)

def build_trans_model(n_classes, variant, dino_state_dict, warmup_state_dict, device):
    classifier_trans = ReinDinov2_trans(variant, dino_state_dict, n_classes)
    classifier_trans.eval()
    # stitch weights from warmup classifier where names match
    temp = OrderedDict()
    params = classifier_trans.state_dict()
    for name, parameter in warmup_state_dict.items():
        if name in params:
            temp[name] = parameter
    params.update(temp)
    classifier_trans.load_state_dict(params)
    return classifier_trans.to(device)

# ---------- Teacher weights (pre-generate & reuse) ----------
def make_teacher_state_dicts(base_model_sd, w_avg, w_var, w_norm, k, var_scale=0.1):
    """Return list[OrderedDict] of length k. Only perturb keys present in w_*; others stay as base."""
    teachers = []
    # (선택) 키 정규화 유틸 – 필요시 사용
    # base_model_sd = strip_prefix_if_needed(base_model_sd, want_prefix=None)

    keys_to_perturb = set(w_avg.keys()) & set(base_model_sd.keys())
    with torch.no_grad():
        for _ in range(k):
            # 시작점: 베이스 그대로 복사
            t_sd = OrderedDict((n, p.clone()) for n, p in base_model_sd.items())
            for name in keys_to_perturb:
                mean = w_avg[name]
                var  = torch.clamp(w_var[name], 1e-6, 1e2)
                eps  = torch.randn_like(mean)
                delta = (mean + torch.sqrt(var) * eps * var_scale) * w_norm[name]
                t_sd[name] = delta + base_model_sd[name]
            teachers.append(t_sd)
    return teachers
# ---------- Main ----------
def main(args):
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

    # dataset
    dataset_train, dataset_test, dict_users = get_dataset(args)
    print(f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    print(f"test:  {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # noisy labels
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users, total_dataset=dataset_train)
    original_targets = dataset_train.targets
    dataset_train.targets = y_train_noisy

    # loaders (global & per client) — build once and reuse
    test_loader = build_loader(dataset_test, args.batch_size, shuffle=False, num_workers=args.num_workers)

    clients_train_loader_list = []
    clients_train_loader_batch_list = []
    clients_train_class_num_list = []

    for i in range(args.n_clients):
        ds_split = DatasetSplit(dataset_train, dict_users[i], original_targets)
        train_loader = build_loader(ds_split, args.batch_size, True, args.num_workers)
        batch_eval_loader = build_loader(ds_split, args.batch_size, False, args.num_workers)

        # class prior (for logit adjustment)
        class_num = [0 for _ in range(args.n_classes)]
        for idx in dict_users[i]:
            class_num[int(dataset_train.targets[idx])] += 1
        class_num_tensor = torch.tensor(class_num, device=device, dtype=torch.float32)
        class_p = (class_num_tensor / class_num_tensor.sum()).clamp_min(1e-8).log().view(1, -1)

        clients_train_loader_list.append(train_loader)
        clients_train_loader_batch_list.append(batch_eval_loader)
        clients_train_class_num_list.append(class_p)

    # model & weights
    print('constructing model...')
    variant = _small_variant
    dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth', weights_only=False)
    classifier = build_model(args.n_classes, variant, dino_state_dict, device)

    # -------------------- Warm Up --------------------
    print('----------Starting Warm Up Classifier Model----------')
    best_acc = 0.0; best_round = 0; best_model_weights_list = []
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    # cache global init state_dict once
    global_sd = copy.deepcopy(classifier.state_dict())

    for rd in range(args.round1):
        local_weights_list, local_acc_list = [], []
        selected_id = list(range(args.n_clients))  # 동일 비교 위해 전체 사용(샘플링 가능)
        random.shuffle(selected_id)

        for client_id in selected_id:
            # fresh local model from cached global_sd
            model_local = build_model(args.n_classes, variant, dino_state_dict, device)
            model_local.load_state_dict(global_sd, strict=True)
            model_local.train()

            optimizer_w = torch.optim.AdamW(model_local.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # local epochs
            for epoch in range(args.local_ep):
                # 'train' 함수 그대로 사용 (내부 미변)
                _ = train(
                    clients_train_loader_list[client_id],
                    epoch, model_local, optimizer_w, criterion, args,
                    clients_train_class_num_list[client_id]
                )

            local_weights_list.append(copy.deepcopy(model_local.state_dict()))

        # average
        classifier.load_state_dict(average_weights(local_weights_list))
        global_sd = copy.deepcopy(classifier.state_dict())  # update cache

        # eval (inference/amp)
        with torch.inference_mode(), torch.amp.autocast('cuda',enabled=use_amp):
            val_acc = evaluate(test_loader, classifier)
        print(f'Warm up Round [{rd+1}] Val Acc: {val_acc:.4f} %')

        if val_acc > best_acc:
            best_acc = val_acc; best_round = rd + 1
            best_model_weights_list = copy.deepcopy(local_weights_list)
            torch.save(classifier.state_dict(), os.path.join(model_dir, 'warmup_model.pth'))
    print(f'Best WarmUp Round [{best_round}]')
    print('----------Finishing Warm Up Classifier Model----------')

    # -------------------- Distill --------------------
    print('----------Start Distilling----------')
    warmup_sd = torch.load(os.path.join(model_dir, 'warmup_model.pth'), map_location='cpu')
    classifier.load_state_dict(warmup_sd, strict=True)
    classifier.to(device)

    base_model = copy.deepcopy(classifier.state_dict())
    w_avg, w_sq_avg, w_norm = compute_mean_sq(best_model_weights_list, base_model)
    w_var = compute_var(w_avg, w_sq_avg)

    with torch.inference_mode(), torch.amp.autocast('cuda',enabled=use_amp):
        test_acc = evaluate(test_loader, classifier)
    print(f'Test Accuracy after WarmUp: {test_acc:.4f} %')

    distilled_dataset_clients_list = []
    distilled_loader_clients_list = []

    # teachers count
    TEACHERS_K = args.n_clients if args.teachers_k == -1 else max(1, args.teachers_k)
    # prepare global teacher SDs once per whole distill phase (deterministic under seed)
    teachers_sd_global = make_teacher_state_dicts(base_model, w_avg, w_var, w_norm, TEACHERS_K, var_scale=0.1)

    for client_id in range(args.n_clients):
        distilled_idx_list = []
        distilled_pseudo_list = []

        # evaluate batches with ensemble teachers
        for data, noisy_label, clean_label, indexes in clients_train_loader_batch_list[client_id]:
            data = data.to(device, non_blocking=True)

            logits_sum = None
            for t_sd in teachers_sd_global:
                classifier.load_state_dict(t_sd, strict=False)
                classifier.eval()
                with torch.inference_mode(), torch.amp.autocast('cuda',enabled=use_amp):
                    out = classifier(data)  # (B, C)
                    probs = out.softmax(dim=1)
                logits_sum = probs if logits_sum is None else (logits_sum + probs)

            logits_avg = logits_sum / float(len(teachers_sd_global))
            conf, pseudo = torch.max(logits_avg, dim=1)
            mask = conf > args.tau

            distilled_idx_list.extend(indexes[mask.cpu()].tolist())
            distilled_pseudo_list.extend(pseudo[mask].cpu().tolist())

        distilled_example_index = np.array(distilled_idx_list, dtype=np.int64)
        distilled_pseudo_labels = np.array(distilled_pseudo_list, dtype=np.int64)

        # assemble arrays in-memory (dataset may not expose raw tensors; keep current safe path)
        if len(distilled_pseudo_labels) > 0:
            # caution: dataset_train[i][0] returns Tensor(C,H,W); convert to uint8 image-like
            src_imgs = []
            for i in distilled_example_index:
                img_tensor = dataset_train[i][0]  # Tensor
                img = (img_tensor.clamp(0,1).mul(255).byte().cpu().numpy()
                       if img_tensor.dtype.is_floating_point
                       else img_tensor.cpu().numpy())
                src_imgs.append(img)
            distilled_images = np.stack(src_imgs, axis=0)
            distilled_noisy_labels = np.array(dataset_train.targets)[distilled_example_index]
            distilled_clean_labels = np.array(original_targets)[distilled_example_index]
            distilled_acc = (distilled_pseudo_labels == distilled_clean_labels).mean() if len(distilled_clean_labels)>0 else 0.0
            print(f"[Client {client_id}] distilled={len(distilled_pseudo_labels)} acc={distilled_acc:.4f}")
        else:
            distilled_images = np.empty((0,), dtype=np.uint8)
            distilled_noisy_labels = np.empty((0,), dtype=np.int64)
            distilled_clean_labels = np.empty((0,), dtype=np.int64)
            print(f"[Client {client_id}] distilled=0")

        # optional save
        if args.save_distilled:
            np.save(os.path.join(model_dir, f'{client_id}_distilled_images.npy'), distilled_images)
            np.save(os.path.join(model_dir, f'{client_id}_distilled_pseudo_labels.npy'), distilled_pseudo_labels)
            np.save(os.path.join(model_dir, f'{client_id}_distilled_noisy_labels.npy'), distilled_noisy_labels)
            np.save(os.path.join(model_dir, f'{client_id}_distilled_clean_labels.npy'), distilled_clean_labels)

        # build distilled dataset directly from memory (no reload)
        distilled_dataset_ = dataset.distilled_dataset(
            distilled_images,
            distilled_noisy_labels,
            distilled_pseudo_labels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        )
        distilled_dataset_clients_list.append(distilled_dataset_)
        if len(distilled_dataset_) > 0:
            train_loader_distilled = build_loader(distilled_dataset_, args.batch_size, True, args.num_workers)
        else:
            train_loader_distilled = []  # keep sentinel as in original
        distilled_loader_clients_list.append(train_loader_distilled)

    print('----------Finishing Distilling----------')

    # -------------------- Train Transition Matrix --------------------
    print('----------Starting Training Trans Matrix Estimation Model----------')
    classifier.load_state_dict(warmup_sd, strict=True)
    classifier.to(device)
    classifier_trans = build_trans_model(args.n_classes, variant, dino_state_dict, warmup_sd, device)
    loss_function = nn.NLLLoss()

    # vectorized step uses bmm; keep same schedule
    lr = args.lr
    for rd in range(args.round2):
        lr = lr * 0.99
        local_weights_list = []
        for client_id in range(args.n_clients):
            client_loader = distilled_loader_clients_list[client_id]
            if client_loader == [] or len(client_loader) == 0:
                # nothing to train on this client
                continue

            model_local_trans = ReinDinov2_trans(variant, dino_state_dict, args.n_classes).to(device)
            # stitch warmup
            model_local_trans.load_state_dict(classifier_trans.state_dict(), strict=True)
            model_local_trans.train()
            optimizer_trans = torch.optim.AdamW(model_local_trans.parameters(), lr=lr)

            for epoch in range(args.local_ep):
                epoch_loss = 0.0
                for data, noisy_labels, pseudo_labels, index in client_loader:
                    data = data.to(device, non_blocking=True)
                    noisy_labels = noisy_labels.to(device, non_blocking=True)
                    pseudo_labels = pseudo_labels.to(device, non_blocking=True)

                    with torch.amp.autocast('cuda',enabled=use_amp):
                        # (B, C, C)
                        batch_matrix = model_local_trans(data)
                        # (B,C) one-hot
                        one_hot = F.one_hot(pseudo_labels, num_classes=args.n_classes).float()
                        # (B,1,C) @ (B,C,C) -> (B,1,C) -> (B,C)
                        noisy_class_post = torch.bmm(one_hot.unsqueeze(1), batch_matrix).squeeze(1)
                        loss = loss_function((noisy_class_post + 1e-12).log(), noisy_labels)

                    optimizer_trans.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer_trans); scaler.update()
                    else:
                        loss.backward(); optimizer_trans.step()
                    epoch_loss += float(loss.detach().cpu())

                # minimal logging
                print(f"[Trans RD {rd+1}/{args.round2}] Epoch {epoch+1}/{args.local_ep} Loss {epoch_loss/ max(1,len(client_loader)):.4f}")

            local_weights_list.append(copy.deepcopy(model_local_trans.state_dict()))

        if local_weights_list:
            classifier_trans.load_state_dict(average_weights_weighted(local_weights_list, distilled_dataset_clients_list))
    torch.save(classifier_trans.state_dict(), os.path.join(model_dir, 'trans_model.pth'))
    print('----------Finishing Training Trans Matrix Estimation Model----------')

    # -------------------- Finetuning --------------------
    print('----------Starting Finetuning Classifier Model----------')
    classifier.load_state_dict(warmup_sd, strict=True)
    classifier_trans.load_state_dict(torch.load(os.path.join(model_dir, 'trans_model.pth'), map_location='cpu'), strict=True)
    classifier.to(device); classifier_trans.to(device)

    with torch.inference_mode(), torch.amp.autocast('cuda',enabled=use_amp):
        base_test = evaluate(test_loader, classifier)
    print(f'WarmUp model Test Acc: {base_test:.4f} %')

    best_acc = 0.0; best_round = 0; test_acc_list = []
    for rd in range(args.round3):
        local_weights_list = []
        selected_ids = list(range(args.n_clients))
        random.shuffle(selected_ids)

        for client_id in selected_ids:
            model_local = ReinDinov2(variant=variant, dino_state_dict=dino_state_dict, num_classes=args.n_classes).to(device)
            model_local.load_state_dict(classifier.state_dict(), strict=True)
            model_local.train()

            optimizer_f = torch.optim.Adam(model_local.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)

            for epoch in range(args.local_ep2):
                classifier_trans.eval()
                # train_forward 내부 로직 유지
                train_forward(model_local, clients_train_loader_list[client_id], optimizer_f, classifier_trans)
                with torch.inference_mode(), torch.amp.autocast('cuda',enabled=use_amp):
                    cur_acc = evaluate(test_loader, model_local)
                print(f'[Fine RD {rd+1}/{args.round3}] Ep {epoch+1}/{args.local_ep2} Cl {client_id+1} Test {cur_acc:.4f}%')

            local_weights_list.append(copy.deepcopy(model_local.state_dict()))

        classifier.load_state_dict(average_weights(local_weights_list))
        with torch.inference_mode(), torch.amp.autocast('cuda',enabled=use_amp):
            test_acc = evaluate(test_loader, classifier)
        test_acc_list.append(test_acc)
        print(f'Round [{rd+1}/{args.round3}] Test Acc: {test_acc:.4f} %')

        if test_acc > best_acc:
            best_acc = test_acc; best_round = rd + 1
            torch.save(classifier.state_dict(), os.path.join(model_dir, 'final_model.pth'))

    print(f'Best Round [{best_round}]')
    if len(test_acc_list) > 0:
        print('Test Acc Max:', max(test_acc_list))

    return (max(test_acc_list) if test_acc_list else best_acc)

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    acc_list = []
    for index_exp in range(args.num_exp):
        print(f"--- Running Experiment {index_exp+1}/{args.num_exp} ---")
        args.seed = index_exp + 1

        # optional file logging
        if args.print_txt:
            log_path = os.path.join(args.result_dir, args.dataset, f'log_{args.noise_rate}_{args.tau}_{(args.level_n_upperb+args.level_n_lowerb)/2}.txt')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            sys.stdout = open(log_path, 'a')

        print(f"Arguments: {args}")
        acc = main(args)
        acc_list.append(acc)

        if args.print_txt:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    print(f"All Accuracies: {acc_list}")
    if len(acc_list) > 0:
        arr = np.array(acc_list, dtype=np.float32)
        print(f"Average Accuracy: {arr.mean():.4f}")
        if len(arr) > 1:
            print(f"Std (ddof=1): {arr.std(ddof=1):.4f}")
