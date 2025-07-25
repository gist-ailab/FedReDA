import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import copy
import random
from collections import Counter

from util.options import args_parser
from util.util import *
from util import dataset
from util.util import get_prob, load_data, wrap_as_local_dataset, split_equally_across_clients, combine_data
from update import get_local_update_objects, FedAvg, evaluate
import model_dino
import dino_variant


def main(args):
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )
    
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    save_dir = os.path.join(args.result_dir, args.dataset, f'iid_{args.iid}')
    model_dir = os.path.join(save_dir, f"{args.seed}_rate_{args.noise_rate}_threshold_{args.tau}")
    os.makedirs(model_dir, exist_ok=True)

    # Load and prepare dataset
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    val_dataset = wrap_as_local_dataset(val_dataset, tag='val', dataset_type='ham10000')
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)
    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)
    clients_test_combined = combine_data(clients_test_dataset_list, args.dataset)

    test_loader = torch.utils.data.DataLoader(dataset=clients_test_combined,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    # Model
    model_load = dino_variant._small_dino
    variant = dino_variant._small_variant
    dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth')
    net_glob = model_dino.ReinDinov2(variant, dino_state_dict, args.num_classes).cuda()
    net_glob.train()

    # Initialize f_G
    f_G = torch.randn(args.num_classes, 384, device=args.device)

    # Forget rate schedule
    forget_rate_schedule = np.ones(args.round3) * args.forget_rate
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, args.forget_rate ** 1, args.num_gradual)

    # Prepare local update objects
    dict_users = {i: list(range(len(clients_train_dataset_list[i]))) for i in range(args.num_clients)}
    dataset_train_all = combine_data(clients_train_dataset_list, args.dataset)
    local_update_objects = get_local_update_objects(args, dataset_train_all, dict_users, net_glob)

    # Training loop
    for epoch in range(args.round3):
        args.g_epoch = epoch
        args.forget_rate = forget_rate_schedule[epoch]

        if (epoch + 1) in args.schedule:
            args.lr *= args.lr_decay
            print(f"LR Decay Epoch {epoch+1}: lr -> {args.lr}")

        idxs_users = np.random.choice(range(args.num_clients), max(int(args.frac * args.num_clients), 1), replace=False)

        local_weights, local_losses, f_locals = [], [], []

        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args
            w, loss, f_k = local.train(copy.deepcopy(net_glob).cuda(), copy.deepcopy(f_G).cuda(), client_num)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)
            f_locals.append(f_k)

        # FedAvg
        w_glob = FedAvg(local_weights)
        net_glob.load_state_dict(w_glob)

        # Update f_G with cosine similarity
        sim = torch.nn.CosineSimilarity(dim=1)
        tmp, w_sum = 0, 0
        for f_k in f_locals:
            sim_weight = sim(f_G, f_k).reshape(args.num_classes, 1)
            w_sum += sim_weight
            tmp += sim_weight * f_k
        f_G = tmp / w_sum

        # Evaluation
        acc = evaluate(test_loader, net_glob)
        print(f"[Epoch {epoch+1}] Test Accuracy: {acc:.4f} %")

    print("Training Complete.")


if __name__ == '__main__':
    args = args_parser()
    main(args)