import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from collections import Counter
from util.options import args_parser
from util.util import *
from util import dataset
from util.util import load_data, wrap_as_local_dataset, split_equally_across_clients, combine_data
from update import evaluate, average_reins
from util.averaging import FoolsGold
import model_dino
import dino_variant
from util.attack import LocalUpdate, aggregate_weights, get_valid_models
from tensorboardX import SummaryWriter

def main(args):
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    save_dir = os.path.join(args.result_dir, args.dataset, f'iid_{args.iid}')
    model_dir = os.path.join(save_dir, f"{args.seed}_rate_{args.noise_rate}_threshold_{args.tau}")
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)

    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)
    clients_test_combined = combine_data(clients_test_dataset_list, args.dataset)
    
    print("\n[Client-wise Dataset Sizes]")
    clients_train_loader_list = []
    clients_test_loader_list = []

    def get_label_from_local_dataset(dataset, index):
        return dataset.local_noisy_labels[index]

    for i in range(args.num_clients):
        print(f"Client {i+1}:")
        print(f"  Train samples: {len(clients_train_dataset_list[i])}")
        print(f"  Test samples : {len(clients_test_dataset_list[i])}")

        train_loader = torch.utils.data.DataLoader(clients_train_dataset_list[i], batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
        
        # sampler = ImbalancedDatasetSampler(
        #     dataset=clients_train_dataset_list[i],  # 또는 local_dataset instance
        #     callback_get_label=get_label_from_local_dataset
        # )

        # train_loader = DataLoader(clients_train_dataset_list[i], sampler=sampler, batch_size=args.batch_size,
        #                         shuffle=False, num_workers=args.num_workers)

        clients_train_loader_list.append(train_loader)
    
    test_loader = torch.utils.data.DataLoader(dataset=clients_test_combined,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    # Model
    model_load = dino_variant._small_dino
    variant = dino_variant._small_variant
    # pretrained = torch.hub.load('facebookresearch/dinov2', model_load)
    # dino_state_dict = pretrained.state_dict()
    dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth')
    net_glob = model_dino.ReinDinov2(variant, dino_state_dict, args.num_classes).cuda()
    net_glob.eval()
    net_glob.train()
    
    

    summary = SummaryWriter('local')
    reweights = []
    if args.agg == 'fg':
        fg = FoolsGold(args)
    else:
        fg = None
    
    # Federated training loop with attack logic
    print('----------Starting Attack-Aware Federated Training----------')
    best_acc = 0.
    for round in tqdm(range(args.round3)):
        print(f'[Global Round {round + 1}]')
        net_glob.train()

        local_weights = []
        # local_models = []
        m = max(int(args.frac * args.num_clients), 1)
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)

        for client_id in idxs_users:
            local_model = copy.deepcopy(net_glob)
            local_model.cuda()
            local_model.train()

            local = LocalUpdate(args=args, dataset=[clients_train_loader_list[client_id], test_loader], idxs=None, tb=summary)
            w, loss, net_local = local.update_weights(net=local_model)
            
            local_weights.append(copy.deepcopy(w))
            # local_models.append(copy.deepcopy(net_local))

        # Remove invalid weights
        local_weights, _ = get_valid_models(local_weights)
        if len(local_weights) == 0:
            continue

        # Aggregate weights
        w_glob = aggregate_weights(args, local_weights, net_glob, reweights, fg)
        # average_reins(net_glob, local_models)
        net_glob.load_state_dict(w_glob)

        # Evaluate global model
        acc = evaluate(test_loader, net_glob)
        print(f'[Epoch {round + 1}] Test Accuracy: {acc:.4f} %')
        best_acc = max(best_acc, acc)

    print(f'Best Test Accuracy: {best_acc:.4f} %')
    print('----------Finished Training----------')


if __name__ == '__main__':
    args = args_parser()
    main(args)