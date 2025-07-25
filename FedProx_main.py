import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import copy
from collections import Counter
from util.util import *
from util.options import args_parser
from util.util import load_data, wrap_as_local_dataset, get_prob, create_data, combine_data
from update import train_prox, average_weights, evaluate, average_reins
import model
import model_dino
import dino_variant

args = args_parser()
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True

# result path
save_dir = args.result_dir + '/' + args.dataset + '/iid_' + str(args.iid)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_dir = save_dir + str(args.seed)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("loading dataset...")
    train_dataset, val_dataset, test_dataset = load_data(args)
    train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    val_dataset = wrap_as_local_dataset(val_dataset, tag='val', dataset_type='ham10000')
    test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    if args.iid:
        train_prob = (1.0 / args.num_classes) * np.ones((args.num_clients, args.num_classes))
    else:
        if not os.path.exists(model_dir + '/' + 'train_prob.npy'):
            train_prob = get_prob(args.num_clients, p=1.0)
            np.save(model_dir + '/' + 'train_prob.npy', np.array(train_prob))
        else:
            train_prob = np.load(model_dir + '/' + 'train_prob.npy')

    # clients_train_dataset_list = create_data(
    #     train_prob, len(train_dataset.local_data) // args.num_clients,
    #     train_dataset, args.num_classes, args.dataset, args.seed
    # )
    # clients_test_dataset_list = create_data(
    #     train_prob, len(test_dataset.local_data) // args.num_clients,
    #     test_dataset, args.num_classes, args.dataset, args.seed
    # )
    train_subsets = split_equally_across_clients(train_dataset, args.num_clients, seed=args.seed)
    test_subsets = split_equally_across_clients(test_dataset, args.num_clients, seed=args.seed)

    clients_train_dataset_list = wrap_subsets_to_local_dataset(train_subsets, train_dataset)
    clients_test_dataset_list = wrap_subsets_to_local_dataset(test_subsets, test_dataset)
    clients_test_dataset_combination = combine_data(clients_test_dataset_list, args.dataset)
    clients_train_dataset_combination = combine_data(clients_train_dataset_list, args.dataset)

    test_dataset_all = combine_data(clients_test_dataset_list, args.dataset)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_all, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False
    )

    clients_train_loader_list = [
        torch.utils.data.DataLoader(
            dataset=clients_train_dataset_list[i], batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=True, drop_last=False
        ) for i in range(args.num_clients)
    ]
    
    clients_train_class_num_list = []
    for i in range(args.num_clients):
        class_num_list = [0 for _ in range(args.num_classes)]
        for j in range(len(clients_train_dataset_list[i])):
            class_num_list[int(clients_train_dataset_list[i][j][1])] += 1
        
        # Create a tensor from class_num_list and move it to the specified device
        class_num_tensor = torch.cuda.FloatTensor(class_num_list)

        # Perform calculations
        class_p_list = class_num_tensor / class_num_tensor.sum()
        class_p_list = torch.log(class_p_list)
        class_p_list = class_p_list.view(1, -1)
        
        clients_train_class_num_list.append(class_p_list)

    # construct model
    print('constructing model...')
    if args.dataset == 'ham10000':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
        dino_state_dict = torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth')
        global_model = model_dino.ReinDinov2(variant, dino_state_dict, args.num_classes).cuda()
    else:
        raise NotImplementedError

    print('Starting Federated Proximal Training...')
    Final_acc = 0.
    for rd in range(args.round3):
        # client_model_list = []
        local_weights_list = []
        selected = random.sample(range(args.num_clients), args.num_clients)
        for cid in selected:
            print(f"[Round {rd+1}] Training client {cid+1}")
            local_model = copy.deepcopy(global_model).cuda()
            # optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr_w, momentum=args.momentum)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=args.lr_f, weight_decay=args.weight_decay)
            train_loader = clients_train_loader_list[cid]
            train_prox(train_loader, epoch=rd, model=local_model,
                       global_model=global_model, optimizer1=optimizer, mu=args.mu, args=args,
                       class_p_list=clients_train_class_num_list[cid])
            local_weights_list.append(copy.deepcopy(local_model.state_dict()))
            # client_model_list.append(local_model.backbone)

        global_weights = average_weights(local_weights_list)
        global_model.load_state_dict(global_weights)
        # average_reins(global_model.backbone, client_model_list)

        test_acc = evaluate(test_loader, global_model)
        print(f"Round [{rd+1}] Test Accuracy: {test_acc:.4f}")

        if round+1 > args.round3-10:
            Final_acc += test_acc
        
    print('----------Finished FedAvg Training----------')
    Final_acc /= 10
    print(f'Final Test Accuracy: {Final_acc:.4f} %')
        
    # return evaluate(test_loader, global_model)

if __name__ == '__main__':
    acc_list = []
    for exp_id in range(args.num_exp):
        print(f"\n[Experiment {exp_id+1}]")
        args.seed = exp_id + 1
        acc = main(args)
        acc_list.append(acc)
    print("All Results:", acc_list)
    print("Mean Accuracy:", np.mean(acc_list))
    print("Std:", np.std(acc_list, ddof=1))