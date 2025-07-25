import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import sys
import numpy as np
import copy
import random
from PIL import Image
import torchvision.transforms.functional as TF

from collections import OrderedDict, Counter
from util import util
from util.util import *
from torch.utils.data import DataLoader
from util import dataset
from util.options import args_parser
from util.util import get_prob, create_data, balance_data, combine_data, load_data, transform_target, wrap_as_local_dataset

from rein import *
import update
import model
import model_trans
import model_dino
from dino_variant import _small_dino, _small_variant
from update import train, average_weights, average_weights_weighted, evaluate, train_forward, average_reins
from ensemble import compute_var, compute_mean_sq

args = args_parser()
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True

# result path
save_dir = args.result_dir + '/' + args.dataset + '/iid_' + str(args.iid)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(args)

    device = torch.device(f"cuda:{args.gpu}")
    config = {
        'device': device,
        'kd_lambda1': 10.0,
        'kd_lambda2': 10.0,
        'num_classes': args.num_classes,
    }
    # load and balance dataset
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)
    # train_dataset = wrap_as_local_dataset(train_dataset, tag='train', dataset_type='ham10000')
    # val_dataset = wrap_as_local_dataset(val_dataset, tag='val', dataset_type='ham10000')
    # test_dataset = wrap_as_local_dataset(test_dataset, tag='test', dataset_type='ham10000')

    # print('total original data counter')
    # print(Counter(np.array(train_dataset.local_clean_labels)))
    # print(Counter(np.array(val_dataset.local_clean_labels)))
    # print(Counter(np.array(test_dataset.local_clean_labels)))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    # used for validation
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                          batch_size=args.batch_size,
    #                                          num_workers=args.num_workers,
    #                                          drop_last=False,
    #                                          shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    class_num_list = [0 for _ in range(args.num_classes)]
    for data in train_dataset:
        class_num_list[int(data[2])] += 1
    class_num_list = torch.cuda.FloatTensor(class_num_list)

    # print('total test data counter')
    # print(Counter(test_dataset.local_clean_labels))

    # classifier = ReinsDinoVisionTransformer(**_small_variant)
    print('num layers : ', _small_variant['depth'])
    classifier = SelectiveReinsDinoVisionTransformer([0],**_small_variant)
    classifier.load_state_dict(torch.load('/home/work/Workspaces/yunjae_heo/FedLNL/checkpoints/dinov2_vits14_pretrain.pth'), strict=False)
    classifier.reins2 = copy.deepcopy(classifier.reins)
    classifier.linear_rein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    classifier.linear_rein2 = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    # classifier.linear_norein = nn.Linear(_small_variant['embed_dim'], args.num_classes).to(device)
    classifier.to(device)
    classifier.eval()
    classifier.train()

    optimizer = torch.optim.AdamW(list(classifier.reins2.parameters()) + list(classifier.linear_rein2.parameters()),
                                    lr=args.lr_f, 
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
            int(0.5 * args.round3), int(0.75 * args.round3), int(0.9 * args.round3)
        ])
    focal_loss_fn = FocalLossWithLogitAdjustment(
            gamma=2.0,
            class_log_prior=class_num_list,  # 기존 log-p list
            alpha=None, 
            reduction='none'
        )
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

    # for ep in range(args.round3):
    #     for inputs, _, targets, _ in train_loader:  # 3번째가 Clean targets
    #         inputs, targets = inputs.to(device), targets.to(device)
            
    #         with torch.no_grad():
    #             feats_norein = classifier.forward_features_no_rein(inputs)[:, 0, :]
    #         logits_norein = classifier.linear_norein(feats_norein)
    #         pred_norein = logits_norein.argmax(dim=1)
            
    #         feats = classifier.forward_features(inputs)[:, 0, :]
    #         logits_rein = classifier.linear_rein(feats)
    #         pred_rein = logits_rein.argmax(dim=1)
            
    #         feats2 = classifier.forward_features2(inputs)[:, 0, :]
    #         logits_rein2 = classifier.linear_rein2(feats2)
    #         pred_rein2 = logits_rein2.argmax(dim=1)

    #         with torch.no_grad():
    #             linear_accurate_norein = (pred_norein==targets)
    #             linear_accurate_rein = (pred_rein==targets)

    #         # loss_norein = focal_loss_fn(logits_norein, targets)
    #         # loss_rein = focal_loss_fn(logits_rein, targets)
    #         # loss_rein2 = focal_loss_fn(logits_rein2, targets)

    #         loss_norein = ce_loss_fn(logits_norein, targets)
    #         loss_rein = ce_loss_fn(logits_rein, targets)
    #         loss_rein2 = ce_loss_fn(logits_rein2, targets)

    #         loss = (linear_accurate_rein*loss_rein2).mean() +\
    #                 (linear_accurate_norein*loss_rein).mean() +\
    #                 loss_norein.mean()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     scheduler.step()

    #     if (ep+1) % 5 == 0:
    #         acc = util.validation_accuracy(classifier, test_loader, device, mode='rein2')
    #         print(f"[Epoch {ep+1}: Rein Test Acc = {acc * 100:.2f}%")
    for ep in range(args.round3):
        for inputs, targets, clean_targets, _ in train_loader:  # 3번째가 Clean targets
            inputs, targets, clean_targets = inputs.to(device), targets.to(device), clean_targets.to(device)
            # if ep==0:
            #     print(inputs[0].shape)
            #     print(torch.min(inputs[0]))
            #     print(torch.max(inputs[0]))
            #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(inputs.device)
            #     std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(inputs.device)

            #     # 첫 번째 이미지 복원
            #     # img = inputs[0] * std + mean  # (C, H, W)
            #     img = inputs[0]
            #     img = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # (H, W, C), [0,1]
            #     img = (img * 255).astype(np.uint8)  # uint8 변환
            #     Image.fromarray(img).save("input_tensor.png")
            #     exit()

            feats2 = classifier.forward_features2(inputs)[:, 0, :]
            logits_rein2 = classifier.linear_rein2(feats2)
            pred_rein2 = logits_rein2.argmax(dim=1)

            # loss_rein2 = focal_loss_fn(logits_rein2, targets).mean()
            loss_rein2 = ce_loss_fn(logits_rein2, targets).mean()
            # loss_rein2 = ce_loss_fn(logits_rein2, clean_targets).mean()

            optimizer.zero_grad()
            loss_rein2.backward()
            optimizer.step()
        scheduler.step()

        if (ep+1) % 5 == 0:
            bacc, nacc = util.validation_accuracy(classifier, test_loader, device, mode='rein2')
            print(f"[Epoch {ep+1}: Rein Test Acc = Balanced {bacc * 100:.2f}%, Normal {nacc * 100:.2f}")

if __name__ == "__main__":
    args = args_parser()
    torch.cuda.set_device(args.gpu)
    main(args)
