from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, test_transform

def get_dataset(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data == 'cifar10':
        data_path = '../data/cifar10'
        args.num_classes = 10
        trans_train, trans_val = get_transform()
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
        
    elif args.data == 'cifar100':
        data_path = '../data/cifar100'
        args.num_classes = 100
        trans_train, trans_val = get_transform()
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.data == 'clothing1m':
        data_path = os.path.abspath('..') + '/data/clothing1M/'
        args.num_classes = 14
        trans_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
        trans_val = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
        dataset_train = Clothing(data_path, trans_train, "train")
        dataset_test = Clothing(data_path, trans_val, "test")
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.data == 'ham10000':
        data_path = '/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000'
        args.num_classes = 7  # HAM10000은 7개 클래스

        train_transform, test_transform = get_transform()

        dataset_train = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
        dataset_test = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=test_transform)

        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    
    else:
        exit('Error: unrecognized dataset')

    if args.noisy_mod == 'iid':
        dict_users = iid_sampling(n_train, args.num_users, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)

    return dataset_train, dataset_test, dict_users


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.noisy_labels = {}
        self.clean_labels = {}
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        with open(self.root + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(self.root + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        if self.mode == 'train':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'minitrain':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + 'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.data)

