from PIL import Image
import torch.utils.data as Data
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils
import util

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
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

class ham10000_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.3, split_percentage=0.9, seed=1, num_classes=7, feature_size=512, norm_std=0.1):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000/train_images.npy', allow_pickle=True)
        original_labels = np.load('/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000/train_labels.npy', allow_pickle=True)
        data = torch.from_numpy(original_images)
        # targets = torch.from_numpy(original_labels)
        targets = original_labels

        dataset = zip(data, targets)
        
        new_labels = util.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, norm_std, seed, cache_path='/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000/soft_labels.npy')

        self.train_data, self.val_data, self.train_noisy_labels, self.val_noisy_labels, self.train_clean_labels, self.val_clean_labels = \
            util.data_split(original_images, targets, new_labels, num_classes, split_percentage, seed)

        # Image.fromarray(original_images[0].astype(np.uint8)).save("original_0.png")

        # train_img = self.train_data[0].astype(np.uint8)
        # Image.fromarray(train_img).save("split_train_0.png")
        # exit()

        # if self.train:      
        #     self.train_data = self.train_data.reshape((-1, 3, 224, 224))
        #     self.train_data = self.train_data.transpose((0, 2, 3, 1)).astype(np.uint8)
        #     print('building ham10000 train dataset')
        #     print(self.train_data.shape)
        
        # else:
        #     self.val_data = self.val_data.reshape((-1, 3, 224, 224))
        #     self.val_data = self.val_data.transpose((0, 2, 3, 1)).astype(np.uint8)
        #     print('building ham10000 val dataset')
        #     print(self.val_data.shape)
        if self.train:
            self.train_data = self.train_data.astype(np.uint8)
            print("building ham10000 train dataset")
            print(self.train_data.shape)
        else:
            self.val_data = self.val_data.astype(np.uint8)
            print("building ham10000 val dataset")
            print(self.val_data.shape)

    def __getitem__(self, index):
           
        if self.train:
            img, noisy_label, clean_label = self.train_data[index], self.train_noisy_labels[index], self.train_clean_labels[index]
            
        else:
            img, noisy_label, clean_label = self.val_data[index], self.val_noisy_labels[index], self.val_clean_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            noisy_label = self.target_transform(noisy_label)
            clean_label = self.target_transform(clean_label)
     
        return img, noisy_label, clean_label, index

    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class ham10000_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000/test_images.npy', allow_pickle=True)
        self.test_labels = np.load('/home/work/Workspaces/yunjae_heo/FedLNL/data/ham10000/test_labels.npy', allow_pickle=True)
        # self.test_data = self.test_data.reshape((1512,3,224,224))
        # self.test_data = self.test_data.transpose((0, 2, 3, 1))

        print('building ham10000 test dataset')
        print(self.test_data.shape)

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, label, index
    
    def __len__(self):
        return len(self.test_data)
    
class distilled_dataset(Data.Dataset):
    def __init__(self, distilled_images, distilled_noisy_labels, distilled_pseudo_labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.distilled_images = distilled_images
        self.distilled_noisy_labels = distilled_noisy_labels
        self.distilled_pseudo_labels = distilled_pseudo_labels

    def __getitem__(self, index):
        img, pseudo_label, noisy_label = self.distilled_images[index], self.distilled_pseudo_labels[index], self.distilled_noisy_labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            pseudo_label, noisy_label = self.target_transform(pseudo_label), self.target_transform(noisy_label)

        return img, noisy_label, pseudo_label, index

    def __len__(self):
        return len(self.distilled_images)


class local_dataset(Data.Dataset):
    def __init__(self, local_data, local_noisy_labels, local_clean_labels, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.local_data = local_data
        self.local_noisy_labels = local_noisy_labels
        self.local_clean_labels = local_clean_labels

    def __getitem__(self, index):
        img, noisy_label, clean_label = self.local_data[index], self.local_noisy_labels[index], \
                                        self.local_clean_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            clean_label, noisy_label = self.target_transform(clean_label), self.target_transform(noisy_label)

        return img, noisy_label, clean_label, index

    def __len__(self):
        return len(self.local_data)