import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import numpy as np
import os
from PIL import Image
from sparselearning.utils import TinyImageNetDataset
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly over multiple epochs
    Arguments:
        data_source (Dataset): dataset to sample from
        num_epochs (int) : Number of times to loop over the dataset
        start_itr (int) : which iteration to begin from
    """

    def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.num_epochs = num_epochs
        self.start_itr = start_itr
        self.batch_size = batch_size

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self):
        n = len(self.data_source)
        # Determine number of epochs
        num_epochs = int(np.ceil((n * self.num_epochs
                                  - (self.start_itr * self.batch_size)) / float(n)))
        # Sample all the indices, and then grab the last num_epochs index sets;
        # This ensures if we're starting at epoch 4, we're still grabbing epoch 4's
        # indices
        out = [torch.randperm(n)
               for epoch in range(self.num_epochs)][-num_epochs:]
        # Ignore the first start_itr % n indices of the first epoch
        out[0] = out[0][(self.start_itr * self.batch_size % n):]
        # if self.replacement:
        # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # return iter(.tolist())
        output = torch.cat(out).tolist()
        print('Length dataset output is %d' % len(output))
        return iter(output)

    def __len__(self):
        return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size


class ImageDataset(object):
    def __init__(self, args):

        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'baby_imagenet':
            assert args.img_size == 64, 'default image size of baby_imagenet is 64'
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            Dt_train = BabyImageNetDataset(root=args.data_path + '/train', transform=transform)
            Dt_test = BabyImageNetDataset(root=args.data_path + '/valid', transform=transform)


        elif args.dataset.lower() == 'tinyimagenet':
            assert args.img_size == 64, 'default image size of tinyimagenet is 64'
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                #                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            Dt_train = TinyImageNetDataset(root_dir=args.data_path, mode='train', transform=transform)
            Dt_test = TinyImageNetDataset(root_dir=args.data_path, mode='val', transform=transform)

        elif args.dataset.lower() == 'cub200':
            assert args.img_size == 128, 'default image size of CUB200 is 128'
            # Define transform
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # Load the CUB-200 dataset
            # print('?', args.data_path)
            Dt_train = Cub2011(args.data_path, train=True,
                               transform=transform, loader=default_loader, download=False)
            Dt_test = Cub2011(args.data_path, train=False,
                              transform=transform, loader=default_loader, download=False)


        elif args.dataset.lower() == 'imagenet':
            Dt = datasets.ImageNet
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'cifar10':
            train_DT = Dt(root=args.data_path, train=True, transform=transform, download=True)
            if args.mesample:
                self.train = torch.utils.data.DataLoader(
                    train_DT,
                    batch_size=args.dis_batch_size * args.n_critic * args.accumulation_steps,
                    num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed),
                    sampler=MultiEpochSampler(
                        train_DT, args.max_epoch, args.start_iter,
                        args.dis_batch_size * args.n_critic * args.accumulation_steps))
            else:
                self.train = torch.utils.data.DataLoader(
                    train_DT,
                    batch_size=args.dis_batch_size * args.n_critic * args.accumulation_steps, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed),
                    drop_last=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed))

        elif args.dataset.lower() == 'baby_imagenet':
            self.train = torch.utils.data.DataLoader(Dt_train,
                                                     batch_size=args.dis_batch_size * args.n_critic * args.accumulation_steps,
                                                     shuffle=True,
                                                     num_workers=args.num_workers, pin_memory=True,
                                                     worker_init_fn=random.seed(args.seed), drop_last=True)
            self.valid = torch.utils.data.DataLoader(Dt_test, batch_size=args.dis_batch_size, shuffle=True,
                                                     num_workers=args.num_workers, pin_memory=True,
                                                     worker_init_fn=random.seed(args.seed), drop_last=True)

        elif args.dataset.lower() == 'tinyimagenet':
            self.train = torch.utils.data.DataLoader(Dt_train,
                                                     batch_size=args.dis_batch_size * args.n_critic * args.accumulation_steps,
                                                     num_workers=args.num_workers, pin_memory=True,
                                                     worker_init_fn=random.seed(args.seed), drop_last=True,
                                                     sampler=train_sampler)
            self.valid = torch.utils.data.DataLoader(Dt_test, batch_size=args.dis_batch_size,
                                                     num_workers=args.num_workers, pin_memory=True,
                                                     worker_init_fn=random.seed(args.seed), drop_last=True)
        elif args.dataset.lower() == 'cub200':

            self.train = torch.utils.data.DataLoader(Dt_train,
                                                     batch_size=args.dis_batch_size * args.n_critic * args.accumulation_steps,
                                                     shuffle=True,
                                                     num_workers=args.num_workers, pin_memory=True,
                                                     worker_init_fn=random.seed(args.seed), drop_last=True)
            self.valid = torch.utils.data.DataLoader(Dt_test, batch_size=args.dis_batch_size, shuffle=True,
                                                     num_workers=args.num_workers, pin_memory=True,
                                                     worker_init_fn=random.seed(args.seed), drop_last=True)

        else:
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='unlabeled', transform=transform, download=True),
                batch_size=args.dis_batch_size * args.n_critic * args.accumulation_steps, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed), drop_last=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='unlabeled', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed))

        self.test = self.valid


class ImageDatasetLess(object):
    def __init__(self, args):

        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'imagenet':
            Dt = datasets.ImageNet
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'cifar10':
            dataset = Dt(root=args.data_path, train=True, transform=transform, download=True)
            subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.ratio)))
            self.train = torch.utils.data.DataLoader(
                subset,
                batch_size=args.dis_batch_size * args.n_critic, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed))

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed))
        else:
            dataset = Dt(root=args.data_path, split='train', transform=transform, download=False)
            subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.ratio)))
            self.train = torch.utils.data.DataLoader(
                subset,
                batch_size=args.dis_batch_size * args.n_critic, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed))

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=random.seed(args.seed))

        self.test = self.valid
