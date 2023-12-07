from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import random

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay, ConstantRate
# from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18
from sparselearning.gan_datasets import ImageDataset
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, \
    flops_calculator
from sparselearning.gan_utils.utils import set_log_dir, save_checkpoint, save_checkpoint_finetune, create_logger, \
    set_seed, infiniteloop, plot_FDB

# GAN-related imports
# from sparselearning.gan_models import SNGAN_Generator, SNGAN_Discriminator
from sparselearning.gan_functions import train, validate, LinearLrDecay, load_params, copy_params
from sparselearning.gan_utils.inception_score import _init_inception
from sparselearning.gan_utils.fid_score import create_inception_graph, check_or_download_inception
from sparselearning.backbones import resnet, biggan, dcgan, lottery_sngan
from sparselearning.backbones.big_resnet import Discriminator, Generator
from sparselearning.post_prune import post_hoc_prune

import torchvision
import torchvision.transforms as transforms
import warnings
import collections
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True


# For easy manipulation of flags
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


logger = None

models = {}

net_G_models = {
    'dcgan.32': dcgan.Generator32,
    'dcgan.48': dcgan.Generator48,
    'resnet.32': resnet.ResGenerator32,
    'resnet.48': resnet.ResGenerator48,
    'resnet.64': resnet.ResGenerator64,
    'biggan.32': biggan.Generator32,
    'biggan.64': Generator,
    'biggan.128': Generator,
    'lottery': lottery_sngan.SNGAN_Generator,
}

net_D_models = {
    'dcgan.32': dcgan.Discriminator32,
    'dcgan.48': dcgan.Discriminator48,
    'resnet.32': resnet.ResDiscriminator32,
    'resnet.48': resnet.ResDiscriminator48,
    'resnet.64': resnet.ResDiscriminator64,
    'biggan.32': biggan.Discriminator32,
    'biggan.64': Discriminator,
    'biggan.128': Discriminator,
    'lottery': lottery_sngan.SNGAN_Discriminator,
}


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = os.path.join(args.save_path, '{0}.log'.format(args.expid))

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def main(rank, world_size):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=-1, metavar='S', help='random seed (default: random)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many iterations to wait before logging training status and save models')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', ], help='The optimizer to use. Default: adam. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save-path', type=str, default='./models', help='path to save the models')
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'stl10', 'baby_imagenet', 'tinyimagenet', 'cub200', 'imagenet'],
                        default='cifar10')
    parser.add_argument('--data-path', type=str, default='', help='path to dataset.')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-iter', type=int, default=1)
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1,
                        help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true',
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true',
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--num_workers', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--ddp', action='store_true', help='Enables parallel DDP training')
    #####################
    # GAN SETTING
    #####################

    # GAN model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet.32', choices=list(net_G_models.keys()),
                        help='backbone used for GAN')
    parser.add_argument('--df-dim', type=int, default=128, help='')
    parser.add_argument('--gf-dim', type=int, default=256, help='')
    parser.add_argument('--bottom-width', type=int, default=4, help='')
    parser.add_argument('--latent-dim', type=int, default=128, help='')
    parser.add_argument('--g-spectral-norm', type=str2bool, default=False,
                        help='whether add spectral norm on generator.')
    parser.add_argument('--d-norm', type=str, default='SN', choices=['SN', 'GN', 'none'],
                        help='what norm to use for discriminator.')
    parser.add_argument('--init-type', type=str, choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        default='xavier_uniform', help='initialization method for models.')
    parser.add_argument('--cr', type=float, default=0.0, help='weight for consistency regularization')
    parser.add_argument('--num-classes', type=int, default=10, help='')

    # GAN training hyper-parameters:
    parser.add_argument('--dis-batch-size', type=int, default=64, help='')
    parser.add_argument('--gen-batch-size', type=int, default=64, help='')
    parser.add_argument('--img-size', type=int, default=64, help='resolution size of image')
    parser.add_argument('--channels', type=int, default=3, help='channel size of image')
    parser.add_argument('--loss', type=str, default='hinge', choices=['hinge', 'wass', 'BCE', 'softplus', 'softhinge'],
                        help='what loss function to choose, default Hinge loss.')

    parser.add_argument('--max-iter', type=int, default=0, help='')
    parser.add_argument('--max-epoch', type=int, default=0, help='')
    parser.add_argument('--gen-lr', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--dis-lr', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--lr-decay', action='store_true', help='Use learning rate decay or not')
    parser.add_argument('--ema', type=float, default=0.9999, help='EMA coefficient')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam')
    parser.add_argument('--n-critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--lr-decay-start', type=int, default=0, help='start iter to learning rate decay')
    parser.add_argument('--condition', action='store_true', help='Condition GAN')
    parser.add_argument('--mesample', action='store_true', help='Multi Epoch sampler')
    parser.add_argument('--diff_aug', action='store_true', help='DiffAugment')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='accumulation_steps for generator and discriminator')

    # GAN validation hyper-parameters
    parser.add_argument('--val-freq', type=int, default=5000, help='validation frequency')
    parser.add_argument('--fid-path', type=str, default='', help='path to fid_stat')

    # GAN test hyper-parameters:
    parser.add_argument('--eval-batch-size', type=int, default=100, help='')
    parser.add_argument('--num-eval-imgs', type=int, default=10000, help='')

    # Future use:
    parser.add_argument('--ratio', type=float, default=1.0, help='')

    #
    parser.add_argument('--parallel', action='store_true', help='Enables parallel training')

    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()

    local_rank = rank % 8
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{10003}', world_size=world_size, rank=rank)
    args.dis_batch_size = int(args.dis_batch_size / world_size)
    args.gen_batch_size = int(args.gen_batch_size / world_size)
    print('batch_size per gpu:', args.dis_batch_size)

    # Create random seed
    if args.seed is None or args.seed < 0:
        args.seed = random.randint(1, 100000)

    # Assert both G and D are sparse for post-hoc pruning
    if args.posthoc:
        assert (not args.sparse_G) and (not args.sparse_D)
        print('[*] Will perform dense training, then fine-tuning with generator sparsity: {}'.format(args.density_G))

    # Set up training set
    dataset = ImageDataset(args)
    train_loader = dataset.train

    train_iter = infiniteloop(train_loader)

    if args.max_epoch:
        args.max_iter = args.max_epoch * len(dataset.train)

    # Make sure mask is on for dynamic adjust
    if args.adjust_mode != 'none':
        assert args.sparse_D == True

    # Make sure that we are computing BR if we are using dynamic adjust
    if args.adjust_mode == 'dynamic_adjust':
        assert args.cal_br == True
        # make sure we have at least 10 br measurements for each discriminator update
        assert args.br_freq <= args.update_frequency_D // 10

    # Create special save_path
    # timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))
    hparams = '{0}_G_{1}_D_{2}_'.format(args.model, args.density_G, args.density_D)
    args.expid = hparams + str(args.seed)
    args.save_path = os.path.join(args.save_path, args.expid)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    # if not os.path.exists('./logs'): os.mkdir('./logs')

    setup_logger(args)
    # print_and_log(args)
    print_and_log('All arguments: \n')
    for arg_name in vars(args):
        print_and_log('{}: {}'.format(arg_name, getattr(args, arg_name)))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)

    # Set up IS computations
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # Set up FID computations
    if args.dataset.lower() == 'cifar10':
        fid_stat = os.path.join(args.fid_path, 'fid_stats_cifar10_train.npz')
        print(fid_stat)
        args.fid_stat_test = os.path.join(args.fid_path, 'fid_stats_cifar10_test.npz')
    elif args.dataset.lower() == 'stl10':
        fid_stat = os.path.join(args.fid_path, 'stl10.unlabeled.48.npz')
        args.fid_stat_test = os.path.join(args.fid_path, 'stl10.unlabeled.48.npz')
    elif args.dataset.lower() == 'tinyimagenet':
        fid_stat = os.path.join(args.fid_path, 'tiny_train.npz')
        args.fid_stat_test = os.path.join(args.fid_path, 'tiny_val.npz')
    else:
        raise NotImplementedError('no fid stat for %s' % args.dataset.lower())
    assert os.path.exists(fid_stat)

    # Create models
    if args.model not in net_G_models.keys():
        print('You need to select an existing model via the --model argument. Available models include: ')
        for key in net_G_models.keys():
            print('\t{0}'.format(key))
        raise Exception('You need to select a model')
    else:
        model = {}
        model['G'] = net_G_models[args.model](args).to(device)
        model['D'] = net_D_models[args.model](args).to(device)

        model['G'] = DDP(model['G'], device_ids=[local_rank])
        model['D'] = DDP(model['D'], device_ids=[local_rank])

        if args.parallel:
            class G_D(nn.Module):
                def __init__(self, G, D):
                    super(G_D, self).__init__()
                    self.G = G
                    self.D = D

            model_GD = G_D(model['G'], model['D'])
            model_GD = torch.nn.DataParallel(model_GD)

        # Add spectral norm if necessary:
        if args.model != 'biggan.64' and args.model != 'biggan.128':
            if args.d_norm == 'SN':
                def add_sn(m):
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                        return torch.nn.utils.spectral_norm(m)
                    else:
                        return m

                model['D'].apply(add_sn)

    print_and_log(model)
    print_and_log('=' * 60)
    print_and_log(args.model)
    print_and_log('=' * 60)

    print_and_log('=' * 60)
    print_and_log('Prune mode: [G] {0}, [D] {1}'.format(args.death_G, args.death_D))
    print_and_log('Growth mode: [G] {0}, [D] {1}'.format(args.growth_G, args.growth_D))
    print_and_log('Redistribution mode: {0}'.format(args.redistribution))
    print_and_log('=' * 60)

    optimizer = None
    # if args.optimizer == 'sgd':
    # optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)

    if args.optimizer == 'adam':
        gen_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model['G'].parameters()),
                                   lr=args.gen_lr, betas=(args.beta1, args.beta2))
        dis_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model['D'].parameters()),
                                   lr=args.dis_lr, betas=(args.beta1, args.beta2))
        optimizer = {'G': gen_optimizer, 'D': dis_optimizer, }
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer.')

    if args.lr_decay:
        def decay_rate(step):
            period = max(args.max_iter * args.multiplier - args.lr_decay_start, 1)
            return 1 - max(step - args.lr_decay_start, 0) / period

        gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=decay_rate)
        dis_scheduler = torch.optim.lr_scheduler.LambdaLR(dis_optimizer, lr_lambda=decay_rate)
        scheduler = {'G': gen_scheduler, 'D': dis_scheduler}
    else:
        scheduler = None

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))

    start_iter = 0
    best_fid = 1e4

    mask = {}
    if args.sparse_G:
        if args.death_rate_schedule_G == 'cosine':
            decay_G = CosineDecay(args.death_rate_G, (args.max_iter * args.multiplier))
        elif args.death_rate_schedule_G == 'linear':
            decay_G = LinearDecay(args.death_rate_G, (args.max_iter * args.multiplier))
        elif args.death_rate_schedule_G == 'constant':
            decay_G = ConstantRate(args.death_rate_G, (args.max_iter * args.multiplier))
        mask_G = Masking(optimizer['G'], death_rate=args.death_rate_G, death_mode=args.death_G,
                         death_rate_decay=decay_G, growth_mode=args.growth_G,
                         redistribution_mode=args.redistribution, fix=args.fix_G, adjust_mode='none',
                         update_frequency=args.update_frequency_G, hybrid_alpha=args.hybrid_alpha, obj_name='generator',
                         args=args, resurrect=args.resu_G, resu_decay=args.ema)
        ## TODO: Make sparse_init different for G and D?
        print_and_log('=' * 60)
        print_and_log('Start to initialize Generator.')
        mask_G.add_module(model['G'], sparse_init=args.sparse_init, density=args.density_G)
        mask['G'] = mask_G

    if args.sparse_D:
        ## TODO: Maybe consider removing *args.n_critic?
        if args.death_rate_schedule_D == 'cosine':
            decay_D = CosineDecay(args.death_rate_D, (args.max_iter * args.n_critic * args.multiplier))
        elif args.death_rate_schedule_D == 'linear':
            decay_D = LinearDecay(args.death_rate_D, (args.max_iter * args.n_critic * args.multiplier))
        elif args.death_rate_schedule_D == 'constant':
            decay_D = ConstantRate(args.death_rate_D, (args.max_iter * args.n_critic * args.multiplier))
            if args.dd_b:
                decay_D_cos = CosineDecay(args.death_rate_D_dd_b, (args.max_iter * args.n_critic * args.multiplier))
                decay_D = [decay_D, decay_D_cos]
                # also update death rate with a list [first_half, second_half]
                args.death_rate_D = [args.death_rate_D, args.death_rate_D_dd_b]
        # decay_D = CosineDecay(args.death_rate_D, (args.max_iter * args.n_critic * args.multiplier))

        mask_D = Masking(optimizer['D'], death_rate=args.death_rate_D, death_mode=args.death_D,
                         death_rate_decay=decay_D, growth_mode=args.growth_D,
                         redistribution_mode=args.redistribution, fix=args.fix_D,
                         adjust_mode=args.adjust_mode, da_bound=(args.da_lb, args.da_ub, args.da_disb),
                         update_frequency=args.update_frequency_D * args.n_critic, hybrid_alpha=args.hybrid_alpha,
                         obj_name='discriminator',
                         args=args, dynamic_bound=args.dynamic_bound, resurrect=args.resu_D, resu_decay=args.ema,
                         doub_dst=args.dd_b)
        print_and_log('=' * 60)
        print_and_log('Start to initialize Discriminator.')
        mask_D.add_module(model['D'], sparse_init=args.sparse_init, density=args.density_D)
        mask['D'] = mask_D

    # TODO: resume flops
    # Construct flops calculator
    #                   |           during D training               |               during G training 
    flops_batch_size_G = 1 * args.dis_batch_size * args.n_critic + 3 * args.gen_batch_size
    flops_calculator_G = flops_calculator(model['G'], (args.latent_dim,), device, batch_size=flops_batch_size_G,
                                          condition=args.condition)
    if 'G' in mask.keys():
        # mask will change during training
        if mask['G'].prune_every_k_steps:
            flops_calculator_G.update_iter_and_training_flops(mask['G'].density_dict, mask['G'].prune_every_k_steps)
            print('[*] Ongoing training G FLOPS:{}={:e}/{:e}'.format(
                flops_calculator_G.training_flops / flops_calculator_G.dense_training_flops,
                flops_calculator_G.training_flops, flops_calculator_G.dense_training_flops))
        # mask is fixed during training
        else:
            flops_calculator_G.update_iter_and_training_flops(mask['G'].density_dict, args.max_iter * args.multiplier)
            print('[*] Total sparse training G FLOPS:{}={:e}/{:e}'.format(
                flops_calculator_G.training_flops / flops_calculator_G.dense_training_flops,
                flops_calculator_G.training_flops, flops_calculator_G.dense_training_flops))
    else:
        flops_calculator_G.update_training_flops(args.max_iter * args.multiplier)
        print('[*] Total dense training G FLOPS:{}={:e}/{:e}'.format(
            flops_calculator_G.training_flops / flops_calculator_G.dense_training_flops,
            flops_calculator_G.training_flops, flops_calculator_G.dense_training_flops))

    #                   |           during D training               |               during G training 
    flops_batch_size_D = 3 * args.dis_batch_size * args.n_critic + 1 * args.gen_batch_size
    flops_calculator_D = flops_calculator(model['D'], (3, args.img_size, args.img_size), device,
                                          batch_size=flops_batch_size_D, condition=args.condition)
    if 'D' in mask.keys():
        # mask will change during training
        if mask['D'].prune_every_k_steps:
            flops_calculator_D.update_iter_and_training_flops(mask['D'].density_dict, mask['D'].prune_every_k_steps)
            print('[*] Ongoing training D FLOPS:{}={:e}/{:e}'.format(
                flops_calculator_D.training_flops / flops_calculator_D.dense_training_flops,
                flops_calculator_D.training_flops, flops_calculator_D.dense_training_flops))
        # mask is fixed during training
        else:
            flops_calculator_D.update_iter_and_training_flops(mask['D'].density_dict,
                                                              args.max_iter * args.n_critic * args.multiplier)
            print('[*] Total sparse training D FLOPS:{}={:e}/{:e}'.format(
                flops_calculator_D.training_flops / flops_calculator_D.dense_training_flops,
                flops_calculator_D.training_flops, flops_calculator_D.dense_training_flops))
    else:
        flops_calculator_D.update_training_flops(args.max_iter * args.n_critic * args.multiplier)
        print('[*] Total dense training D FLOPS:{}={:e}/{:e}'.format(
            flops_calculator_D.training_flops / flops_calculator_D.dense_training_flops,
            flops_calculator_D.training_flops, flops_calculator_D.dense_training_flops))

    FC = {'G': flops_calculator_G, 'D': flops_calculator_D}

    gen_avg_param = copy_params(model['G'])

    if args.resume:
        if os.path.isfile(args.resume):
            print_and_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['curr_iter']
            best_fid = checkpoint['best_fid']
            model['G'].load_state_dict(checkpoint['gen_state_dict'])
            model['D'].load_state_dict(checkpoint['dis_state_dict'])
            optimizer['G'].load_state_dict(checkpoint['gen_optimizer'])
            optimizer['D'].load_state_dict(checkpoint['dis_optimizer'])
            # TODO: support weight resurrection

            # avg_gen_net = copy.deepcopy(model['G'])
            avg_gen_net = net_G_models[args.model](args).to(device)
            avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
            gen_avg_param = copy_params(avg_gen_net)
            del avg_gen_net

            if 'G' in mask.keys():
                assert checkpoint['decay_G_state_dict'] is not None
                decay_G.cosine_stepper.load_state_dict(checkpoint['decay_G_state_dict'])
            if 'D' in mask.keys():
                assert checkpoint['decay_D_state_dict'] is not None
                decay_D.cosine_stepper.load_state_dict(checkpoint['decay_D_state_dict'])

            print_and_log("=> loaded checkpoint '{}' (iteration {})"
                          .format(args.resume, checkpoint['curr_iter']))
        else:
            print_and_log("=> no checkpoint found at '{}'".format(args.resume))

    discriminator_loss_queue = deque([])
    br_record = []
    density_record = []
    fid_record = []
    training_flops = 0

    for curr_iter in range(int(args.start_iter), int(args.max_iter) + 1):
        train(args, model, optimizer, gen_avg_param, train_iter, curr_iter, discriminator_loss_queue, br_record,
              density_record, scheduler, mask, args.condition, FC)
        np.save(os.path.join(args.save_path, 'br.npy'), br_record)
        np.save(os.path.join(args.save_path, 'density.npy'), density_record)
        if curr_iter and curr_iter % args.val_freq == 0 or curr_iter == int(args.max_iter):
            backup_param = copy_params(model['G'])
            load_params(model['G'], gen_avg_param)

            # Check SEMA sparsity
            total = 0
            remain_params = 0

            for name, weight in model['G'].named_parameters():
                if 'bn' not in name:
                    total += weight.numel()
                    remain_params += (weight != 0).sum()

            print_and_log('Sparsity of SEMA is {}'.format(1.0 * remain_params / total))

            inception_score, fid_score = validate(args, fixed_z, fid_stat, model['G'], curr_iter,
                                                  condition=args.condition)
            print_and_log(
                'Inception score: %.4f, FID score: %.4f || @ iteration %d.' % (inception_score, fid_score, curr_iter))
            load_params(model['G'], backup_param)

            fid_record.append(fid_score)

            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
            print_and_log('Best FID so far: {}'.format(best_fid))
        else:
            is_best = False

        # Save when validate or log_interval
        if (curr_iter and curr_iter % args.val_freq == 0) or (curr_iter == int(args.max_iter)) or (
                curr_iter and curr_iter % args.log_interval == 0):
            # avg_gen_net = copy.deepcopy(model['G'])
            avg_gen_net = net_G_models[args.model](args).to(device)
            load_params(avg_gen_net, gen_avg_param)
            if 'G' in mask.keys() and hasattr(decay_G, 'cosine_stepper'):
                decay_G_state_dict = decay_G.cosine_stepper.state_dict()
            else:
                decay_G_state_dict = None

            if 'D' in mask.keys() and hasattr(decay_D, 'cosine_stepper'):
                decay_D_state_dict = decay_D.cosine_stepper.state_dict()
            else:
                decay_D_state_dict = None

            save_checkpoint({
                'curr_iter': curr_iter + 1,
                'model': args.model,
                'gen_state_dict': model['G'].state_dict(),
                'dis_state_dict': model['D'].state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': optimizer['G'].state_dict(),
                'dis_optimizer': optimizer['D'].state_dict(),
                'decay_G_state_dict': decay_G_state_dict,
                'decay_D_state_dict': decay_D_state_dict,
                'best_fid': best_fid,
                # 'path_helper': args.path_helper,
                'seed': args.seed,
            }, is_best, args.save_path)

            # also save the last checkpoint
            if curr_iter == int(args.max_iter):
                save_checkpoint({
                    'curr_iter': curr_iter + 1,
                    'model': args.model,
                    'gen_state_dict': model['G'].state_dict(),
                    'dis_state_dict': model['D'].state_dict(),
                    'avg_gen_state_dict': avg_gen_net.state_dict(),
                    'gen_optimizer': optimizer['G'].state_dict(),
                    'dis_optimizer': optimizer['D'].state_dict(),
                    'decay_G_state_dict': decay_G_state_dict,
                    'decay_D_state_dict': decay_D_state_dict,
                    'best_fid': best_fid,
                    # 'path_helper': args.path_helper,
                    'seed': args.seed,
                }, is_best=False, output_dir=args.save_path, filename='final.pth')
            del avg_gen_net
        dist.barrier()
    dist.destroy_process_group()
    # Summarizing total flops
    total_training_flops = FC['G'].training_flops + FC['D'].training_flops
    total_dense_training_flops = FC['G'].dense_training_flops + FC['D'].dense_training_flops
    print('G FLOPS sparsity:{}={:e}/{:e}'.format(FC['G'].training_flops / FC['G'].dense_training_flops,
                                                 FC['G'].training_flops, FC['G'].dense_training_flops))
    print('D FLOPS sparsity:{}={:e}/{:e}'.format(FC['D'].training_flops / FC['D'].dense_training_flops,
                                                 FC['D'].training_flops, FC['D'].dense_training_flops))
    print('FLOPS sparsity:{}={:e}/{:e}'.format(total_training_flops / total_dense_training_flops, total_training_flops,
                                               total_dense_training_flops))

    print('-' * 60)
    print('Testing model:')

    best_ckpt_path = os.path.join(args.save_path, 'checkpoint_best.pth')

    if os.path.isfile(best_ckpt_path):
        print_and_log("=> loading checkpoint '{}'".format(best_ckpt_path))
        checkpoint = torch.load(best_ckpt_path)
        model['G'].load_state_dict(checkpoint['avg_gen_state_dict'])

        print_and_log("=> loaded best checkpoint '{}' (from iteration {})"
                      .format(best_ckpt_path, checkpoint['curr_iter']))
    else:
        print_and_log("=> no checkpoint found at '{}'".format(best_ckpt_path))

    inception_score, fid_score = validate(args, fixed_z, fid_stat, model['G'], checkpoint['curr_iter'], also_test=True,
                                          condition=args.condition)

    if args.posthoc:

        finetune_iters_ratio = 0.5

        best_finetune_fid = 1e4
        # if post-hoc pruning and fine-tuning
        print('Start to post-hoc prune')
        print_and_log('=' * 60)
        print_and_log('Start to post-hoc prune Generator with sparsity {}.'.format(args.density_G))
        mask = {}
        mask_G = Masking(optimizer['G'], fix=True, adjust_mode='none', obj_name='generator', args=args)
        mask_G.add_module(model['G'], sparse_init='lottery_ticket', density=args.density_G)
        mask['G'] = mask_G

        # Set the new gen_avg_param after post-hoc pruning
        gen_avg_param = copy_params(model['G'])

        # Test again
        backup_param = copy_params(model['G'])
        load_params(model['G'], gen_avg_param)

        # Check SEMA sparsity
        total = 0
        remain_params = 0

        for name, weight in model['G'].named_parameters():
            if 'bn' not in name:
                total += weight.numel()
                remain_params += (weight != 0).sum()

        print_and_log('Sparsity of SEMA is {}'.format(1.0 * remain_params / total))
        print('Start to test pruned generator')

        inception_score, fid_score = validate(args, fixed_z, fid_stat, model['G'], args.max_iter, also_test=True,
                                              condition=args.condition)
        print_and_log('Inception score: %.4f, FID score: %.4f ' % (inception_score, fid_score))

        # Load back the model['G'] params
        load_params(model['G'], backup_param)

        fid_record.append(fid_score)
        if fid_score < best_finetune_fid:
            best_finetune_fid = fid_score
            is_best = True
        else:
            is_best = False

        print_and_log('Best finetune FID so far: {}'.format(best_finetune_fid))

        # Calculate FLOPs for fine-tuning
        flops_calculator_G.update_iter_and_training_flops(mask['G'].density_dict,
                                                          args.max_iter * args.multiplier * finetune_iters_ratio)
        flops_calculator_D.update_training_flops(args.max_iter * args.n_critic * args.multiplier * finetune_iters_ratio)

        # Fine-tuning with finetune_iters_ratio more iterations
        for curr_iter in range(int(args.max_iter * finetune_iters_ratio) + 1):
            train(args, model, optimizer, gen_avg_param, train_iter, curr_iter, discriminator_loss_queue, br_record,
                  density_record, scheduler, mask, args.condition, FC)
            np.save(os.path.join(args.save_path, 'br.npy'), br_record)
            np.save(os.path.join(args.save_path, 'density.npy'), density_record)
            if curr_iter and curr_iter % args.val_freq == 0 or curr_iter == int(args.max_iter * finetune_iters_ratio):
                backup_param = copy_params(model['G'])
                load_params(model['G'], gen_avg_param)

                # Check SEMA sparsity
                total = 0
                remain_params = 0

                for name, weight in model['G'].named_parameters():
                    if 'bn' not in name:
                        total += weight.numel()
                        remain_params += (weight != 0).sum()

                print_and_log('Sparsity of SEMA is {}'.format(1.0 * remain_params / total))

                inception_score, fid_score = validate(args, fixed_z, fid_stat, model['G'], curr_iter + args.max_iter,
                                                      condition=args.condition)
                print_and_log('Inception score: %.4f, FID score: %.4f || @ iteration %d.' % (
                    inception_score, fid_score, curr_iter + args.max_iter))
                load_params(model['G'], backup_param)

                fid_record.append(fid_score)

                if fid_score < best_finetune_fid:
                    best_finetune_fid = fid_score
                    is_best = True
                else:
                    is_best = False
                print_and_log('Best Fine-tune FID so far: {}'.format(best_finetune_fid))
            else:
                is_best = False

            # Save when validate or log_interval
            if (curr_iter and curr_iter % args.val_freq == 0) or (
                    curr_iter == int(args.max_iter * finetune_iters_ratio)) or (
                    curr_iter and curr_iter % args.log_interval == 0):
                # avg_gen_net = copy.deepcopy(model['G'])
                avg_gen_net = net_G_models[args.model](args).to(device)
                load_params(avg_gen_net, gen_avg_param)

                save_checkpoint_finetune({
                    'curr_iter': curr_iter + args.max_iter + 1,
                    'model': args.model,
                    'gen_state_dict': model['G'].state_dict(),
                    'dis_state_dict': model['D'].state_dict(),
                    'avg_gen_state_dict': avg_gen_net.state_dict(),
                    'gen_optimizer': optimizer['G'].state_dict(),
                    'dis_optimizer': optimizer['D'].state_dict(),
                    'decay_G_state_dict': None,
                    'decay_D_state_dict': None,
                    'best_fid': best_fid,
                    # 'path_helper': args.path_helper,
                    'seed': args.seed,
                }, is_best, args.save_path)

                # also save the last checkpoint
                if curr_iter == int(args.max_iter * finetune_iters_ratio):
                    save_checkpoint_finetune({
                        'curr_iter': curr_iter + args.max_iter + 1,
                        'model': args.model,
                        'gen_state_dict': model['G'].state_dict(),
                        'dis_state_dict': model['D'].state_dict(),
                        'avg_gen_state_dict': avg_gen_net.state_dict(),
                        'gen_optimizer': optimizer['G'].state_dict(),
                        'dis_optimizer': optimizer['D'].state_dict(),
                        'decay_G_state_dict': None,
                        'decay_D_state_dict': None,
                        'best_fid': best_fid,
                        # 'path_helper': args.path_helper,
                        'seed': args.seed,
                    }, is_best=False, output_dir=args.save_path, filename='final.pth')
                del avg_gen_net

    total_training_flops = FC['G'].training_flops + FC['D'].training_flops
    total_dense_training_flops = FC['G'].dense_training_flops + FC['D'].dense_training_flops
    print('Total G FLOPS sparsity:{}={:e}/{:e}'.format(FC['G'].training_flops / FC['G'].dense_training_flops,
                                                       FC['G'].training_flops, FC['G'].dense_training_flops))
    print('Total D FLOPS sparsity:{}={:e}/{:e}'.format(FC['D'].training_flops / FC['D'].dense_training_flops,
                                                       FC['D'].training_flops, FC['D'].dense_training_flops))
    print('Total FLOPS sparsity:{}={:e}/{:e}'.format(total_training_flops / total_dense_training_flops,
                                                     total_training_flops, total_dense_training_flops))

    np.save(os.path.join(args.save_path, 'fid.npy'), fid_record)
    if args.cal_br:
        br_record = np.nan_to_num(np.array(br_record), nan=10000, posinf=10000, neginf=-10000)
        print('Average BR is :{}'.format(np.mean(br_record)))
    plot_FDB(args, fid_record, density_record, br_record, path=args.save_path)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size,), join=True)
