import os
import torch

import random
import dateutil.tz
from datetime import datetime
import time
import logging

import pdb
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from contextlib import contextmanager
import matplotlib.pyplot as plt

def pruning_generate(model,px,method='l1'):
    parameters_to_prune =[]
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            parameters_to_prune.append((m,'weight'))
    parameters_to_prune = tuple(parameters_to_prune)
    if method=='l1':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
    elif method=='random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )

def see_remain_rate(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
def see_remain_rate_orig(model):
    sum_list = 0
    zero_sum = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            sum_list = sum_list+float(m.weight_orig.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight_orig == 0))     
    print('remain weight = ', 100*(1-zero_sum/sum_list),'%')


def rewind_weight(model_dict, target_model_dict_keys):

    new_dict = {}
    for key in target_model_dict_keys:
        if 'mask' not in key:
            if 'orig' in key:
                ori_key = key[:-5]
            else:
                ori_key = key 
            new_dict[key] = model_dict[ori_key]

    return new_dict

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def save_checkpoint_finetune(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_finetune_best.pth'))

def save_checkpoint_imp(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, "checkpoint_{}.pth".format(states['round'])))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_{}_best.pth".format(states['round'])))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y

def plot_FDB(args, fid, density, br, path=''):
    iterations = args.max_iter
    num_fid = len(fid)
    num_density = len(density)
    num_br = len(br)
    
    fid_gap = iterations//num_fid
    density_gap = iterations//num_density
    if num_br:
        br_gap = iterations//num_br

    figure, axis = plt.subplots(3, 1)
    figure.set_size_inches(20, 10.5)

    x1 = np.arange(0, iterations, fid_gap)
    x2 = np.arange(0, iterations, density_gap)
    if num_br:
        x3 = np.arange(0, iterations, br_gap)

    minor_ticks_top = np.linspace(0, 50, 11)

    axis[0].plot(x1, fid)
    axis[0].set_title("FID")
    axis[0].set_ylim([0, 100])
    axis[0].set_yticks(minor_ticks_top)
    axis[0].grid()

    axis[1].plot(x2, density)
    axis[1].set_title("Density of discriminator")
    axis[1].grid()

    if num_br:
        axis[2].plot(x3, br)
        axis[2].set_title("Balance Ratio")
        axis[2].set_ylim([0.0, 1.5])
        axis[2].grid()

    # plt.show()
    plt.savefig(os.path.join(path, 'Stats.png'))


@contextmanager
def module_no_grad(m: torch.nn.Module):
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad_(False)
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])