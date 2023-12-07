import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.utils import make_grid
from imageio import imsave
from copy import deepcopy
import logging
from sparselearning.gan_utils.inception_score import get_inception_score
from sparselearning.gan_utils.fid_score import calculate_fid_given_paths
from sparselearning.gan_utils.losses import consistency_loss
from sparselearning.gan_utils.utils import module_no_grad
from sparselearning.gan_utils.diffAugment_pytorch import *
from sparselearning.gradnorm import normalize_gradient
import random

logger = logging.getLogger(__name__)


def calculate_br(discriminator_loss_queue):
    fake_validity_G_part, fake_validity_D_part, real_validity_D_part = zip(*discriminator_loss_queue)
    fake_validity_G_part = np.array(fake_validity_G_part)
    fake_validity_D_part = np.array(fake_validity_D_part)
    real_validity_D_part = np.array(real_validity_D_part)
    balance_ratio = np.mean(
        (fake_validity_G_part - fake_validity_D_part) / (real_validity_D_part - fake_validity_D_part))
    return balance_ratio


def train(args, model, optimizer, gen_avg_param, train_iter,
          curr_iter, discriminator_loss_queue, br_record, density_record, scheduler=None, mask={}, condition=False,
          FC=None):
    start_time = time.time()
    np.random.seed(args.seed + curr_iter)

    gen_net = model['G']
    dis_net = model['D']
    gen_optimizer = optimizer['G']
    dis_optimizer = optimizer['D']
    policy = "color,translation,cutout"
    DiffAug_seed = random.randint(0x8000000000000000, 0xFFFFFFFFFFFFFFFF)
    if scheduler:
        gen_scheduler, dis_scheduler = scheduler['G'], scheduler['D']

    # gen_net = gen_net.train()
    # dis_net = dis_net.train()

    # Gradient normalization:
    gradient_norm = True if args.d_norm == 'GN' else False

    gen_net.train()
    dis_net.train()

    # -----------------
    #  Train Discriminator
    # -----------------
    (imgs, labels) = next(train_iter)
    imgs = iter(torch.split(imgs, args.dis_batch_size))
    labels = iter(torch.split(labels, args.dis_batch_size))
    # E_signs_batches are for ADA-RDDST
    E_signs_batches = []
    # for _ in range(args.n_critic):
    for d_iter in range(args.n_critic):
        accumulated_loss = 0
        for acml_step in range(args.accumulation_steps):
            # Adversarial ground truths
            real_imgs = next(imgs).type(torch.cuda.FloatTensor)
            if args.diff_aug:
                real_imgs = DiffAugment(real_imgs, seed=DiffAug_seed, policy=policy)
            real_labels = next(labels).type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim)))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            if gradient_norm:
                real_validity = normalize_gradient(net_D=dis_net, x=real_imgs)
            elif condition:
                real_validity = dis_net(real_imgs, y=real_labels.to(torch.int64))
            else:
                real_validity = dis_net(real_imgs)

            if args.cal_br and (curr_iter % args.br_freq == 0):
                if args.da_criterion == 'ADA':
                    E_signs_batches.append(torch.mean((real_validity > 0.0).float()).item())

            with torch.no_grad():
                if condition:
                    # TODO: support multiple datasets with more classes
                    fake_labels = torch.randint(args.num_classes, (real_imgs.shape[0],)).type(torch.cuda.IntTensor)
                    fake_imgs = gen_net(z, y=fake_labels).detach()
                else:
                    fake_imgs = gen_net(z).detach()

                if args.diff_aug:
                    fake_imgs = DiffAugment(fake_imgs, seed=DiffAug_seed, policy=policy)
            assert fake_imgs.size() == real_imgs.size()

            if gradient_norm:
                fake_validity = normalize_gradient(net_D=dis_net, x=fake_imgs)
            elif condition:
                fake_validity = dis_net(fake_imgs, fake_labels)
            else:
                fake_validity = dis_net(fake_imgs)

            # cal loss
            if args.loss == 'hinge':
                d_loss_real = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity))
                d_loss_fake = torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            elif args.loss == 'wass':
                d_loss_real = torch.mean(-real_validity)
                d_loss_fake = torch.mean(fake_validity)
            elif args.loss == 'BCE':
                d_loss_real = nn.BCEWithLogitsLoss()(real_validity, torch.ones_like(real_validity))
                d_loss_fake = nn.BCEWithLogitsLoss()(fake_validity, torch.zeros_like(fake_validity))
            elif args.loss == 'softhinge':
                d_loss_real = torch.mean(F.softplus(1.0 - real_validity))
                d_loss_fake = torch.mean(F.softplus(1 + fake_validity))
            elif args.loss == 'softplus':
                d_loss_real = torch.mean(F.softplus(- real_validity))
                d_loss_fake = torch.mean(F.softplus(fake_validity))
            else:
                raise NotImplementedError

            d_loss = d_loss_real + d_loss_fake

            if args.cr > 0:
                loss_cr = consistency_loss(net_D=dis_net, real=real_imgs, y_real=None, pred_real=real_validity)
                d_loss += args.cr * loss_cr

            accumulated_loss += d_loss

        d_loss = accumulated_loss / args.accumulation_steps
        d_loss.backward()

        if 'D' in mask.keys():

            """
            Mask of discriminator is a class which modified the optimizer
            Thus, mask['D'] has to be called every step to use optimize the weights
            of discriminator
            """
            mask['D'].step(discriminator_loss_queue)
        else:
            dis_optimizer.step()

    # -----------------
    #  Train Generator
    # -----------------
    with module_no_grad(dis_net):
        gen_optimizer.zero_grad()

        accumulated_loss = 0
        for acml_step in range(args.accumulation_steps):
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            if condition:
                gen_labels = torch.randint(args.num_classes, (args.gen_batch_size,)).type(torch.cuda.IntTensor)
                gen_imgs = gen_net(gen_z, y=gen_labels)
            else:
                gen_imgs = gen_net(gen_z)

            if args.diff_aug:
                gen_imgs = DiffAugment(gen_imgs, seed=DiffAug_seed, policy=policy)

            # apply gradient normalization here:
            if gradient_norm:
                fake_validity = normalize_gradient(net_D=dis_net, x=gen_imgs)
            elif condition:
                fake_validity = dis_net(gen_imgs, y=gen_labels)
            else:
                fake_validity = dis_net(gen_imgs)

            if args.loss == 'BCE':
                g_loss = nn.BCEWithLogitsLoss()(fake_validity, torch.ones_like(fake_validity))
            else:
                # cal loss
                g_loss = -torch.mean(fake_validity)

            accumulated_loss += g_loss
        g_loss = accumulated_loss / args.accumulation_steps
        g_loss.backward()

        if 'G' in mask.keys():
            mask['G'].step()

        else:
            gen_optimizer.step()

        # REMOVE THIS
        if args.cal_br and (curr_iter % args.br_freq == 0):
            fake_validity_D_part = torch.mean(fake_validity).item()
        else:
            fake_validity_D_part = 0.0

    if args.cal_br and (curr_iter % args.br_freq == 0):
        # gen_net.eval()
        # dis_net.eval()
        if gradient_norm:
            gen_imgs = gen_net(gen_z)
            if args.diff_aug:
                gen_imgs = DiffAugment(gen_imgs, seed=DiffAug_seed, policy=policy)
            fake_validity = normalize_gradient(net_D=dis_net, x=gen_imgs)
            real_validity = normalize_gradient(net_D=dis_net, x=real_imgs)
        elif condition:
            with torch.no_grad():
                gen_imgs = gen_net(gen_z, gen_labels)
                if args.diff_aug:
                    gen_imgs = DiffAugment(gen_imgs, seed=DiffAug_seed, policy=policy)
                fake_validity = dis_net(gen_imgs, y=gen_labels)
                real_validity = dis_net(real_imgs, y=real_labels.to(torch.int64))
        else:
            with torch.no_grad():
                gen_imgs = gen_net(gen_z)
                if args.diff_aug:
                    gen_imgs = DiffAugment(gen_imgs, seed=DiffAug_seed, policy=policy)
                fake_validity = dis_net(gen_imgs)
                real_validity = dis_net(real_imgs)

        fake_validity_G_part = torch.mean(fake_validity).item()
        real_validity_D_part = torch.mean(real_validity).item()
        # gen_net.train()
        # dis_net.train()

        # collection of 100 loss from discriminator
        if args.da_criterion == 'ADA':
            E_signs = np.mean(E_signs_batches)
            discriminator_loss_queue.append((E_signs,))
        elif args.da_criterion == 'fake':
            discriminator_loss_queue.append((fake_validity_G_part, fake_validity_D_part, real_validity_D_part))
        else:
            raise NotImplementedError
    else:
        fake_validity_G_part = 0.0
        real_validity_D_part = 0.0

        # DST now
    if 'G' in mask.keys():
        mask['G'].dst()
        if mask['G'].mask_change:
            FC['G'].update_iter_and_training_flops(mask['G'].density_dict, mask['G'].prune_every_k_steps)
            print('[*] Ongoing traing G FLOPS:{}={:e}/{:e}, current FLOPS per iter:{:e}'.format(
                FC['G'].training_flops / FC['G'].dense_training_flops,
                FC['G'].training_flops, FC['G'].dense_training_flops, FC['G'].iter_flops))

    if 'D' in mask.keys():
        mask['D'].dst(discriminator_loss_queue)
        density_record.append(mask['D'].density)
        if mask['D'].mask_change:
            FC['D'].update_iter_and_training_flops(mask['D'].density_dict, mask['D'].prune_every_k_steps)
            print('[*] Ongoing traing D FLOPS:{}={:e}/{:e}, current FLOPS per iter:{:e}'.format(
                FC['D'].training_flops / FC['D'].dense_training_flops,
                FC['D'].training_flops, FC['D'].dense_training_flops, FC['D'].iter_flops))
    else:
        density_record.append(1.0)

    if (len(discriminator_loss_queue) > args.da_iters // args.br_freq): discriminator_loss_queue.popleft()

    # moving average weight
    # special handling of SEMA if there is a mask change
    if args.sparse_G and mask['G'].mask_change:
        for (name, p), avg_p in zip(gen_net.named_parameters(), gen_avg_param):
            if name in mask['G'].masks.keys():
                # first clear weight in SEMA that is pruned
                avg_p.mul_((mask['G'].masks)[name])

                # Then set newly grown statistics
                with torch.no_grad():
                    newly_grown = (mask['G'].newly_grown)[name]
                    # Note (p.data * newly_grown).sum() should be zero
                    avg_p.mul_(args.ema).add_(1 - args.ema, p.data)
                    # avg_p.data = p.data * newly_grown + (args.ema * avg_p.data + (1-args.ema) * p.data) * (1.0 - newly_grown)

            else:
                avg_p.mul_(args.ema).add_(1 - args.ema, p.data)
        # mask['G'].mask_change = False

    # Use normal EMA
    else:
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            avg_p.mul_(args.ema).add_(1 - args.ema, p.data)

    # adjust learning rate only after training generator
    if scheduler:
        gen_scheduler.step(curr_iter)
        dis_scheduler.step(curr_iter)
        g_lr = gen_scheduler.get_lr()[0]
        d_lr = dis_scheduler.get_lr()[0]
    else:
        g_lr = args.gen_lr
        d_lr = args.dis_lr

    end_time = time.time()

    # verbose
    if args.cal_br and (curr_iter % args.br_freq == 0):
        if args.da_criterion == 'fake' and len(discriminator_loss_queue) > 0:
            balance_ratio = calculate_br(discriminator_loss_queue)
            if curr_iter and curr_iter % args.log_interval == 0:
                print(
                    "[Iteration %d/%d] [D=real+fake: %f=%f+%f] [G loss: %f] [D lr: %f] [G lr: %f] [DR=%f,DF=%f,GF=%f] [BR: %f] [Time used: %f mins]" %
                    (curr_iter, args.max_iter, d_loss.item(), d_loss_real.item(), d_loss_fake.item(), g_loss.item(),
                     d_lr, g_lr, real_validity_D_part, fake_validity_D_part, fake_validity_G_part, balance_ratio,
                     (end_time - start_time) / 60.0))
            br_record.append(balance_ratio)
        elif args.da_criterion == 'ADA' and len(discriminator_loss_queue) > 0:
            E_signs, = zip(*discriminator_loss_queue)
            E_signs_mean = np.mean(E_signs)
            if curr_iter and curr_iter % args.log_interval == 0:
                print(
                    "[Iteration %d/%d] [D=real+fake: %f=%f+%f] [G loss: %f] [D lr: %f] [G lr: %f] [ADA: %f] [Time used: %f mins]" %
                    (curr_iter, args.max_iter, d_loss.item(), d_loss_real.item(), d_loss_fake.item(), g_loss.item(),
                     d_lr, g_lr, E_signs_mean, (end_time - start_time) / 60.0))
            br_record.append(E_signs_mean)
    else:
        if curr_iter and curr_iter % args.log_interval == 0:
            print(
                "[Iteration %d/%d] [D=real+fake: %f=%f+%f] [G loss: %f] [D lr: %f] [G lr: %f]   [Time used: %f]" %
                (curr_iter, args.max_iter, d_loss.item(), d_loss_real.item(), d_loss_fake.item(), g_loss.item(), d_lr,
                 g_lr,
                 (end_time - start_time) / 60.0))


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, curr_iter, keep_image=False, also_test=False,
             condition=False):
    start_time = time.time()
    np.random.seed(args.seed + curr_iter)
    gen_labels = torch.randint(10, (fixed_z.shape[0],)).type(torch.IntTensor).to('cuda')

    fid_buffer_dir = os.path.join(args.save_path, 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        if condition:
            z_labels = torch.randint(args.num_classes, (z.shape[0],)).type(torch.cuda.IntTensor)
            gen_imgs = gen_net(z, y=z_labels).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                                torch.uint8).numpy()
        else:
            # Generate a batch of images
            gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                    torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, 'iter%d_b%d.png' % (iter_idx, img_idx))
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))
    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    if also_test and hasattr(args, 'fid_stat_test'):
        fid_score_test = calculate_fid_given_paths([fid_buffer_dir, args.fid_stat_test], inception_path=None)

    if not keep_image:
        os.system('rm -r {}'.format(fid_buffer_dir))

    end_time = time.time()

    # writer.add_image('sampled_images', img_grid, global_steps)
    print('Inception_score/mean at curr_iter {}: {}'.format(curr_iter, mean))
    print('Inception_score/std at curr_iter {}: {}'.format(curr_iter, std))
    print('FID_score at curr_iter {}: {}'.format(curr_iter, fid_score))
    if also_test and args.fid_stat_test:
        print('FID_score of test set at curr_iter {}: {}'.format(curr_iter, fid_score_test))
    print('Time used for testing:{} mins.'.format((end_time - start_time) / 60.0))

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def copy_zero_params(model):
    flatten = deepcopy(list(torch.zeros_like(p.data) for p in model.parameters()))
    return flatten
