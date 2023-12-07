from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from sparselearning.gan_functions import copy_params, copy_zero_params

import numpy as np
import math
import time

EPSILON = 1e-08


def add_sparse_args(parser):
    parser.add_argument('--sparse-G', action='store_true', help='Enable sparse mode of generator. Default: False.')
    parser.add_argument('--sparse-D', action='store_true', help='Enable sparse mode of discriminator. Default: False.')
    parser.add_argument('--fix-G', action='store_true',
                        help='Fix sparse connectivity during training of generator. Default: False.')
    parser.add_argument('--fix-D', action='store_true',
                        help='Fix sparse connectivity during training of discriminator. Default: False.')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--growth-G', type=str,
                        choices=['random', 'global_random', 'momentum', 'momentum_neuron', 'gradient',
                                 'global_gradient', 'global_hybrid'],
                        default='random',
                        help='Growth mode of generator. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--growth-D', type=str,
                        choices=['random', 'global_random', 'momentum', 'momentum_neuron', 'gradient',
                                 'global_gradient', 'global_hybrid', 'global_resrand', 'global_resgrad'],
                        default='global_random',
                        help='Growth mode of discriminator. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death-G', type=str,
                        choices=['magnitude', 'SET', 'Taylor_FO', 'threshold', 'global_magnitude'],
                        default='global_magnitude',
                        help='Death mode / pruning mode of generator. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--death-D', type=str,
                        choices=['magnitude', 'SET', 'Taylor_FO', 'threshold', 'global_magnitude'],
                        default='global_magnitude',
                        help='Death mode / pruning mode of discriminator. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none',
                        help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate-G', type=float, default=0.50, help='The pruning rate / death rate of generator.')
    parser.add_argument('--death-rate-D', type=float, default=0.05,
                        help='The pruning / death / dynamic adjust rate of discriminator.')
    parser.add_argument('--death-rate-D-dd-b', type=float, default=0.5,
                        help='Extra pruning / death / dynamic adjust rate of the discriminator for the second half (DST part) of dd-b; only used when dd-b is active.')
    parser.add_argument('--death-rate-schedule-G', type=str, default='cosine', choices=['cosine', 'linear', 'constant'],
                        help='How to adjust death rate throughout training for generator.')
    parser.add_argument('--death-rate-schedule-D', type=str, default='cosine', choices=['cosine', 'linear', 'constant'],
                        help='How to adjust death rate throughout training for discriminator.')
    parser.add_argument('--density-G', type=float, default=1.0, help='The density of the overall sparse generator.')
    parser.add_argument('--density-D', type=float, default=1.0, help='The density of the overall sparse discriminator.')
    parser.add_argument('--update_frequency_G', type=int, default=500,
                        help='how many iterations to train between G parameter exploration')
    parser.add_argument('--update_frequency_D', type=int, default=500,
                        help='how many iterations to train between D param exploration or dynamic adjust. ')
    parser.add_argument('--decay-schedule', type=str, default='cosine',
                        help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    # parser.add_argument('--dynamic-adjust', action='store_true',
    #                     help='Enable dynamic sparsity adjustment of discriminator')
    parser.add_argument('--adjust-mode', type=str, choices=['linear_adjust', 'dynamic_adjust', 'none'], default='none',
                        help='Enable dynamic sparsity adjustment of discriminator')
    parser.add_argument('--dd-b', action='store_true', help='Double DST with BR.')
    parser.add_argument('--dd-bv', type=float, default=0.5, help='Fraction of [DA,DST].')
    parser.add_argument('--da-ub', type=float, default=0.5, help='Dynamic adjust upper bound.')
    parser.add_argument('--da-lb', type=float, default=0.2, help='Dynamic adjust lower bound.')
    parser.add_argument('--dynamic-bound', action='store_true',
                        help='Enable dynamic adjustment of upper and lower bounds. Default: False.')
    parser.add_argument('--da-disb', type=float, default=0.5, help='Dynamic adjust discriminator bound.')
    parser.add_argument('--da-iters', type=int, default=100,
                        help='Number of iterations for computing dynamic adjust losses.')
    parser.add_argument('--hybrid-alpha', type=float, default=0.5,
                        help='Ratio of gradient-based growth in hybrid mode.')
    parser.add_argument('--da-criterion', type=str, default='fake', choices=['ADA', 'fake'],
                        help='Use which loss to dynamic adjust.')

    # balance ratio frequency
    parser.add_argument('--cal_br', action='store_true', help='Calculate Balance ratio. Default: False.')
    parser.add_argument('--br_freq', type=int, default=100, help='Frequency of computing BR.')

    # FOR weight resurrection
    parser.add_argument('--resu-D', action='store_true',
                        help='Enable weight resurrect during grow for discriminator. Default: False.')
    parser.add_argument('--resu-G', action='store_true',
                        help='Enable weight resurrect during grow for generator. Default: False.')

    # FOR post-hoc pruning
    parser.add_argument('--posthoc', action='store_true', help='Enable post-hoc pruning. Default: False.')


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.T_max = T_max
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class LinearDecay(object):
    def __init__(self, death_rate, T_max):
        self.T_max = T_max
        self.steps = 0
        self.death_rate = death_rate

    def step(self):
        self.steps += 1

    def get_dr(self):
        return (1.0 - 1.0 * self.steps / self.T_max) * self.death_rate


class ConstantRate(object):
    def __init__(self, death_rate, T_max):
        self.T_max = T_max
        self.steps = 0
        self.death_rate = death_rate

    def step(self):
        self.steps += 1

    def get_dr(self):
        return self.death_rate


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude',
                 growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, fix=True, adjust_mode='none',
                 da_bound=(0.5, 0.5, 0.5),
                 update_frequency=500, hybrid_alpha=0.5, obj_name='default_name', args=None, dynamic_bound=False,
                 resurrect=False, resu_decay=0.999, doub_dst=False):
        growth_modes = ['random', 'global_random', 'momentum', 'momentum_neuron', 'gradient', 'global_gradient',
                        'global_hybrid']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.obj_name = obj_name
        self.args = args
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.da_bound = da_bound
        self.da_bound_origin = da_bound
        self.dynamic_bound = dynamic_bound
        self.doub_dst = doub_dst

        self.masks = {}
        self.newly_grown = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer
        self.fix = fix
        self.adjust_mode = adjust_mode
        self.hybrid_alpha = hybrid_alpha
        self.mask_change = False

        # stats
        self.name2zeros = {}
        self.num_remove = {}

        # save_temp is for temperary use: for example global_random_growth
        self.save_temp = None
        self.name2nonzeros = {}
        self.density = None
        self.density_dict = {}
        # self.nonzeros_total = 0
        self.global_threshold = None
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.total_params = None
        self.steps = 0
        self.resurrect = resurrect
        self.resu_decay = resu_decay

        # if fix, then we do not explore the sparse connectivity
        if self.fix:
            assert self.adjust_mode == 'none', 'Please disable dynamic adjust for fixing the discriminator.'
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = update_frequency

        if self.resurrect:
            self.resurrect_weights = {}

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.density = density
        if mode == 'GMP':
            self.baseline_nonzero = 0
            self.total_params = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
                    self.total_params += self.masks[name].numel()

        elif mode == 'lottery_ticket':
            print('initialize by lottery ticket')
            self.baseline_nonzero = 0
            self.total_params = 0
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * self.density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
                    self.total_params += self.masks[name].numel()
                    self.total_params += self.masks[name].numel()

        elif mode == 'uniform':
            self.baseline_nonzero = 0
            self.total_params = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()  # lsw
                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
                    self.total_params += self.masks[name].numel()

        elif mode == 'ERK':
            print('initialize by ERK')
            self.total_params = 0
            for name, weight in self.masks.items():
                self.total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - self.density)
                    n_ones = n_param * self.density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            self.baseline_nonzero = 0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()
            print(f"Overall sparsity {self.baseline_nonzero / self.total_params}")

        # assert abs(self.total_params*self.density - self.baseline_nonzero) < 20, \
        # "Mask density mismatch! target:{}, actual:{}".format(self.total_params*self.density, self.baseline_nonzero)

        # Save for SEMA
        for name, mask in self.masks.items():
            self.newly_grown[name] = mask.detach().clone()

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks)  # used for ITOP
        # self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()
            self.density_dict[name] = (weight != 0).detach().cpu().numpy().mean()

        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))

    def step(self, discriminator_loss_queue=None):
        self.optimizer.step()
        self.apply_mask()
        # this part handles the constant death rate switch 
        # for the strict DDST (SDDST)
        if isinstance(self.death_rate_decay, list):
            self.death_rate_decay[0].step()
            self.death_rate_decay[1].step()
            if self.adjust_mode == 'dynamic_adjust':
                self.death_rate = self.death_rate_decay[0].get_dr()
            else:
                self.death_rate = self.death_rate_decay[1].get_dr()

        elif self.death_rate_decay:
            self.death_rate_decay.step()
            self.death_rate = self.death_rate_decay.get_dr()

        self.steps += 1
        if self.resurrect:
            self.update_resurrect_weight()

    def adjust_bound(self, init_low=0.4, init_up=0.65, step=50000.0):
        print('check', self.da_bound, init_low, self.da_bound_origin[0], (self.da_bound_origin[0] - init_low))
        self.da_bound = list(self.da_bound)
        self.da_bound[0] = min(init_low + (self.da_bound_origin[0] - init_low) / step, self.da_bound_origin[0])
        self.da_bound[1] = max(init_up - (init_up - self.da_bound_origin[1]) / step, self.da_bound_origin[1])
        self.da_bound = tuple(self.da_bound)

    def dst(self, discriminator_loss_queue=None, gen_avg_param_full=None):
        self.mask_change = False
        if self.dynamic_bound: self.adjust_bound()

        if self.prune_every_k_steps is not None:
            # Do not update at the fisrt and last iteration
            T_max = self.death_rate_decay[0].T_max if isinstance(self.death_rate_decay,
                                                                 list) else self.death_rate_decay.T_max
            if (self.steps % self.prune_every_k_steps == 0) and (self.steps != T_max):
                print('-' * 60)
                print('Start to adjust {} with death rate: {}.'.format(self.obj_name, self.death_rate))
                time_start = time.time()
                if self.adjust_mode == 'dynamic_adjust':
                    if len(discriminator_loss_queue) >= self.args.da_iters // self.args.br_freq:
                        print('Use dynamic adjust with da points:{}.'.format(len(discriminator_loss_queue)))
                        self.dynamic_adjust_density(discriminator_loss_queue, da_bound=self.da_bound)
                    else:
                        print('Use dynamic adjust, however do nothing this iteration.')
                    self.apply_mask()
                    print('Current density of discriminator:{}'.format(self.baseline_nonzero / self.total_params))

                elif self.adjust_mode == 'linear_adjust':
                    if len(discriminator_loss_queue) >= self.args.da_iters // self.args.br_freq:
                        print('Use linear adjust.')
                        self.linear_adjust_density(discriminator_loss_queue)
                    else:
                        print('Use linear adjust, however do nothing this iteration.')
                    self.apply_mask()
                    print('Current density of discriminator:{}'.format(self.baseline_nonzero / self.total_params))
                else:
                    print('Use DST.')
                    self.truncate_weights()
                    _, _ = self.fired_masks_update()

                    self.print_nonzero_counts()
                self.mask_change = True

                time_end = time.time()
                print('Dynamic Lower Bounds: {}'.format(self.da_bound[0]))
                print('Dynamic Upper Bounds: {}'.format(self.da_bound[1]))
                print('Adjust done, time used: {}'.format((time_end - time_start) / 60.0))

                # if self.args.sema == 'v2' and self.obj_name == 'generator':
                if self.resurrect:
                    print('[*] Use weight resurrection to initialize new weights for {}.'.format(self.obj_name))
                    # assert gen_avg_param_full is not None, 'gen_avg_param_full is None.'
                    self.sema_resurrect()

    # store averaged weights so that it can be used to perform weight resurrection
    def update_resurrect_weight(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                with torch.no_grad():
                    # self.resurrect_weights[name].mul_(self.resu_decay).add_(1-self.resu_decay, weight.data)
                    # resurrect_weight.mul_(self.resu_decay).add_(1-self.resu_decay, weight.data)
                    curr_mask = self.masks[name].data.byte()
                    self.resurrect_weights[name].data = self.resurrect_weights[name].data * (
                                1 - curr_mask) + weight.data

    def sema_resurrect(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                with torch.no_grad():
                    newly_grown = (self.newly_grown)[name]
                    # print((resurrect_weight * newly_grown).sum().item())
                    weight.data += self.resurrect_weights[name] * newly_grown

    def linear_adjust_density(self, grow='random', death='global_magnitude'):
        # Only increase density when current density is smaller than 1.0
        if self.density < 1:
            target_sparsity = self.args.density_D + self.steps / self.death_rate_decay.T_max * (1 - self.args.density_D)
            target_sparsity = max(min(1.0, target_sparsity), 0.0)
            self.density = target_sparsity

            self.save_temp = None
            new_baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()
                    # use random_growth for code review, adding more methods/mappings later
                    new_mask = self.global_random_growth(name, new_mask, weight, use_density=True)
                    new_baseline_nonzero += (new_mask != 0).sum().int().item()
                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()

            self.baseline_nonzero = new_baseline_nonzero
            self.save_temp = None
            self.global_threshold = None

        print('*' * 60)
        print('Linear adjust done, final density: {0}/{1} = {2}'.format(self.baseline_nonzero, self.total_params,
                                                                        self.density))
        print('*' * 60)

    def dynamic_adjust_increase_density(self):
        # Increase p% density through self.density
        # target_sparsity = max(min(1.0, self.density * (1.0+self.death_rate)), 0.0)
        target_sparsity = max(min(1.0, self.density + self.death_rate), 0.05)
        if self.doub_dst:
            target_sparsity = min(target_sparsity, 0.5)

        if target_sparsity == self.density:
            print('Limit density, no adjust.')
            return

        print('Increase from {} to target density with {}: {}'.format(self.density, self.growth_mode, target_sparsity))
        self.density = target_sparsity
        new_baseline_nonzero = 0
        self.save_temp = None
        self.global_threshold = None
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()
                old_mask = self.masks[name].data.byte()
                assert self.obj_name == 'discriminator', 'Only use dynmaic adjust for discriminator'
                # use random_growth for code review, adding more methods/mappings later
                if self.growth_mode == 'global_random':
                    new_mask = self.global_random_growth(name=name, new_mask=new_mask, weight=weight, use_density=True)
                elif self.growth_mode == 'global_gradient':
                    new_mask = self.global_gradient_growth(name=name, new_mask=new_mask, weight=weight,
                                                           use_density=True)
                elif self.growth_mode == 'global_resrand':
                    new_mask = self.global_resrand_growth(name=name, new_mask=new_mask, weight=weight,
                                                          use_density=True)
                elif self.growth_mode == 'global_resgrad':
                    new_mask = self.global_resgrad_growth(name=name, new_mask=new_mask, weight=weight,
                                                          use_density=True)
                elif self.growth_mode == 'global_hybrid':
                    new_mask = self.global_hybrid_growth(name=name, new_mask=new_mask, weight=weight,
                                                         use_density=True)
                else:
                    raise NotImplementedError
                new_baseline_nonzero += (new_mask != 0).sum().int().item()
                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                self.density_dict[name] = new_mask.detach().cpu().numpy().mean()
                # need it ?
                self.newly_grown[name] = ((new_mask - old_mask) == 1).detach().clone().float()
        if abs(new_baseline_nonzero - int(self.total_params * self.density)) > 20:
            print('Slight density mismatch after dynamic increase, goal: {}, actual:{}.'.format(
                int(self.total_params * self.density), \
                new_baseline_nonzero))
        self.baseline_nonzero = new_baseline_nonzero
        self.save_temp = None
        self.global_threshold = None

    def dynamic_adjust_decrease_density(self):
        # Decrease p% density through self.density
        # target_sparsity = min(max(0.0, self.density / (1.0+self.death_rate)), 1.0)
        target_sparsity = min(max(0.05, self.density - self.death_rate), 1.0)
        if target_sparsity == self.density:
            print('Limit density, no adjust.')
            return
        print(
            'Decrease from {} to target density with global magnitude_death: {}'.format(self.density, target_sparsity))
        self.density = target_sparsity
        self.save_temp = None
        total_num_to_prune = self.baseline_nonzero - target_sparsity * self.total_params
        new_baseline_nonzero = 0
        self.global_threshold = None
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()
                # use random_growth for code review, adding more methods/mappings later
                new_mask = self.global_magnitude_death(mask=new_mask, weight=weight, name=name,
                                                       total_num_to_prune=total_num_to_prune)
                new_baseline_nonzero += (new_mask != 0).sum().int().item()
                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                self.density_dict[name] = new_mask.detach().cpu().numpy().mean()
                # need it ?
                self.newly_grown[name] = torch.zeros_like(new_mask)

        if abs(new_baseline_nonzero - int(self.total_params * self.density)) > 20:
            print('Slight density mismatch after dynamic decrease, goal: {}, actual:{}.'.format(
                int(self.total_params * self.density), \
                new_baseline_nonzero))
        self.baseline_nonzero = new_baseline_nonzero
        self.save_temp = None
        self.global_threshold = None

    def dynamic_adjust_density(self, loss_queue, da_bound=(0.5, 0.5, 0.5), grow='random', death='global_magnitude'):

        self.check_density()
        # da_bound: (da_lb [pos], da_ub [pos])
        # increase density of discriminator if too weak
        if self.args.da_criterion == 'ADA':
            assert len(loss_queue[0]) == 1
            E_signs, = zip(*loss_queue)
            E_signs_mean = np.mean(E_signs)

            # Note here, a powerful discriminator would leads to larger E_signs
            # since it can distinguish real images much more easily
            # So the operations are different from 'fake' case
            density_check = 0.5 if self.doub_dst else 1.0
            # This means the discriminator is too weak (cant classify images correctly -> less overfitting), so we increase the density
            if (self.density < density_check) and E_signs_mean <= da_bound[0]:
                self.dynamic_adjust_increase_density()

            # this means the discriminator is too strong (classify most images correctly -> overfitting), so we decrease the density
            elif (self.density > 0.0) and E_signs_mean >= da_bound[1]:
                self.dynamic_adjust_decrease_density()

            print('*' * 60)
            print('Mean {} loss over {} iters: {}'.format(self.args.da_criterion, self.args.da_iters, E_signs_mean))

        elif self.args.da_criterion == 'fake':
            assert len(loss_queue[0]) == 3
            balanced_line = 0.0
            fake_validity_G_part, fake_validity_D_part, real_validity_D_part = zip(*loss_queue)
            fake_validity_G_part = np.array(fake_validity_G_part)
            fake_validity_D_part = np.array(fake_validity_D_part)
            real_validity_D_part = np.array(real_validity_D_part)
            # make sure that discriminator distinguishes fake and real images to provide useful information
            balance_ratio = np.mean(
                (fake_validity_G_part - fake_validity_D_part) / (real_validity_D_part - fake_validity_D_part))
            dis_capacity = np.mean(real_validity_D_part - fake_validity_D_part)
            density_check = 0.5 if self.doub_dst else 1.0

            # need to increase discriminator density
            if balance_ratio >= da_bound[1]:
                # we can still increase density of the discirminator
                if (self.density < density_check):
                    self.dynamic_adjust_increase_density()

                # if we reach maximum density limit when using strict double dst
                # then we just perform DST
                elif self.density == density_check and self.doub_dst:
                    # get the death_rate for DST
                    self.death_rate = self.death_rate_decay[1].get_dr()
                    print('Use DST for Strict ddst when density == density_check.')
                    self.truncate_weights()
                    _, _ = self.fired_masks_update()

                    self.print_nonzero_counts()
                # other than that, do thing
                else:
                    print('Want to increase density, but do nothing')
                    print('Current density:{}, Using doub_dst: {}'.format(self.density, self.doub_dst))

            # make sure that discriminator and generator is balanced
            elif (self.density > 0.0) and balance_ratio <= da_bound[0]:
                self.dynamic_adjust_decrease_density()
            print('*' * 60)
            print('Mean {} loss over {} iters: [fake-G] {}, [fake-D] {}, [real-D] {}, [B-Ratio] {}, [Dr-Df] {}'.format(
                self.args.da_criterion, self.args.da_iters, np.mean(fake_validity_G_part),
                np.mean(fake_validity_D_part), np.mean(real_validity_D_part), balance_ratio, dis_capacity))
        else:
            raise NotImplementedError

        print('Dynamic adjust done, final density: {0}/{1} = {2}'.format(self.baseline_nonzero, self.total_params,
                                                                         self.density))
        print('*' * 60)

    def add_module(self, module, density, sparse_init='ER'):
        self.density = density
        self.modules.append(module)

        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            if self.resurrect:
                self.resurrect_weights[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False)

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing gains in the generator...')
        self.remove_weight_partial_name('gain')
        print('Removing classifier layer...')
        self.remove_weight_partial_name('classifier')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        print('Removing embedding layers...')
        self.remove_type(nn.Embedding)
        self.init(mode=sparse_init, density=density)

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        elif name + '.weight_orig' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight_orig'].shape,
                                                                      self.masks[name + '.weight_orig'].numel()))
            self.masks.pop(name + '.weight_orig')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                               np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]
                    # reset momentum
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor][
                                                                              'momentum_buffer'] * self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((
                                        curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density,
                                                                                            sparse_size / total_size,
                                                                                            epoch))

    def truncate_weights(self):
        # For computing newly-grown masks
        old_masks = copy.deepcopy(self.masks)

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask = self.magnitude_death(mask=mask, weight=weight, name=name)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask=mask, weight=weight, name=name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask=mask, weight=weight, name=name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask=mask, weight=weight, name=name)
                elif self.death_mode == 'global_magnitude':
                    new_mask = self.global_magnitude_death(mask=mask, weight=weight, name=name)
                else:
                    raise NotImplementedError

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask

        # Reset threshold
        self.global_threshold = None
        self.save_temp = None

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name=name, new_mask=new_mask, weight=weight)

                elif self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name=name, new_mask=new_mask, weight=weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name=name, new_mask=new_mask, weight=weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name=name, new_mask=new_mask, weight=weight)

                elif self.growth_mode == 'global_gradient':
                    new_mask = self.global_gradient_growth(name=name, new_mask=new_mask, weight=weight)

                elif self.growth_mode == 'global_random':
                    new_mask = self.global_random_growth(name=name, new_mask=new_mask, weight=weight)

                elif self.growth_mode == 'global_hybrid':
                    new_mask = self.global_hybrid_growth(name=name, new_mask=new_mask, weight=weight)

                else:
                    raise NotImplementedError

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                self.newly_grown[name] = ((new_mask - old_masks[name]) == 1).detach().clone().float()

        # Reset threshold
        self.global_threshold = None
        self.save_temp = None

        self.apply_mask()
        del old_masks

    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return (torch.abs(weight.data) > threshold)

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k - 1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k - 1].item()

        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

        new_mask = pos_mask | neg_mask
        return new_mask

    def global_magnitude_death(self, mask, weight, name, total_num_to_prune=None):
        if total_num_to_prune is None:
            total_num_to_prune = self.baseline_nonzero * self.death_rate

        # If no global threshold exists, then calculate global threshold
        if self.global_threshold is None:
            weight_abs = []
            for module_temp in self.modules:
                for name_temp, weight_temp in module_temp.named_parameters():
                    if name_temp not in self.masks: continue
                    weight_abs.append(torch.abs(weight_temp.data) * self.masks[name_temp])

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = self.baseline_nonzero - int(total_num_to_prune)
            # print('Death rate: {}'.format(self.death_rate))
            # print('Numbers of total: {}'.format(self.nonzeros_total))
            # print('Numbers to keep: {}'.format(num_params_to_keep))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            # TO AVOID USE 0 AS threshold
            if len(threshold) >= 1 and threshold[-1] == 0:
                self.global_threshold = EPSILON
            else:
                self.global_threshold = threshold[-1]
            print('Numbers to death: {}, threshold: {}.'.format(self.baseline_nonzero - num_params_to_keep,
                                                                self.global_threshold))

        return ((torch.abs(weight.data) * mask.float()) >= self.global_threshold).detach().clone()

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0:
            # self.newly_grown[name] = torch.zeros_like(new_mask.data)
            return new_mask
        num_nonfired_weights = (self.fired_masks[name] == 0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0

            # save for SEMA
            # newly_grown = torch.zeros_like(new_mask.data)
            # newly_grown.data.view(-1)[idx[indices]] = 1.0
            # self.newly_grown[name] = newly_grown.detach().clone().float()
        else:
            new_mask[self.fired_masks[name] == 0] = 1.0
            n = (new_mask == 0).sum().item()
            # expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            # new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability

            # Use strict version of growth
            extra_num_growth = (total_regrowth - num_nonfired_weights)
            new_weights = torch.zeros_like(new_mask)
            random_number = torch.rand(new_mask.shape).to(new_mask.device)
            y, idx = torch.sort((random_number * (new_mask == 0).float()).flatten(), descending=True)
            new_weights.data.view(-1)[idx[:total_regrowth]] = 1.0

            new_mask = new_mask.byte() | new_weights
            # self.newly_grown[name] = new_weights.detach().clone().float()
        return new_mask

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0:
            # self.newly_grown[name] = torch.zeros_like(new_mask.data)
            return new_mask
        # expeced_growth_probability = (total_regrowth/n)
        # new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability

        # Use a strict version of growth
        new_weights = torch.zeros_like(new_mask)
        random_number = torch.rand(new_mask.shape).to(new_mask.device)
        y, idx = torch.sort((random_number * (new_mask == 0).float()).flatten(), descending=True)
        new_weights.data.view(-1)[idx[:total_regrowth]] = 1.0

        new_mask_ = new_mask.byte() | new_weights
        # For SEMA
        # self.newly_grown[name] = (new_mask_ - new_mask).detach().clone().float()
        if (new_mask_ != 0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    # TODO: remove random_number
    def global_random_growth(self, name, new_mask, weight, use_density=False):
        '''
         A grow method that grows new weights globally
         If use_density is set to True, the final non-zero entries will be
         int(self.total_params*self.density)
         otherwise, the final non-zero entries will be
         self.baseline_nonzero
        '''

        if self.global_threshold is None:
            self.save_temp = {}
            # Count how many weights are active
            # params_total = 0
            params_remain = 0
            random_number = []
            for module_temp in self.modules:
                for name_temp, weight_temp in module_temp.named_parameters():
                    if name_temp not in self.masks: continue
                    random_variable = (torch.rand_like(weight_temp) + 0.1) * (
                            self.masks[name_temp].data.byte() == 0).float()
                    self.save_temp[name_temp] = random_variable
                    random_number.append(random_variable)
                    params_remain += (self.masks[name_temp] != 0).sum().item()
                    # params_total += self.masks[name_temp].numel()

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in random_number])
            if use_density:
                target_num_params = self.density * self.total_params
                num_params_to_grow = int(target_num_params - params_remain)
            else:
                target_num_params = self.baseline_nonzero
                num_params_to_grow = int(target_num_params - params_remain)
            # print('Death rate: {}'.format(self.death_rate))
            # print('Numbers of total: {}'.format(self.nonzeros_total))
            print('Numbers to grow: {} = {} - {}'.format(num_params_to_grow, target_num_params, params_remain))

            threshold, _ = torch.topk(all_scores, num_params_to_grow, sorted=True)
            self.global_threshold = threshold[-1]

        random_variable_layer = self.save_temp[name]

        # y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        # new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
        new_weights = random_variable_layer >= self.global_threshold
        new_mask = (new_mask.byte() | new_weights).detach().clone()

        return new_mask

    def global_hybrid_growth(self, name, new_mask, weight, use_density=False):
        '''
         A grow method that grows new weights globally with gradient and random
         alpha controls how much portion grow by gradient
        '''

        if self.global_threshold is None:

            # global_threshold is a list containing (threshold_for_grad, threshold_for_rand)
            self.global_threshold = []

            # use save_temp to store temporary masks and random variables
            self.save_temp = {
                'temp_mask': {},
                'rand_number': {},
            }

            # First grow based on gradient
            grad_abs = []
            params_remain = 0
            for module_temp in self.modules:
                for name_temp, weight_temp in module_temp.named_parameters():
                    if name_temp not in self.masks: continue
                    grad_abs.append(self.get_gradient_for_weights(weight_temp).abs() * (
                            self.masks[name_temp].data.byte() == 0).float())
                    params_remain += (self.masks[name_temp] != 0).sum().item()

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in grad_abs])

            if use_density:
                target_num_params = self.density * self.total_params
                num_params_to_grow = int(target_num_params - params_remain)
            else:
                target_num_params = self.baseline_nonzero
                num_params_to_grow = int(target_num_params - params_remain)

            # num_params_to_grow = int(self.baseline_nonzero * self.death_rate)
            num_params_to_grow_grad = int(num_params_to_grow * self.hybrid_alpha)
            num_params_to_grow_rand = num_params_to_grow - num_params_to_grow_grad

            print('Numbers to grow: {} = {} grad + {} random grow.'.format(num_params_to_grow, num_params_to_grow_grad,
                                                                           num_params_to_grow_rand))

            if num_params_to_grow_grad:
                threshold, _ = torch.topk(all_scores, num_params_to_grow_grad, sorted=True)
                self.global_threshold.append(threshold[-1])

                # Start to grow randomly
                # random_number = []
                for module_temp in self.modules:
                    for name_temp, weight_temp in module_temp.named_parameters():
                        if name_temp not in self.masks: continue
                        # Update temperary mask
                        grad = self.get_gradient_for_weights(weight_temp)
                        grad = grad * (self.masks[name_temp] == 0).float()

                        # y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                        # new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
                        new_weights = torch.abs(grad) >= self.global_threshold[0]
                        new_mask = (self.masks[name_temp].byte() | new_weights).detach().clone()
                        self.save_temp['temp_mask'][name_temp] = new_mask

                        random_variable = (torch.rand_like(weight_temp) + 0.1) * (
                                self.save_temp['temp_mask'][name_temp].data.byte() == 0).float()
                        self.save_temp['rand_number'][name_temp] = random_variable
                        # random_number.append(random_variable)
                        # params_remain += (self.masks[name_temp]!= 0).sum().item()
                        # params_total += self.masks[name_temp].numel()
            else:
                self.global_threshold.append(None)
                for module_temp in self.modules:
                    for name_temp, weight_temp in module_temp.named_parameters():
                        if name_temp not in self.masks: continue
                        self.save_temp['temp_mask'][name_temp] = self.masks[name_temp].byte().detach().clone()
                        random_variable = (torch.rand_like(weight_temp) + 0.1) * (
                                self.save_temp['temp_mask'][name_temp].data.byte() == 0).float()
                        self.save_temp['rand_number'][name_temp] = random_variable

            if num_params_to_grow_rand:
                # Gather all scores in a single vector and normalise
                all_scores = torch.cat([torch.flatten(x) for x in (self.save_temp['rand_number']).values()])
                # print('Death rate: {}'.format(self.death_rate))
                # print('Numbers of total: {}'.format(self.nonzeros_total))

                threshold, _ = torch.topk(all_scores, num_params_to_grow_rand, sorted=True)
                self.global_threshold.append(threshold[-1])
            else:
                self.global_threshold.append(None)

        assert len(self.global_threshold) == 2, 'A bug occurs during hybrid grow.'

        # we do need to grow randomly
        if self.global_threshold[-1]:
            random_variable_layer = self.save_temp['rand_number'][name]

            # y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
            # new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
            new_weights = random_variable_layer >= self.global_threshold[-1]
            new_mask = (self.save_temp['temp_mask'][name].byte() | new_weights).detach().clone()
        # no need to grow randomly
        else:
            new_mask = self.save_temp['temp_mask'][name].byte().detach().clone()

        return new_mask

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad * (new_mask == 0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        # For SEMA
        # newly_grown = torch.zeros_like(new_mask.data)
        # newly_grown.data.view(-1)[idx[:total_regrowth]] = 1.0
        # self.newly_grown[name] = newly_grown.detach().clone().float()

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        # For SEMA
        # newly_grown = torch.zeros_like(new_mask.data)
        # newly_grown.data.view(-1)[idx[:total_regrowth]] = 1.0
        # self.newly_grown[name] = newly_grown.detach().clone().float()

        return new_mask

    def global_gradient_growth(self, name, new_mask, weight, use_density=False):

        if self.global_threshold is None:
            grad_abs = []
            params_remain = 0
            for module_temp in self.modules:
                for name_temp, weight_temp in module_temp.named_parameters():
                    if name_temp not in self.masks: continue
                    grad_abs.append(self.get_gradient_for_weights(weight_temp).abs() * (
                            self.masks[name_temp].data.byte() == 0).float())
                    params_remain += (self.masks[name_temp] != 0).sum().item()

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in grad_abs])

            if use_density:
                target_num_params = self.density * self.total_params
                num_params_to_grow = int(target_num_params - params_remain)
            else:
                target_num_params = self.baseline_nonzero
                num_params_to_grow = int(target_num_params - params_remain)

            # num_params_to_grow = int(self.baseline_nonzero * self.death_rate)
            # print('Death rate: {}'.format(self.death_rate))
            # print('Numbers of total: {}'.format(self.nonzeros_total))
            print('Numbers to grow: {} = {} - {}'.format(num_params_to_grow, target_num_params, params_remain))

            threshold, _ = torch.topk(all_scores, num_params_to_grow, sorted=True)
            self.global_threshold = threshold[-1]

        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        # y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        # new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
        new_weights = torch.abs(grad) >= self.global_threshold
        new_mask = (new_mask.byte() | new_weights).detach().clone()

        return new_mask

    def global_resrand_growth(self, name, new_mask, weight, use_density=True):
        # Can only be used when using resurrect
        assert self.resurrect
        assert use_density

        if self.global_threshold is None:
            # first count how many we need to grow using the resurrect_weights
            self.save_temp = {}
            res_abs = []
            params_remain = 0
            total_nonzero_res_weights = 0
            for module_temp in self.modules:
                for name_temp, weight_temp in module_temp.named_parameters():
                    if name_temp not in self.masks: continue
                    res_abs.append(self.resurrect_weights[name_temp].abs() * (
                            self.masks[name_temp].data.byte() == 0).float())
                    self.save_temp[name_temp] = (torch.rand_like(weight_temp) + 0.1) * (
                            self.masks[name_temp].data.byte() == 0).float()
                    total_nonzero_res_weights += int(((self.resurrect_weights[name_temp] != 0) * (
                                self.masks[name_temp].data.byte() == 0).float()).sum().item())
                    params_remain += (self.masks[name_temp] != 0).sum().item()

            target_num_params = self.density * self.total_params
            num_params_to_grow = int(target_num_params - params_remain)

            print('Num params to grow:{}, total_nonzero_res_weights: {}'.format(num_params_to_grow,
                                                                                total_nonzero_res_weights))
            if num_params_to_grow > total_nonzero_res_weights:
                # then grow all of them
                threshold_res = None
                rand_all_scores = torch.cat([torch.flatten(x) for x in self.save_temp.values()])
                threshold_rand, _ = torch.topk(rand_all_scores, num_params_to_grow - total_nonzero_res_weights,
                                               sorted=True)
                threshold_rand = threshold_rand[-1]
                self.global_threshold = (threshold_res, threshold_rand)
                print('Threshold:')
                print(self.global_threshold)
                print('Numbers to grow: {} = {} - {} = {} res + {} rand'.format(num_params_to_grow, target_num_params,
                                                                                params_remain,
                                                                                total_nonzero_res_weights,
                                                                                num_params_to_grow - total_nonzero_res_weights))
            else:
                res_all_scores = torch.cat([torch.flatten(x) for x in res_abs])
                threshold_res, _ = torch.topk(res_all_scores, num_params_to_grow, sorted=True)
                threshold_res = threshold_res[-1]
                threshold_rand = np.inf
                self.global_threshold = (threshold_res, threshold_rand)
                print('Threshold:')
                print(self.global_threshold)
                print('Numbers to grow with all res_weights: {} = {} - {}'.format(num_params_to_grow, target_num_params,
                                                                                  params_remain))
            print('*' * 30)

        threshold_res, threshold_rand = self.global_threshold

        if threshold_res is not None:
            # only grows using res
            new_weights_due_to_res = (self.resurrect_weights[name].abs() * (self.masks[name].data.byte() == 0).float()) \
                                     >= threshold_res
            new_mask = (new_mask.byte() | new_weights_due_to_res).detach().clone()
        else:
            # no threshold for res
            new_weights_due_to_res = (self.resurrect_weights[name] != 0) * (self.masks[name].data.byte() == 0).byte()
            new_weights_due_to_rand = self.save_temp[name] >= threshold_rand
            new_mask = (new_mask.byte() | new_weights_due_to_res | new_weights_due_to_rand).detach().clone()

        return new_mask

    def global_resgrad_growth(self, name, new_mask, weight, use_density=True):
        # Can only be used when using resurrect
        assert self.resurrect
        assert use_density

        if self.global_threshold is None:
            # first count how many we need to grow using the resurrect_weights
            res_abs = []
            grad_abs = []
            params_remain = 0
            total_nonzero_res_weights = 0
            for module_temp in self.modules:
                for name_temp, weight_temp in module_temp.named_parameters():
                    if name_temp not in self.masks: continue
                    res_abs.append(self.resurrect_weights[name_temp].abs() * (
                            self.masks[name_temp].data.byte() == 0).float())
                    grad_abs.append(self.get_gradient_for_weights(weight_temp).abs() * (
                            self.masks[name_temp].data.byte() == 0).float())
                    total_nonzero_res_weights += int(((self.resurrect_weights[name_temp] != 0) * (
                                self.masks[name_temp].data.byte() == 0).float()).sum().item())
                    params_remain += (self.masks[name_temp] != 0).sum().item()

            target_num_params = self.density * self.total_params
            num_params_to_grow = int(target_num_params - params_remain)

            print('Num params to grow:{}, total_nonzero_res_weights: {}'.format(num_params_to_grow,
                                                                                total_nonzero_res_weights))
            if num_params_to_grow > total_nonzero_res_weights:
                # then grow all of them
                threshold_res = None
                grad_all_scores = torch.cat([torch.flatten(x) for x in grad_abs])
                threshold_grad, _ = torch.topk(grad_all_scores, num_params_to_grow - total_nonzero_res_weights,
                                               sorted=True)
                threshold_grad = threshold_grad[-1]
                self.global_threshold = (threshold_res, threshold_grad)
                print('Threshold:')
                print(self.global_threshold)
                print('Numbers to grow: {} = {} - {} = {} res + {} grad'.format(num_params_to_grow, target_num_params,
                                                                                params_remain,
                                                                                total_nonzero_res_weights,
                                                                                num_params_to_grow - total_nonzero_res_weights))
            else:
                res_all_scores = torch.cat([torch.flatten(x) for x in res_abs])
                threshold_res, _ = torch.topk(res_all_scores, num_params_to_grow, sorted=True)
                threshold_res = threshold_res[-1]
                threshold_grad = np.inf
                self.global_threshold = (threshold_res, threshold_grad)
                print('Threshold:')
                print(self.global_threshold)
                print('Numbers to grow with all res_weights: {} = {} - {}'.format(num_params_to_grow, target_num_params,
                                                                                  params_remain))
            print('*' * 30)

        threshold_res, threshold_grad = self.global_threshold

        if threshold_res is not None:
            # only grows using res
            new_weights_due_to_res = (self.resurrect_weights[name].abs() * (self.masks[name].data.byte() == 0).float()) \
                                     >= threshold_res
            new_mask = (new_mask.byte() | new_weights_due_to_res).detach().clone()
        else:
            # no threshold for res
            new_weights_due_to_res = (self.resurrect_weights[name] != 0) * (self.masks[name].data.byte() == 0).byte()
            new_weights_due_to_grad = self.get_gradient_for_weights(weight).abs() >= threshold_grad
            new_mask = (new_mask.byte() | new_weights_due_to_res | new_weights_due_to_grad).detach().clone()

        return new_mask

    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2:
            sum_dim = [1]
        elif len(M.shape) == 4:
            sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask == 0).sum(sum_dim)

        # For SEMA
        newly_grown = torch.zeros_like(new_mask.data)

        M = M * (new_mask == 0).float()
        for i, fraction in enumerate(v):
            neuron_regrowth = math.floor(fraction.item() * total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            temp = new_mask[i] | (M[i] > threshold)
            newly_grown[i] = (temp - new_mask[i])
            new_mask[i] = temp

        self.newly_grown[name] = newly_grown.detach().clone().float()

        return new_mask

    '''
                UTILITY
    '''

    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):

        print('-' * 60)
        total_size = 0
        sparse_size = 0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                sparse_size += num_nonzeros
                total_size += mask.numel()
                self.density_dict[name] = num_nonzeros / float(mask.numel())
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                               self.density_dict[name])
                print(val)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}'.format(self.death_rate))
                break

        print(
            'Total parameters under sparsity level of {0}: {1} = {2}/{3}'.format(self.density, sparse_size / total_size,
                                                                                 sparse_size, total_size))
        print('-' * 60)
        self.baseline_nonzero = sparse_size

    def check_density(self):

        print('-' * 60)
        total_size = 0
        sparse_size_w = 0
        sparse_size_m = 0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                sparse_size_w += (tensor != 0).sum().item()
                sparse_size_m += (self.masks[name] != 0).sum().item()
                total_size += tensor.numel()

        print(
            'Density before applying DST: weight {}, mask {}'.format(sparse_size_w / total_size,
                                                                     sparse_size_m / total_size))

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item()) / float(
                    self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights
