import torch.nn as nn
import torch
import numpy as np

def post_hoc_prune(model, args):
            print('Start to do post-hoc pruning.')
            baseline_nonzero = 0
            total_params = 0
            weight_abs = {}
            names = []

            for name, tensor in model.named_parameters():
                names.append(name)
                weight_abs[name] = torch.abs(tensor).detach().clone()


            # Remove bias
            removed = set()
            for name in list(weight_abs.keys()):
                if 'bias' in name:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, weight_abs[name].shape,
                                                                                   np.prod(weight_abs[name].shape)))
                    removed.add(name)
                    weight_abs.pop(name)

            i = 0
            while i < len(names):
                name = names[i]
                if name in removed:
                    names.pop(i)
                else:
                    i += 1

            # Remove BNs
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if name in names:
                        print('Removing {0} of size {1} = {2} parameters.'.format(name, weight_abs[name].shape,
                                                                                  weight_abs[name].numel()))
                        weight_abs.pop(name)
                    elif name + '.weight' in weight_abs:
                        print('Removing {0} of size {1} = {2} parameters.'.format(name, weight_abs[name + '.weight'].shape,
                                                                                  weight_abs[name + '.weight'].numel()))
                        weight_abs.pop(name + '.weight')
                    else:
                        print('ERROR', name)

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs.values()])

            num_params_to_keep = int(len(all_scores) * args.density_G)
            print('Total num params: {}'.format(len(all_scores)))
            print('Remain num params: {}'.format(num_params_to_keep))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            total_params_pre = 0
            remain_params_pre = 0
            total_params = 0
            remain_params = 0

            for name, tensor in model.named_parameters():
                if name in weight_abs:
                    total_params_pre += tensor.data.numel()
                    remain_params_pre += (tensor.data != 0).sum()
                    tensor.data = tensor.data * (torch.abs(tensor.data) >= acceptable_score)
                    total_params += tensor.data.numel()
                    remain_params += (tensor.data != 0).sum()

            print('*'*10)
            print('Initial density: {}'.format(remain_params_pre/total_params_pre))
            print('Final density: {}'.format(remain_params/total_params))
            print('*'*10)


