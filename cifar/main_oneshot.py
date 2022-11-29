from __future__ import print_function

import argparse
import typing

import numpy as np
import os
import random
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import common
import models
from common import LossType, compute_conv_flops
from models.common import SparseGate, Identity
from models.resnet_expand import BasicBlock
from tqdm import tqdm
import copy
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training with Polarization')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='training dataset (default: cifar10)')
parser.add_argument("--loss-type", "-loss", dest="loss",
                    choices=list(LossType.loss_name().keys()), help="the type of loss")
parser.add_argument('--lbd', type=float, default=0.0001,
                    help='scale sparse rate (i.e. lambda in eq.2) (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=1.,
                    help='coefficient of mean term in polarization regularizer. deprecated (default: 1)')
parser.add_argument('--t', type=float, default=1.,
                    help='coefficient of L1 term in polarization regularizer (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--max-epoch', type=int, default=None, metavar='N',
                    help='the max number of epoch, default None')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[],
                    help="the epoch to decay the learning rate (default 0.5, 0.75)")
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, metavar='S', default=666,
                    help='random seed (default: a random int)')
parser.add_argument('--save', type=str, metavar='PATH', required=True,
                    help='path to save prune model')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--gammas', type=float, nargs='+', default=[],
                    help='LR is multiplied by gamma on decay-epoch, number of gammas should be equal to decay-epoch')
parser.add_argument('--bn-init-value', default=0.5, type=float,
                    help='initial value of bn weight (default: 0.5, following NetworkSlimming)')
parser.add_argument('--retrain', type=str, default=None, metavar="PATH",
                    help="Pruned checkpoint for RETRAIN model.")
parser.add_argument('--clamp', default=1.0, type=float,
                    help='Upper bound of the bn scaling factors (only available at Polarization!) (default: 1.0)')
parser.add_argument('--gate', action='store_true', default=False,
                    help='Add an extra scaling factor after the BatchNrom layers.')
parser.add_argument('--backup-path', default=None, type=str, metavar='PATH',
                    help='path to tensorboard log')
parser.add_argument('--backup-freq', default=10, type=float,
                    help='Backup checkpoint frequency')
parser.add_argument('--fix-gate', action='store_true',
                    help='Do not update parameters of SparseGate while training.')
parser.add_argument('--flops-weighted', action='store_true',
                    help='The polarization parameters will be weighted by FLOPs.')
parser.add_argument('--weight-max', type=float, default=None,
                    help='Maximum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--weight-min', type=float, default=None,
                    help='Minimum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--bn-wd', action='store_true',
                    help='Apply weight decay on BatchNorm layers')
parser.add_argument('--target-flops', type=float, default=None,
                    help='Stop when pruned model archive the target FLOPs')
parser.add_argument('--max-backup', type=int, default=None,
                    help='The max number of backup files')
parser.add_argument('--input-mask', action='store_true',
                    help='If use input mask in ResNet models.')
parser.add_argument('--width-multiplier', default=1.0, type=float,
                    help="The width multiplier (only) for ResNet-56 and VGG16-linear. "
                         "Unavailable for other networks. (default 1.0)")
parser.add_argument('--debug', action='store_true',
                    help='Debug mode.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--alphas', type=float, nargs='+', default=[],
                    help='Multiplier of each subnet')
parser.add_argument('--split_running_stat', action='store_true',
                    help='use split running mean/var for different subnets')
parser.add_argument('--load_running_stat', action='store_true',
                    help='load running mean/var for different subnets')
parser.add_argument('--load_enhance', action='store_true',
                    help='load enhancement for different subnets')
parser.add_argument('--OFA', action='store_true',
                    help='OFA training')
parser.add_argument('--partition_ratio', default=0.25, type=float,
                    help="The partition ratio")
parser.add_argument('--VLB_conv', action='store_true',
                    help='enable VLB')
parser.add_argument('--VLB_conv_type', default=0, type=int,
                    help="Type of vlb conv")
parser.add_argument('--split_num', default=2, type=int,
                    help="Number of splits on the ring")
parser.add_argument('--simulate', action='store_true',
                    help='simulate model on validation set')
parser.add_argument('--trace_selection', default=0, type=int,
                    help="0: FCC trace, 1: random trace, 2: recorded trace")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss = LossType.from_string(args.loss)
args.decay_epoch = sorted([int(args.epochs * i if i < 1 else i) for i in args.decay_epoch])

best_prec1 = 0.
global_step = 0
best_avg_prec1 = 0.

if not args.seed:
    args.seed = random.randint(500, 1000)

if args.retrain:
    if not os.path.exists(args.retrain) or not os.path.isfile(args.retrain):
        raise ValueError(f"Path error: {args.retrain}")

if args.clamp != 1.0 and (args.loss == LossType.L1_SPARSITY_REGULARIZATION or args.loss == LossType.ORIGINAL):
    print("WARNING: Clamp only available at Polarization!")

if args.fix_gate:
    if not args.gate:
        raise ValueError("--fix-gate should be with --gate.")

if args.flops_weighted:
    if args.arch not in {'resnet56', 'vgg16_linear'}:
        raise ValueError(f"Unsupported architecture {args.arch}")

if not args.flops_weighted and (args.weight_max is not None or args.weight_min is not None):
    raise ValueError("When --flops-weighted is not set, do not specific --max-weight or --min-weight")

if args.flops_weighted and (args.weight_max is None or args.weight_min is None):
    raise ValueError("When --flops-weighted is set, do specific --max-weight or --min-weight")

if args.max_backup is not None:
    if args.max_backup <= 0:
        raise ValueError("--max-backup is supposed to be greater than 0, got {}".format(args.max_backup))
    pass

if args.target_flops and args.loss != LossType.POLARIZATION:
    raise ValueError(f"Conflict option: --loss {args.loss} --target-flops {args.target_flops}")

if args.target_flops and not args.gate:
    raise ValueError(f"Conflict option: --target-flops only available at --gate mode")

#print(args)
#print(f"Current git hash: {common.get_git_id()}")

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not os.path.exists(args.save):
    os.makedirs(args.save)
if args.backup_path is not None and not os.path.exists(args.backup_path):
    os.makedirs(args.backup_path)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10 if args.dataset == 'cifar10' else 100

if not args.retrain:
    if re.match("resnet[0-9]+", args.arch):
        model = models.__dict__[args.arch](num_classes=num_classes,
                                           gate=args.gate,
                                           bn_init_value=args.bn_init_value, aux_fc=False,
                                           width_multiplier=args.width_multiplier,
                                           use_input_mask=args.input_mask)
    elif re.match("vgg[0-9]+", args.arch):
        model = models.__dict__[args.arch](num_classes=num_classes,
                                           gate=args.gate,
                                           bn_init_value=args.bn_init_value,
                                           width_multiplier=args.width_multiplier)
        pass
    else:
        raise NotImplementedError("Do not support {}".format(args.arch))

else:  # initialize model for retraining with configs
    checkpoint = torch.load(args.retrain)
    if args.arch == "resnet56":
        model = models.resnet_expand.resnet56(cfg=checkpoint['cfg'], num_classes=num_classes,
                                              aux_fc=False)
        # initialize corresponding masks
        if "bn3_masks" in checkpoint:
            bn3_masks = checkpoint["bn3_masks"]
            bottleneck_modules = list(filter(lambda m: isinstance(m[1], BasicBlock), model.named_modules()))
            assert len(bn3_masks) == len(bottleneck_modules)
            for i, (name, m) in enumerate(bottleneck_modules):
                if isinstance(m, BasicBlock):
                    if isinstance(m.expand_layer, Identity):
                        continue
                    mask = bn3_masks[i]
                    assert mask[1].shape[0] == m.expand_layer.idx.shape[0]
                    m.expand_layer.idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze().reshape(-1)
        else:
            raise NotImplementedError("Key bn3_masks expected in checkpoint.")

    elif args.arch == "vgg16_linear":
        model = models.__dict__[args.arch](num_classes=num_classes, cfg=checkpoint['cfg'])
    else:
        raise NotImplementedError(f"Do not support {args.arch} for retrain.")

#training_flops = compute_conv_flops(model, cuda=True)
#print(f"Training model. FLOPs: {training_flops:,}")


def compute_flops_weight(cuda=False):
    # compute the flops weight for each layer in advance
    print("Computing the FLOPs weight...")
    flops_weight = model.compute_flops_weight(cuda=cuda)
    flops_weight_string_builder: typing.List[str] = []
    for fw in flops_weight:
        flops_weight_string_builder.append(",".join(str(w) for w in fw))
    flops_weight_string = "\n".join(flops_weight_string_builder)
    print("FLOPs weight:")
    print(flops_weight_string)
    print()

    return flops_weight_string


if args.flops_weighted:
    flops_weight_string = compute_flops_weight(cuda=True)

if args.cuda:
    model.cuda()

BASEFLOPS = compute_conv_flops(model, cuda=True)

if args.loss in {LossType.PROGRESSIVE_SHRINKING,LossType.PARTITION}:
    teacher_model = copy.deepcopy(model)
    if args.arch == 'resnet56':
        teacher_path = './original/resnet/model_best.pth.tar'
    else:
        teacher_path = './original/vgg/model_best.pth.tar'
    teacher_model.load_state_dict(torch.load(teacher_path)['state_dict'])

def compute_conv_flops_par(model: torch.nn.Module, cuda=False) -> float:
    """
    compute the FLOPs for CIFAR models
    NOTE: ONLY compute the FLOPs for Convolution layers and Linear layers
    """

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        if self.groups == 1:
            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        else:
            kernel_ops = self.kernel_size[0] * self.kernel_size[1]

        flops = kernel_ops * output_channels * output_height * output_width

        if hasattr(self, 'flops_multiplier'):
            flops *= self.flops_multiplier

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement()

        flops = weight_ops

        if hasattr(self, 'flops_multiplier'):
            flops *= self.flops_multiplier

        list_linear.append(flops)

    def add_hooks(net, hook_handles: list):
        """
        apply FLOPs handles to conv layers recursively
        """
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            add_hooks(c, hook_handles)

    handles = []
    add_hooks(model, handles)
    demo_input = torch.rand(8, 3, 32, 32)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops

if len(args.alphas)>1:
    args.ps_batch = len(args.alphas)*4
else:
    args.ps_batch = 1

if args.VLB_conv:
    print('Neural bridge type:',args.VLB_conv_type)
    if args.VLB_conv_type == 0:
        print('Deprecated. Not quite effective.')
        exit(0)
        sampling_interval = 1
        cfg = [1024,model.in_planes]
    elif args.VLB_conv_type == 1:
        print('Deprecated. Too computation heavy.')
        exit(0)
        sampling_interval = 1
        cfg = [1024,512,256,128,model.in_planes]
    elif args.VLB_conv_type == 2:
        # best for 0.25 now, but a little expensive
        sampling_interval = 3
        cfg = [352,128,128,model.in_planes]
    elif args.VLB_conv_type == 10:
        # best for two split
        sampling_interval = 3
        cfg = [352,144,model.in_planes]
    elif args.VLB_conv_type == 11:
        # best for two split
        sampling_interval = 3
        cfg = [352,192,model.in_planes]
    elif args.VLB_conv_type == 12:
        # best for two split
        sampling_interval = 3
        cfg = [352,224,model.in_planes]
    # elif args.VLB_conv_type == 3:
    #     sampling_interval = 3
    #     cfg = [352,96,96,96,model.in_planes]
    # elif args.VLB_conv_type == 4:
    #     sampling_interval = 3
    #     cfg = [352,64,64,model.in_planes]
    # elif args.VLB_conv_type == 5:
    #     # not useful
    #     sampling_interval = 3
    #     cfg = [352,128,model.in_planes]
    # elif args.VLB_conv_type == 6:
    #     sampling_interval = 3
    #     cfg = [352,128,128,128,model.in_planes]
    # elif args.VLB_conv_type == 7:
    #     # to try
    #     sampling_interval = 3
    #     cfg = [352,96,96,model.in_planes]
    # elif args.VLB_conv_type == 8:
    #     sampling_interval = 3
    #     cfg = [352,192,192,model.in_planes]
    # elif args.VLB_conv_type == 9:
    #     # two sub good, four sub not god
    #     sampling_interval = 3
    #     cfg = [352,model.in_planes]
    else:
        exit(0)
    layers = []
    for i in range(1,len(cfg)):
        layers.append(nn.Conv2d(cfg[i-1], cfg[i], kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(cfg[i]))
        layers.append(nn.ReLU())
    model.aggr = nn.Sequential(*layers).cuda()

    from types import MethodType
    # 3->352
    def modified_forward(self,x):
        end = time.time()
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(F.avg_pool2d(out, 4))
        for idx,l in enumerate(self.layer1):
            out = l(out)
            if idx%sampling_interval == sampling_interval-1:
                out_list.append(F.avg_pool2d(out, 4))
        for idx,l in enumerate(self.layer2):
            out = l(out)
            if idx%sampling_interval == sampling_interval-1:
                out_list.append(F.avg_pool2d(out, 2))
        for idx,l in enumerate(self.layer3):
            out = l(out)
            if idx%sampling_interval == sampling_interval-1:
                out_list.append(out)
        map_time = time.time() - end
        end = time.time()
        out = torch.cat(out_list,1)
        # aggregate layer
        out = self.aggr(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        reduce_time = time.time() - end
        return out, (map_time,reduce_time)
    model.forward = MethodType(modified_forward, model)
    model.aggr_sizes = [model.conv1.weight.size(0)]
    for layer in [model.layer1,model.layer2,model.layer3]:
        for idx,l in enumerate(layer):
            if idx%sampling_interval == sampling_interval-1:
                model.aggr_sizes += [l.conv2.weight.size(0)]

if args.split_running_stat:
    for module_name, bn_module in model.named_modules():
        if not isinstance(bn_module, nn.BatchNorm2d) and not isinstance(bn_module, nn.BatchNorm1d): continue
        for nid in range(len(args.alphas)):
            bn_module.register_buffer(f"mean{nid}",bn_module.running_mean.data.clone().detach())
            bn_module.register_buffer(f"var{nid}",bn_module.running_var.data.clone().detach())

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        # reinitialize model with resumed config
        if "vgg" in args.arch and 'cfg' in checkpoint:
            model = models.__dict__[args.arch](num_classes=num_classes,
                                               bn_init_value=args.bn_init_value,
                                               gate=args.gate)
            if args.cuda:
                model.cuda()

        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        if args.evaluate:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
else:
    checkpoint = None

# build optim
if args.bn_wd:
    no_wd_type = [models.common.SparseGate]
else:
    # do not apply weight decay on bn layers
    no_wd_type = [models.common.SparseGate, nn.BatchNorm2d, nn.BatchNorm1d]

no_wd_params = []  # do not apply weight decay on these parameters
for module_name, sub_module in model.named_modules():
    for t in no_wd_type:
        if isinstance(sub_module, t):
            for param_name, param in sub_module.named_parameters():
                if not isinstance(sub_module, models.common.SparseGate): continue
                no_wd_params.append(param)
                # print(f"No weight decay param: module {module_name} param {param_name}")

no_wd_params_set = set(no_wd_params)  # apply weight decay on the rest of parameters
wd_params = []
for param_name, model_p in model.named_parameters():
    if model_p not in no_wd_params_set:
        wd_params.append(model_p)
        # print(f"Weight decay param: parameter name {param_name}")

optimizer = torch.optim.SGD([{'params': list(no_wd_params), 'weight_decay': 0.},
                             {'params': list(wd_params), 'weight_decay': args.weight_decay}],
                            args.lr,
                            momentum=args.momentum)

def bn_weights(model):
    weights = []
    bias = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))
            bias.append((name, m.bias.data))

    return weights, bias


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    if args.loss == LossType.L1_SPARSITY_REGULARIZATION:
        sparsity = args.lbd
        bn_modules = list(filter(lambda m: (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)),
                                 model.named_modules()))
        bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name
        for m in bn_modules:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))
    else:
        raise NotImplementedError(f"Do not support loss: {args.loss}")


def clamp_bn(model, lower_bound=0, upper_bound=1):
    if model.gate:
        sparse_modules = list(filter(lambda m: isinstance(m, SparseGate), model.modules()))
    else:
        sparse_modules = list(
            filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))

    for m in sparse_modules:
        m.weight.data.clamp_(lower_bound, upper_bound)


def set_bn_zero(model: nn.Module, threshold=0.0) -> (nn.Module, int):
    """
    Set bn bias to zero
    Note: The operation is inplace. Parameters of the model will be changed!
    :param model: to set
    :param threshold: set bn bias to zero if corresponding lambda <= threshold
    :return modified model, the number of zero bn channels
    """
    with torch.no_grad():
        mask_length = 0
        for name, sub_module in model.named_modules():
            # only process bn modules
            if not (isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d)):
                continue

            mask = sub_module.weight.detach() <= threshold
            sub_module.weight[mask] = 0.
            sub_module.bias[mask] = 0.

            mask_length += torch.sum(mask).item()

    return model, mask_length


def bn_sparsity(model, loss_type, sparsity, t, alpha,
                flops_weighted: bool, weight_min=None, weight_max=None):
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    """
    bn_modules = model.get_sparse_layers()

    if loss_type == LossType.POLARIZATION or loss_type == LossType.L2_POLARIZATION:
        # compute global mean of all sparse vectors
        n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))
        sparse_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_

        sparsity_loss = 0.
        if flops_weighted:
            for sub_module in model.modules():
                if isinstance(sub_module, model.building_block):
                    flops_weight = sub_module.get_conv_flops_weight(update=True, scaling=True)
                    sub_module_sparse_layers = sub_module.get_sparse_modules()

                    for sparse_m, flops_w in zip(sub_module_sparse_layers, flops_weight):
                        # linear rescale the weight from [0, 1] to [lambda_min, lambda_max]
                        flops_w = weight_min + (weight_max - weight_min) * flops_w

                        sparsity_term = t * torch.sum(torch.abs(sparse_m.weight.view(-1))) - torch.sum(
                            torch.abs(sparse_m.weight.view(-1) - alpha * sparse_weights_mean))
                        sparsity_loss += flops_w * sparsity * sparsity_term
            return sparsity_loss
        else:
            for m in bn_modules:
                if loss_type == LossType.POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        torch.abs(m.weight - alpha * sparse_weights_mean))
                elif loss_type == LossType.L2_POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        (m.weight - alpha * sparse_weights_mean) ** 2)
                else:
                    raise ValueError(f"Unexpected loss type: {loss_type}")
                sparsity_loss += sparsity * sparsity_term

            return sparsity_loss
    else:
        raise ValueError()

def gen_partition_mask(net_id,weight_size):
    if args.split_num == 2:
        return gen_partition_mask_two_split(net_id,weight_size)
    elif args.split_num == 3:
        return gen_partition_mask_three_split(net_id,weight_size)
    elif args.split_num == 4:
        return gen_partition_mask_four_split(net_id,weight_size)
    else:
        exit(0)

def gen_partition_mask_four_split(net_id,weight_size):
    mask = torch.zeros(weight_size[:2]).long().cuda()
    c1,c2 = weight_size[:2]
    r = args.partition_ratio
    # linear layer
    if len(weight_size)==2:
        if net_id < 4:
            mask[:] = 1
            flops_multiplier = 1
        elif net_id >= 4:
            mask[:,int(c2*(net_id-4)/4):int(c2*((net_id-4)/4)+1-r)] = 1
            mask[:,:int(c2*((net_id-4)/4-r))] = 1
            flops_multiplier = 1-r
        return mask,flops_multiplier
    # conv layer
    if net_id <= 3:
        start = net_id
        if 3 != c2:
            mask[int(c1*(start/4-r)):int(c1*start/4),int(c2*(start/4-r)):int(c2*start/4)] = 1
            mask[int(c1*(1-r+start/4)):,int(c2*(1-r+start/4)):] = 1
            flops_multiplier = (1-r)**2 + r**2
        else:
            mask[:] = 1
            flops_multiplier = 1
    elif net_id >= 4:
        start = net_id-4
        if 3 != c2:
            flops_multiplier = (1-r)**2
        else:
            mask[int(c1*start/4):int(c1*(start/4+1-r))] = 1
            mask[:int(c1*(start/4-r))] = 1
            flops_multiplier = 1-r
    if 3 != c2:
        left0,right0,right1 = start/4,(start/4)+1-r,start/4-r
        mask[int(c1*left0):int(c1*right0),int(c2*left0):int(c2*right0)] = 1
        mask[:int(c1*right1),:int(c2*right1)] = 1
        mask[int(c1*left0):int(c1*right0),:int(c2*right1)] = 1
        mask[:int(c1*right1),int(c2*left0):int(c2*right0)] = 1
    return mask.view(c1,c2,1,1),flops_multiplier

def gen_partition_mask_three_split(net_id,weight_size):
    mask = torch.zeros(weight_size[:2]).long().cuda()
    c1,c2 = weight_size[:2]
    r = args.partition_ratio
    # linear layer
    if len(weight_size)==2:
        if net_id < 3:
            mask[:] = 1
            flops_multiplier = 1
        elif net_id == 3:
            mask[:,:int(c2*(1-r))] = 1
            flops_multiplier = 1-r
        elif net_id == 4:
            mask[:,int(c2*5/16):] = 1
            mask[:,:int(c2*(5/16-r))] = 1
            flops_multiplier = 1-r
        elif net_id == 5:
            mask[:,int(c2*10/16):] = 1
            mask[:,:int(c2*(10/16-r))] = 1
            flops_multiplier = 1-r
        return mask,flops_multiplier
    # conv layer
    if net_id == 0:
        if 3 != c2:
            mask[:int(c1*(1-r)),:int(c2*(1-r))] = 1
            mask[int(c1*(1-r)):,int(c2*(1-r)):] = 1
            flops_multiplier = (1-r)**2 + r**2
        else:
            mask[:] = 1
            flops_multiplier = 1
    elif net_id == 1:
        if 3 != c2:
            mask[int(c1*5/16):,int(c2*5/16):] = 1
            mask[:int(c1*(5/16-r)),:int(c2*(5/16-r))] = 1
            mask[int(c1*5/16):,:int(c2*(5/16-r))] = 1
            mask[:int(c1*(5/16-r)),int(c2*5/16):] = 1
            mask[int(c1*(5/16-r)):int(c1*5/16),int(c2*(5/16-r)):int(c2*5/16)] = 1
            flops_multiplier = (1-r)**2 + r**2
        else:
            mask[:] = 1
            flops_multiplier = 1
    elif net_id == 2:
        if 3 != c2:
            mask[int(c1*10/16):,int(c2*10/16):] = 1
            mask[:int(c1*(10/16-r)),:int(c2*(10/16-r))] = 1
            mask[int(c1*10/16):,:int(c2*(10/16-r))] = 1
            mask[:int(c1*(10/16-r)),int(c2*10/16):] = 1
            mask[int(c1*(10/16-r)):int(c1*10/16),int(c2*(10/16-r)):int(c2*10/16)] = 1
            flops_multiplier = (1-r)**2 + r**2
        else:
            mask[:] = 1
            flops_multiplier = 1
    elif net_id == 3:
        if 3 != c2:
            mask[:int(c1*(1-r)),:int(c2*(1-r))] = 1
            flops_multiplier = (1-r)**2
        else:
            mask[:int(c1*(1-r))] = 1
            flops_multiplier = 1-r
    elif net_id == 4:
        if 3 != c2:
            mask[int(c1*5/16):,int(c2*5/16):] = 1
            mask[:int(c1*(5/16-r)),:int(c2*(5/16-r))] = 1
            mask[int(c1*5/16):,:int(c2*(5/16-r))] = 1
            mask[:int(c1*(5/16-r)),int(c2*5/16):] = 1
            flops_multiplier = (1-r)**2
        else:
            mask[int(c1*5/16):] = 1
            mask[:int(c1*(5/16-r))] = 1
            flops_multiplier = 1-r
    elif net_id == 5:
        if 3 != c2:
            mask[int(c1*10/16):,int(c2*10/16):] = 1
            mask[:int(c1*(10/16-r)),:int(c2*(10/16-r))] = 1
            mask[int(c1*10/16):,:int(c2*(10/16-r))] = 1
            mask[:int(c1*(10/16-r)),int(c2*10/16):] = 1
            flops_multiplier = (1-r)**2
        else:
            mask[int(c1*10/16):] = 1
            mask[:int(c1*(10/16-r))] = 1
            flops_multiplier = 1-r
    return mask.view(c1,c2,1,1),flops_multiplier

def gen_partition_mask_two_split(net_id,weight_size):
    # different net_id map to different nets
    # different layer map to differnet subnets
    mask = torch.zeros(weight_size[:2]).long().cuda()
    c1,c2 = weight_size[:2]
    r = args.partition_ratio
    if len(weight_size)==2:
        if net_id < 2:
            mask[:] = 1
            flops_multiplier = 1
        elif net_id == 2:
            mask[:,:int(c2*(1-r))] = 1
            flops_multiplier = 1-r
        elif net_id == 3:
            mask[:,int(c2*r):] = 1
            flops_multiplier = 1-r
        return mask,flops_multiplier
    if net_id == 0:
        if 3 != c2:
            mask[:int(c1*(1-r)),:int(c2*(1-r))] = 1
            mask[int(c1*(1-r)):,int(c2*(1-r)):] = 1
            flops_multiplier = (1-r)**2 + r**2
        else:
            mask[:] = 1
            flops_multiplier = 1
    elif net_id == 1:
        if 3 != c2:
            mask[:int(c1*r),:int(c2*r)] = 1
            mask[int(c1*r):,int(c2*r):] = 1
            flops_multiplier = (1-r)**2 + r**2
        else:
            mask[:] = 1
            flops_multiplier = 1
    elif net_id == 2:
        if 3 != c2:
            mask[:int(c1*(1-r)),:int(c2*(1-r))] = 1
            flops_multiplier = (1-r)**2
        else:
            mask[:int(c1*(1-r))] = 1
            flops_multiplier = 1-r
    elif net_id == 3:
        if 3 != c2:
            mask[int(c1*r):,int(c2*r):] = 1
            flops_multiplier = (1-r)**2
        else:
            mask[int(c1*r):] = 1
            flops_multiplier = 1-r
    return mask.view(c1,c2,1,1),flops_multiplier

def sample_partition_network(old_model,net_id=None,deepcopy=True,inplace=True):
    if deepcopy:
        dynamic_model = copy.deepcopy(old_model)
    else:
        dynamic_model = old_model
    for module_name,bn_module in dynamic_model.named_modules():
        if not isinstance(bn_module, nn.BatchNorm2d) and not isinstance(bn_module, nn.BatchNorm1d): continue
        if args.split_running_stat:
            bn_module.running_mean.data = bn_module._buffers[f"mean{net_id}"]
            bn_module.running_var.data = bn_module._buffers[f"var{net_id}"]

    for bn_module,sub_module in zip(*dynamic_model.get_partitionable_bns_n_convs()):
        with torch.no_grad():
            if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.Linear): 
                mask,flops_multiplier = gen_partition_mask(net_id,sub_module.weight.size())
                sub_module.weight.data *= mask
                sub_module.flops_multiplier = flops_multiplier
                # realistic prune
                if not inplace and args.split_num == 2 and net_id >=2 and args.VLB_conv_type==10:
                    if net_id == 2:
                        in_chan_mask = mask[0,:,0,0]==1
                        out_chan_mask = mask[:,0,0,0]==1
                    elif net_id == 3:
                        in_chan_mask = mask[-1,:,0,0]==1
                        out_chan_mask = mask[:,-1,0,0]==1
                    if sub_module.weight.size(1) == 3:
                        sub_module.weight.data = sub_module.weight.data[out_chan_mask,:].clone()
                        bn_module.weight.data = bn_module.weight.data[out_chan_mask].clone()
                        bn_module.bias.data = bn_module.bias.data[out_chan_mask].clone()
                        bn_module.running_mean.data = bn_module._buffers[f"mean{net_id}"].data[out_chan_mask].clone()
                        bn_module.running_var.data = bn_module._buffers[f"var{net_id}"].data[out_chan_mask].clone()
                    else:
                        sub_module.weight.data = sub_module.weight.data[out_chan_mask,:].clone()
                        sub_module.weight.data = sub_module.weight.data[:,in_chan_mask].clone()
                        bn_module.weight.data = bn_module.weight.data[out_chan_mask].clone()
                        bn_module.bias.data = bn_module.bias.data[out_chan_mask].clone()
                        bn_module.running_mean.data = bn_module._buffers[f"mean{net_id}"].data[out_chan_mask].clone()
                        bn_module.running_var.data = bn_module._buffers[f"var{net_id}"].data[out_chan_mask].clone()
    if not inplace and args.split_num == 2 and net_id >=2 and args.VLB_conv_type==10:
        class LambdaLayer(nn.Module):
            def __init__(self, lambd):
                super(LambdaLayer, self).__init__()
                self.lambd = lambd

            def forward(self, x):
                out = self.lambd(x)
                return out
        if net_id == 2:
            dynamic_model.layer2[0].shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 8, 4), "constant", 0))
            dynamic_model.layer3[0].shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 16, 8), "constant", 0))
        elif net_id == 3:
            dynamic_model.layer2[0].shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 4, 8), "constant", 0))
            dynamic_model.layer3[0].shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 8, 16), "constant", 0))
        # modify aggr, only use a portion connections by concat masks
        mask = torch.tensor([]).long().cuda()
        for sz in dynamic_model.aggr_sizes:
            mask_par = torch.zeros(sz).long().cuda()
            r = args.partition_ratio
            if net_id == 2:
                mask_par[:int(sz*(1-r))] = 1
            elif net_id == 3:
                mask_par[int(sz*r):] = 1
            mask = torch.cat((mask,mask_par))
        with torch.no_grad():
            dynamic_model.aggr[0].weight.data = dynamic_model.aggr[0].weight.data[:,mask==1,:,:].clone()
    return dynamic_model

def update_partitioned_model(old_model,new_model,net_id,batch_idx):
    def copy_module_grad(old_module,new_module,subnet_mask=None):
        # copy running mean/var
        if isinstance(new_module,nn.BatchNorm2d) or isinstance(new_module,nn.BatchNorm1d):
            if args.split_running_stat:
                old_module._buffers[f"mean{net_id}"] = new_module.running_mean.data.clone().detach()
                old_module._buffers[f"var{net_id}"] = new_module.running_var.data.clone().detach()
            else:
                old_module.running_mean.data = new_module.running_mean.data
                old_module.running_var.data = new_module.running_var.data

        # weight
        w_grad0 = new_module.weight.grad.clone().detach()
        if subnet_mask is not None:
            w_grad0.data *= subnet_mask

        copy_param_grad(old_module.weight,w_grad0)
        # only update grad for specific targets
        if batch_idx%args.ps_batch == args.ps_batch-1:
            old_module.weight.grad = old_module.weight.grad_tmp.clone().detach()
            old_module.weight.grad_tmp = None

        # bias
        if hasattr(new_module,'bias') and new_module.bias is not None:
            b_grad0 = new_module.bias.grad.clone().detach()
            copy_param_grad(old_module.bias,b_grad0)
            if batch_idx%args.ps_batch == args.ps_batch-1:
                old_module.bias.grad = old_module.bias.grad_tmp.clone().detach()
                old_module.bias.grad_tmp = None
            
    def copy_param_grad(old_param,new_grad):
        new_grad *= args.alphas[net_id]
        if not hasattr(old_param,'grad_tmp') or old_param.grad_tmp is None:
            old_param.grad_tmp = new_grad
        else:
            old_param.grad_tmp += new_grad

    bns1,convs1 = old_model.get_partitionable_bns_n_convs()
    bns2,convs2 = new_model.get_partitionable_bns_n_convs()
    with torch.no_grad():
        for conv1,conv2 in zip(convs1,convs2):
            subnet_mask,_ = gen_partition_mask(net_id,conv1.weight.size())
            copy_module_grad(conv1,conv2,subnet_mask)
        for bn1,bn2 in zip(bns1,bns2):
            if bn1 is None:continue
            copy_module_grad(bn1,bn2)

    with torch.no_grad():
        old_non_par_modules = old_model.get_non_partitionable_modules()
        new_non_par_modules = new_model.get_non_partitionable_modules()
        for old_module,new_module in zip(old_non_par_modules,new_non_par_modules):
            copy_module_grad(old_module,new_module)
    
def sample_network(old_model,net_id=None,eval=False,check_size=False):
    num_subnets = len(args.alphas)
    if net_id is None:
        if not args.OFA:
            net_id = torch.tensor(0).random_(0,num_subnets)
        else:
            net_id = torch.rand(1)

    dynamic_model = copy.deepcopy(old_model)

    # config old model
    if not args.OFA or eval:
        for module_name,bn_module in dynamic_model.named_modules():
            if not isinstance(bn_module, nn.BatchNorm2d) and not isinstance(bn_module, nn.BatchNorm1d): continue
            # set the right running mean/var
            if args.split_running_stat:
                # choose the right running mean/var for a subnet
                # updated in the last update
                bn_module.running_mean.data = bn_module._buffers[f"mean{net_id}"]
                bn_module.running_var.data = bn_module._buffers[f"var{net_id}"]

    bn_modules,convs = dynamic_model.get_sparse_layers_and_convs()
    all_scale_factors = torch.tensor([]).cuda()
    for bn_module in bn_modules:
        all_scale_factors = torch.cat((all_scale_factors,bn_module.weight.data.abs()))
    
    # total channels
    total_channels = len(all_scale_factors)
    
    _,ch_indices = all_scale_factors.sort(dim=0)
    
    weight_valid_mask = torch.zeros(total_channels).long().cuda()
    if not args.OFA or eval:
        weight_valid_mask[ch_indices[total_channels//num_subnets*(num_subnets-1-net_id):]] = 1
    else:
        weight_valid_mask[ch_indices[int(total_channels*(1 - net_id)):]] = 1

    freeze_mask = 1-weight_valid_mask
    
    ch_start = 0
    for bn_module,conv in zip(bn_modules,convs):
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            inactive = weight_valid_mask[ch_start:ch_start+ch_len]==0
            bn_module.weight.data[inactive] = 0
            bn_module.bias.data[inactive] = 0
            out_channel_mask = weight_valid_mask[ch_start:ch_start+ch_len]==1
            bn_module.out_channel_mask = out_channel_mask.clone().detach()
            ch_start += ch_len
    # prune additional memory, save ckpt, check size, delete
    if check_size:
        if net_id == 0:
            static_model = copy.deepcopy(old_model)

            ch_start = 0
            bn_modules,convs = static_model.get_sparse_layers_and_convs()

            ckpt = static_model.state_dict()
            if args.load_running_stat:
                key_of_running_stat = []
                for k in ckpt.keys():
                    if 'running_mean' in k or 'running_var' in k:
                        key_of_running_stat.append(k)
                for k in key_of_running_stat:
                    del ckpt[k]

            torch.save({'state_dict':ckpt}, os.path.join(args.save, 'static.pth.tar'))

    if not eval:
        return freeze_mask,net_id,dynamic_model,ch_indices
    else:
        return dynamic_model
    
def update_shared_model(old_model,new_model,mask,batch_idx,ch_indices,net_id):
    def copy_module_grad(old_module,new_module,subnet_mask=None,enhance_mask=None):
        if subnet_mask is not None:
            freeze_mask = subnet_mask == 1
            keep_mask = subnet_mask == 0

        # copy running mean/var
        if isinstance(new_module,nn.BatchNorm2d) or isinstance(new_module,nn.BatchNorm1d):
            if args.split_running_stat:
                if subnet_mask is not None:
                    old_module._buffers[f"mean{net_id}"][keep_mask] = new_module.running_mean.data[keep_mask].clone().detach()
                    old_module._buffers[f"var{net_id}"][keep_mask] = new_module.running_var.data[keep_mask].clone().detach()
                else:
                    old_module._buffers[f"mean{net_id}"] = new_module.running_mean.data.clone().detach()
                    old_module._buffers[f"var{net_id}"] = new_module.running_var.data.clone().detach()
            else:
                if subnet_mask is not None:
                    old_module.running_mean.data[keep_mask] = new_module.running_mean.data[keep_mask]
                    old_module.running_var.data[keep_mask] = new_module.running_var.data[keep_mask]
                else:
                    old_module.running_mean.data = new_module.running_mean.data
                    old_module.running_var.data = new_module.running_var.data

        # weight
        w_grad0 = new_module.weight.grad.clone().detach()
        if subnet_mask is not None:
            w_grad0.data[freeze_mask] = 0

        copy_param_grad(old_module.weight,w_grad0)
        if batch_idx%args.ps_batch == args.ps_batch-1:
            old_module.weight.grad = old_module.weight.grad_tmp.clone().detach()
            old_module.weight.grad_tmp = None

        # bias
        if hasattr(new_module,'bias') and new_module.bias is not None:
            b_grad0 = new_module.bias.grad.clone().detach()
            if subnet_mask is not None:
                b_grad0.data[freeze_mask] = 0

            copy_param_grad(old_module.bias,b_grad0)
            if batch_idx%args.ps_batch == args.ps_batch-1:
                old_module.bias.grad = old_module.bias.grad_tmp.clone().detach()
                old_module.bias.grad_tmp = None
            
    def copy_param_grad(old_param,new_grad):
        if not args.OFA:
            new_grad *= args.alphas[net_id]
        if not hasattr(old_param,'grad_tmp') or old_param.grad_tmp is None:
            old_param.grad_tmp = new_grad
        else:
            old_param.grad_tmp += new_grad

    bns1,convs1 = old_model.get_sparse_layers_and_convs()
    bns2,convs2 = new_model.get_sparse_layers_and_convs()
    ch_start = 0
    for conv1,bn1,conv2,bn2 in zip(convs1,bns1,convs2,bns2):
        ch_len = conv1.weight.data.size(0)
        with torch.no_grad():
            subnet_mask = mask[ch_start:ch_start+ch_len]
            copy_module_grad(bn1,bn2,subnet_mask)
            copy_module_grad(conv1,conv2,subnet_mask)
        ch_start += ch_len
    
    with torch.no_grad():
        old_non_sparse_modules = get_non_sparse_modules(old_model)
        new_non_sparse_modules = get_non_sparse_modules(new_model)
        for old_module,new_module in zip(old_non_sparse_modules,new_non_sparse_modules):
            copy_module_grad(old_module,new_module)
            
def get_non_sparse_modules(model,get_name=False):
    sparse_modules = []
    bn_modules,conv_modules = model.get_sparse_layers_and_convs()
    for bn,conv in zip(bn_modules,conv_modules):
        sparse_modules.append(bn)
        sparse_modules.append(conv)
    sparse_modules_set = set(sparse_modules)
    non_sparse_modules = []
    for module_name, module in model.named_modules():
        if module not in sparse_modules_set:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
                if not get_name:
                    non_sparse_modules.append(module)
                else:
                    non_sparse_modules.append((module_name,module))
    return non_sparse_modules

def prune_while_training(model, arch, prune_mode, num_classes, avg_loss=None, inplace_prune=True, check_size=False):
    model.eval()
    saved_flops = []
    saved_prec1s = []
    if arch == "resnet56":
        from resprune_gate import prune_resnet
        from models.resnet_expand import resnet56 as resnet50_expand
        for i in range(len(args.alphas)):
            masked_model = sample_network(model,i,eval=True,check_size=check_size)
            pruned_model = prune_resnet(sparse_model=masked_model, pruning_strategy='fixed', prune_type='mask',
                                             sanity_check=False, prune_mode=prune_mode, num_classes=num_classes, inplace_prune=inplace_prune)
            prec1 = test(pruned_model)
            flop = compute_conv_flops(pruned_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
    elif arch == 'vgg16_linear':
        from vggprune_gate import prune_vgg
        from models import vgg16_linear
        # todo: update
        for i in range(len(args.alphas)):
            masked_model = sample_network(model,i,eval=True,check_size=check_size)
            pruned_model = prune_vgg(sparse_model=masked_model, pruning_strategy='fixed', prune_type='mask',
                                          sanity_check=False, prune_mode=prune_mode, num_classes=num_classes, inplace_prune=inplace_prune)
            prec1 = test(pruned_model)
            flop = compute_conv_flops(pruned_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
    else:
        # not available
        raise NotImplementedError(f"do not support arch {arch}")

    baseline_flops = compute_conv_flops(model, cuda=True)
    
    prune_str = f'{baseline_flops}. '
    for flop,prec1 in zip(saved_flops,saved_prec1s):
        prune_str += f"[{prec1:.4f}({(1-flop / baseline_flops)*100:.2f}%)],"
    log_str = ''
    if avg_loss is not None:
        log_str += f"{avg_loss:.3f} "
    for prec1 in saved_prec1s:
        log_str += f"{prec1:.4f} "
    with open(os.path.join(args.save,'train.log'),'a+') as f:
        f.write(log_str+'\n')
    return prec1,prune_str,saved_prec1s

def partition_while_training(model, arch, prune_mode, num_classes, avg_loss=None, fake_prune=True ,epoch=0,lr=0):
    model.eval()
    saved_prec1s = []
    saved_flops = []
    if arch == "resnet56":
        for i in range(len(args.alphas)):
            if args.alphas[i]==0:continue
            masked_model = sample_partition_network(model,net_id=i)
            prec1 = test(masked_model)
            flop = compute_conv_flops_par(masked_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
    else:
        # not available
        raise NotImplementedError(f"do not support arch {arch}")

    prune_str = ''
    for flop,prec1 in zip(saved_flops,saved_prec1s):
        prune_str += f"{prec1:.4f}({(flop / BASEFLOPS):.4f}),"
    log_str = f'{epoch} '
    if avg_loss is not None:
        log_str += f"{avg_loss:.3f},{lr:.4f},"
    for flop,prec1 in zip(saved_flops,saved_prec1s):
        log_str += f"{prec1:.4f}({(flop / BASEFLOPS):.4f}),"
    with open(os.path.join(args.save,'train.log'),'a+') as f:
        f.write(log_str+'\n')
    return saved_prec1s[0],prune_str,saved_prec1s

def create_wan_trace(trace_selection,num_query):
    # query size
    query_size = 3*32*32*4*args.test_batch_size # bytes
    wanlatency_list = [[] for _ in range(2)]
    if args.trace_selection == 0:
        # read network traces 
        import csv
        print('Reading network trace...')
        with open('curr_videostream.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                # bytes per second-> kilo bytes per second
                # micro seconds->milli seconds
                wanlatency_list[line_count//num_query] += [query_size/float(row["downthrpt"]) + float(row["latency"])/1e6] 
                line_count += 1
                if line_count == num_query*2:break
    elif args.trace_selection == 1:
        # recorded trace
        with open('DC/wanlatency','r') as f:
            line_count = 0
            for l in f.readlines():
                l = l.strip().split(' ')
                wanlatency_list[line_count//num_query] += [float(l[0])/1000.]
                line_count += 1
                if line_count == num_query*2:break
    elif args.trace_selection == 2:
        # generate random traces with increased std
        print('Generate random network traces...')
        wanlatency_mean,wanlatency_std = 0.25,0.025
        wanlatency_list = [np.random.normal(wanlatency_mean, wanlatency_std, num_query) for i in range(2)]
    else:
        print('Unsupported trace selction.')
        exit(0)
    return wanlatency_list

def analyze_all_recorded_traces():
    print('Analyzing all recorded traces...')
    trace_filenames = ['Local/000012','Local/000024','Local/000048','Local/000096','Local/000192','Local/000384','Local/000768',
                        'DC/000088','DC/000176','DC/000352','DC/000704','DC/001408','DC/002816','DC/005632']
    for filename in trace_filenames:
        latency_list = []
        bandwidth_list = []
        with open(filename,'r') as f:
            for l in f.readlines():
                l = l.strip().split(' ')
                latency_list += [float(l[0])/1e3]
                bandwidth_list += [float(l[1])/1e6]
        latency_mean,latency_std = np.array(latency_list).mean(),np.array(latency_list).std()
        bw_mean,bw_std = np.array(bandwidth_list).mean(),np.array(bandwidth_list).std()
        print(filename,f'{latency_mean:.3f}({latency_std:.3f}), {bw_mean:.3f}({bw_std:.3f})')

def evaluate_one_trace(trace_selection,lanlatency_list,wanlatency_list,all_map_time,all_reduce_time,all_correct,infer_time_lst,correct_lst):
    # analyze RMLaaS
    # calculate complete time for different subnets
    # find the result on each node and find the best among nodes
    # complete partition's time is infinite if its deadline is passed
    latency_thresh = 0.016 #s
    # each node keeps the largest partition within time limit
    # result of the fastest node will be kept 
    # complete and in-complete execution time
    RMLaaS_res = []
    RMLaaS_latency = []
    RMLaaS_latency_breakdown = []
    query_index = 0
    for mt0,mt1,mt2,mt3,rt0,rt1,rt2,rt3,c0,c1,c2,c3 in zip(*all_map_time,*all_reduce_time,*all_correct):
        # consider node 0: subnet{0,2}
        # mt0: complete latency on node 0
        # mt1: complete latency on node 1
        # mt2: partial latency on node 0
        # mt3: partial latency on node 1
        if mt3 + lanlatency_list[0][query_index] <= mt2 + latency_thresh:
            # complete partition
            node0_res = c0 # complete result on node 0
            node0_latency = [mt3 + rt0, lanlatency_list[0][query_index]] # latency on node 0:full reduce
        else:
            # in-complete
            node0_res = c2 # partial result on node 0
            node0_latency = [mt2 + rt2, latency_thresh] # latency on node 0:partial reduce
        # consider node 1: subnet{1,3}
        if mt2 + lanlatency_list[1][query_index] <= mt3 + latency_thresh:
            node1_res = c1
            node1_latency = [mt2 + rt1, lanlatency_list[1][query_index]] # full reduce
        else:
            node1_res = c3
            node1_latency = [mt3 + rt3, latency_thresh] # partial reduce
        # add cloud-edge latency
        node0_latency += [wanlatency_list[0][query_index]]
        node1_latency += [wanlatency_list[1][query_index]]
        # choose between 0 and 1
        if node0_latency <= node1_latency:
            res = node0_res
            latency = node0_latency
        else:
            res = node1_res
            latency = node1_latency
        RMLaaS_res += [res]
        RMLaaS_latency += [sum(latency)]
        RMLaaS_latency_breakdown += [latency]
        query_index += 1

    # latency breakdown: datacenter (communication/wait cost)+outer network+compute
    RMLaaS_latency_breakdown = np.array(RMLaaS_latency_breakdown)
    print('Latency break down:',RMLaaS_latency_breakdown.mean(axis=0),RMLaaS_latency_breakdown.std(axis=0))

    evaluate_service_metrics(RMLaaS_res,RMLaaS_latency,trace_selection,service_type=0)

    # analyze no replication
    no_rep_res = []
    no_rep_latency = []
    query_index = 0
    for ift0,c0 in zip(infer_time_lst,correct_lst):
        latency = ift0 + wanlatency_list[0][query_index]
        no_rep_res += [c0]
        no_rep_latency += [latency]
        query_index += 1

    evaluate_service_metrics(no_rep_res,no_rep_latency,trace_selection,service_type=1)

    # analyze total replication
    total_rep_res = []
    total_rep_latency = []
    query_index = 0
    for ift0,c0 in zip(infer_time_lst,correct_lst):
        if wanlatency_list[0][query_index]<=wanlatency_list[1][query_index]:
            latency = ift0 + wanlatency_list[0][query_index]
            total_rep_res += [c0]
        else:
            latency = ift0 + wanlatency_list[1][query_index]
            total_rep_res += [c1]
        total_rep_latency += [latency]
        query_index += 1

    evaluate_service_metrics(total_rep_res,total_rep_latency,trace_selection,service_type=2)

def evaluate_service_metrics(result_list,latency_list,trace_selection=0,num_experiments=10,service_type=0):
    print('SERVICE TYPE:',service_type)
    # consistency
    mean_acc = np.array(result_list).mean()
    print('Mean Top-1 accuracy:',mean_acc)

    # availability
    mean_latency = np.array(latency_list).mean()
    print('Mean latency:',mean_latency)
    N = len(latency_list)
    cdf_x = np.sort(np.array(latency_list))
    cdf_p = np.array(range(N))/float(N)
    cdf_str = 'LatencyCDF: '
    for i in range(10):
        cdf_str += f"{cdf_x[int((N-1)*(i+1)/10)]:.4f}({cdf_p[int((N-1)*(i+1)/10)]:.4f}),"
    print(cdf_str)

    # availability and consistency
    # loss/failure request rate: packets never arrive over all packets
    # vary loss rate 0.1,0.2,...
    # only work on FCC trace
    # effective accuracy under loss
    if trace_selection == 0:
        for loss_rate in [0.05,0.1,0.15,0.2]:
            # multiple exp for mean and std of effective accuracy
            eacc_list = []
            failrate_list = []
            for i in range(num_experiments):
                if service_type == 0 or service_type == 2:
                    avail_mask = np.array([random.random()>loss_rate or random.random()>loss_rate for _ in range(len(result_list))])
                else:
                    avail_mask = np.array([random.random()>loss_rate for _ in range(len(result_list))])
                effective_result = np.array(result_list).copy()
                effective_result[avail_mask==0] = 0.1
                eacc_list += [effective_result.mean()]
                failrate_list += [avail_mask.mean()]
            eacc_mean,eacc_std = np.array(eacc_list).mean(),np.array(eacc_list).std()
            fr_mean,fr_std = np.array(failrate_list).mean(),np.array(failrate_list).std()
            print(f'Loss rate:{loss_rate}, {eacc_mean:.3f}({eacc_std:.3f}), {fr_mean:.3f}({fr_std:.3f})')

    # consistency+availability
    # effective accuracy (treating missing deadline as random guess) vs. deadline for different approaches.
    # effective accuracy under deadline, e.g., 100ms, 200ms, ...
    for ddl in [0.1,0.2,0.3,0.4,0.5]:
        eacc_list = []
        for i in range(num_experiments):
            avail_mask = np.array(latency_list)<ddl
            effective_result = np.array(result_list).copy()
            effective_result[avail_mask==0] = 0.1
            eacc = effective_result.mean()
            eacc_list += [eacc]
        eacc_mean,eacc_std = np.array(eacc_list).mean(),np.array(eacc_list).std()
        print(f'Deadline:{ddl}, {eacc_mean:.3f}({eacc_std:.3f})')

def simulation(model, arch, prune_mode, num_classes, avg_loss=None, fake_prune=True ,epoch=0,lr=0):
    # analyze trace
    analyze_all_recorded_traces()
    print('Simulation with test batch size:',args.test_batch_size)
    model.eval()
    all_map_time = []
    all_reduce_time = []
    all_correct = []
    all_flop_ratios = []
    # map/reduce time for net[0-1] will not be used, but their preds will be used
    # every thing for net[2-3] will be used
    print('Running RMLaaS...')
    if arch == "resnet56":
        for i in range(len(args.alphas)):
            masked_model = sample_partition_network(model,net_id=i,inplace=False)
            flop = compute_conv_flops_par(masked_model, cuda=True)
            all_flop_ratios += [flop/BASEFLOPS]
            map_time_lst,reduce_time_lst,correct_lst = test(masked_model,map_reduce=True)
            all_map_time += [map_time_lst]
            all_reduce_time += [reduce_time_lst]
            all_correct += [correct_lst]
    else:
        # not available
        raise NotImplementedError(f"do not support arch {arch}")
    # evaluate map/reduce time
    # map
    node0_map_mean,node0_map_std = np.array(all_map_time[2]).mean(),np.array(all_map_time[2]).std()
    node1_map_mean,node1_map_std = np.array(all_map_time[3]).mean(),np.array(all_map_time[3]).std()
    # reduce
    node0_red_mean,node0_red_std = np.array(all_reduce_time[0]).mean(),np.array(all_reduce_time[0]).std()
    node1_red_mean,node1_red_std = np.array(all_reduce_time[1]).mean(),np.array(all_reduce_time[1]).std()
    print('Break compute latency down...')
    print(f'Map time: {node0_map_mean:.6f}({node0_map_std:.6f}), {node1_map_mean:.6f}({node1_map_std:.6f})')
    print(f'Reduce time: {node0_red_mean:.6f}({node0_red_std:.6f}), {node1_red_mean:.6f}({node1_red_std:.6f})')
    # flop ratios
    print('FLOPS ratios:',all_flop_ratios)

    # run originial model
    print('Running original ML service')
    infer_time_lst,correct_lst = test(teacher_model,standalone=True)
    # evaluate standalone running time
    infer_time_mean,infer_time_std = np.array(infer_time_lst).mean(),np.array(infer_time_lst).std()
    print(f'Standalone inference time:{infer_time_mean:.6f}({infer_time_std:.6f})')

    num_query = len(all_correct[0])
    # inter-node latency
    lanlatency_list = [[] for _ in range(2)]
    with open('Local/lanlatency','r') as f:
        line_count = 0
        for l in f.readlines():
            l = l.strip().split(' ')
            lanlatency_list[line_count//num_query] += [float(l[0])/1000.]
            line_count += 1
            if line_count == num_query*2:break
    # comm_size = 352*8*8*4*args.test_batch_size

    # wan latency
    for trace_selection in range(3):
        wanlatency_list = create_wan_trace(trace_selection,num_query)
        evaluate_one_trace(trace_selection,lanlatency_list,wanlatency_list,
                            all_map_time,all_reduce_time,all_correct,
                            infer_time_lst,correct_lst)

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

def train(epoch):
    model.train()
    global global_step
    avg_loss = 0.
    avg_sparsity_loss = 0.
    train_acc = 0.
    total_data = 0
    train_iter = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_iter):
        optimizer.zero_grad()
        if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
            if not args.OFA:
                nonzero = torch.nonzero(torch.tensor(args.alphas))
                net_id = int(nonzero[batch_idx%len(nonzero)][0])
                freeze_mask,net_id,dynamic_model,ch_indices = sample_network(model,net_id)
            else:
                freeze_mask,net_id,dynamic_model,ch_indices = sample_network(model)
        elif args.loss in {LossType.PARTITION}:
            deepcopy = len(args.alphas)>1
            nonzero = torch.nonzero(torch.tensor(args.alphas))
            # net_id = int(nonzero[batch_idx%len(nonzero)][0])
            net_id = int(nonzero[torch.tensor(0).random_(0,len(nonzero))][0])
            dynamic_model = sample_partition_network(model,net_id,deepcopy=deepcopy)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.loss in {LossType.PROGRESSIVE_SHRINKING} or (args.loss in {LossType.PARTITION} and deepcopy):
            output = dynamic_model(data)
        else:
            output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        if args.loss in {LossType.PROGRESSIVE_SHRINKING,LossType.PARTITION}:
            soft_logits = teacher_model(data)
            if isinstance(soft_logits, tuple):
                soft_logits, _ = soft_logits
            soft_label = F.softmax(soft_logits.detach(), dim=1)
            loss = cross_entropy_loss_with_soft_target(output, soft_label)
        else:
            loss = F.cross_entropy(output, target)
        
        # logging
        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_data += target.data.shape[0]

        if args.loss in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            sparsity_loss = bn_sparsity(model, args.loss, args.lbd,
                                        t=args.t, alpha=args.alpha,
                                        flops_weighted=args.flops_weighted,
                                        weight_max=args.weight_max, weight_min=args.weight_min)
            loss += sparsity_loss
            avg_sparsity_loss += sparsity_loss.data.item()
        loss.backward()
        if args.loss in {LossType.L1_SPARSITY_REGULARIZATION}:
            updateBN()
        if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
            update_shared_model(model,dynamic_model,freeze_mask,batch_idx,ch_indices,net_id)
        if args.loss in {LossType.PARTITION} and deepcopy:
            update_partitioned_model(model,dynamic_model,net_id,batch_idx)
        if args.loss not in {LossType.PROGRESSIVE_SHRINKING, LossType.PARTITION} or batch_idx%args.ps_batch==(args.ps_batch-1):
            optimizer.step()
        if args.loss in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            clamp_bn(model, upper_bound=args.clamp)
        global_step += 1
        train_iter.set_description(
            'Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]. Loss: {:.6f}'.format(
            global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), avg_loss / len(train_loader)))
    return avg_loss / len(train_loader)


def test(modelx,map_reduce=False,standalone=False):
    modelx.eval()
    correct = 0
    map_time_lst = []
    reduce_time_lst = []
    infer_time_lst = []
    correct_lst = []
    test_iter = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_iter):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if standalone:
                end = time.time()
            output = modelx(data)
            if isinstance(output, tuple):
                output, output_aux = output
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if map_reduce:
                map_time_lst.append(output_aux[0])
                reduce_time_lst.append(output_aux[1])
            elif standalone:
                infer_time_lst.append(time.time()-end)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if map_reduce or standalone:
                correct_lst.append(pred.eq(target.data.view_as(pred)).cpu().sum()/data.size(0))
    
    if map_reduce:
        return map_time_lst,reduce_time_lst,correct_lst
    elif standalone:
        return infer_time_lst,correct_lst
    else:
        return float(correct) / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, backup: bool, backup_path: str, epoch: int, max_backup: int, is_avg_best: bool=False):
    state['args'] = args

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    if is_avg_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_avg_best.pth.tar'))
    if backup and backup_path is not None:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))

        if max_backup is not None:
            while True:
                # remove redundant backup checkpoints to save space
                checkpoint_match = map(lambda f_name: re.fullmatch("checkpoint_([0-9]+).pth.tar", f_name),
                                       os.listdir(backup_path))
                checkpoint_match = filter(lambda m: m is not None, checkpoint_match)
                checkpoint_id: typing.List[int] = list(map(lambda m: int(m.group(1)), checkpoint_match))
                checkpoint_count = len(checkpoint_id)
                if checkpoint_count > max_backup:
                    min_checkpoint_epoch = min(checkpoint_id)
                    min_checkpoint_path = os.path.join(backup_path,
                                                       'checkpoint_{}.pth.tar'.format(min_checkpoint_epoch))
                    print(f"Too much checkpoints (Max {max_backup}, got {checkpoint_count}).")
                    print(f"Remove file: {min_checkpoint_path}")
                    os.remove(min_checkpoint_path)
                else:
                    break


if args.evaluate:
    if args.loss in {LossType.PARTITION}:
        prec1,prune_str,_ = partition_while_training(model, arch=args.arch,prune_mode="default",num_classes=num_classes)
    else:
        prec1,prune_str,_ = prune_while_training(model, arch=args.arch,prune_mode="default",num_classes=num_classes)
    print(prec1,prune_str)
    exit(0)

if args.simulate:
    assert args.split_num == 2
    assert args.test_batch_size == 8
    simulation(model, arch=args.arch,prune_mode="default",num_classes=num_classes)
    exit(0)

for epoch in range(args.start_epoch, args.epochs):
    if args.max_epoch is not None and epoch >= args.max_epoch:
        break

    args.current_lr = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)

    avg_loss = train(epoch) # train with regularization

    if args.loss in {LossType.PARTITION}:
        prec1,prune_str,saved_prec1s = partition_while_training(model, arch=args.arch,prune_mode="default",num_classes=num_classes,avg_loss=avg_loss,epoch=epoch,lr=args.current_lr)
    else:
        prec1,prune_str,saved_prec1s = prune_while_training(model, arch=args.arch,prune_mode="default",num_classes=num_classes,avg_loss=avg_loss)
    print(f"Epoch {epoch}/{args.epochs} learning rate {args.current_lr:.4f}",args.arch,args.save,prune_str,args.alphas)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    avg_prec1 = sum(saved_prec1s)/len(saved_prec1s)
    is_avg_best = avg_prec1 > best_avg_prec1
    best_avg_prec1 = max(avg_prec1, best_avg_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': prec1,
    }, is_best, filepath=args.save,
        backup_path=args.backup_path,
        backup=epoch % args.backup_freq == 0,
        epoch=epoch,
        max_backup=args.max_backup,
        is_avg_best=is_avg_best
    )
    
print("Best accuracy: " + str(best_prec1))
