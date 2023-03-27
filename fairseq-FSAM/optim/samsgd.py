from typing import Iterable
import torch
from torch.optim._multi_tensor import SGD
import torch.optim
from . import LegacyFairseqOptimizer, register_optimizer

__all__ = ["SAMSGD"]

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List
import random
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class
from omegaconf import II, OmegaConf
from .scheduler import *
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)


@dataclass
class FairseqSAMConfig(FairseqDataclass):
    momentum: float = field(
        default=0, metadata={"help": "epsilon for Adam optimizer"}
    )
    dampening: float = field(
        default=0, metadata={"help": "epsilon for Adam optimizer"}
    )
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    rho: float = field(
        default=0.05, metadata={"help": "epsilon for Adam optimizer"}
    )
    alpha: float = field(
        default=0.4, metadata={"help": "epsilon for Adam optimizer"}
    )
    beta: float = field(
        default=0.5, metadata={"help": "epsilon for Adam optimizer"}
    )
    gamma: float = field(
        default=0.5, metadata={"help": "epsilon for Adam optimizer"}
    )
    num_samples: int = field(
        default=32, metadata={"help": "epsilon for Adam optimizer"}
    )
    keep_ratio: float = field(
        default=0.1, metadata={"help": "epsilon for Adam optimizer"}
    )
    mask_iter_e: float = field(
        default=1, metadata={"help": "epsilon for Adam optimizer"}
    )
    max_updates_gsam: int = field(
        default=0, metadata={"help": "epsilon for Adam optimizer"}
    )
    warmup_updates_gsam: int = field(
        default=0, metadata={"help": "epsilon for Adam optimizer"}
    )
    amsgrad: bool = field(
        default=False, metadata={"help": "whether using amsgrad"}
    )
    sam_type: str = field(
        default="sam", metadata={"help": "sam type"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    nesterov: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")


@register_optimizer("samsgd", dataclass=FairseqSAMConfig)
class FairseqSAM(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqSAMConfig, params, model=None):
        super().__init__(cfg)
        # base_optimizer=torch.optim.SGD
        base_optimizer=torch.optim.Adam
        self._optimizer = self.create_optimizer(cfg, params, base_optimizer, model=model, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "weight_decay": self.cfg.weight_decay,
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "amsgrad":self.cfg.amsgrad
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)

    def create_optimizer(self, args, params, base_optimizer, model=None, **kwargs):
        if args.sam_type == 'sam':
            optimizer = SAM(params, base_optimizer=base_optimizer,rho=args.rho, **kwargs)
        elif args.sam_type == 'fisher-sam':
            optimizer = SAM_Mask(params, model=model, base_optimizer=base_optimizer, 
                                mask_iter_e=args.mask_iter_e,
                                keep_ratio=args.keep_ratio,
                                rho=args.rho,
                                **kwargs)
            optimizer.init_mask()
        elif args.sam_type == 'gsam':
            scheduler = CosineScheduler(T_max=args.max_updates_gsam, max_value=kwargs["lr"], min_value=0.0, optimizer=base_optimizer, warmup_steps=args.warmup_updates_gsam)
            rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=kwargs["lr"], min_lr=0.0,
                max_value=0.01, min_value=0.001)
            optimizer = GSAM(params,model=model, base_optimizer=base_optimizer,gsam_alpha=args.alpha, rho_scheduler=rho_scheduler, **kwargs)
        elif args.sam_type == 'fisher-gsam':
            scheduler = CosineScheduler(T_max=args.max_updates_gsam, max_value=kwargs["lr"], min_value=0.0, optimizer=base_optimizer, warmup_steps=args.warmup_updates_gsam)
            rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=kwargs["lr"], min_lr=0.0,
                max_value=0.01, min_value=0.001)
            optimizer = GSAM_Mask(params,model=model, base_optimizer=base_optimizer,keep_ratio=args.keep_ratio,mask_iter_e=args.mask_iter_e, rho_scheduler=rho_scheduler, **kwargs)
            optimizer.init_mask()
        elif args.sam_type == 'esam':
            optimizer = ESAM(params,base_optimizer=base_optimizer, rho=args.rho, beta=args.beta, gamma=args.gamma, **kwargs)
        elif args.sam_type == 'fisher-esam':
            optimizer = ESAM_Mask(params, model=model, base_optimizer=base_optimizer, 
                                num_samples=args.num_samples,
                                keep_ratio=args.keep_ratio,
                                mask_iter_e=args.mask_iter_e,
                                beta=args.beta,
                                rho=args.rho,
                                gamma=args.gamma, **kwargs)
            optimizer.init_mask()
        else:
            raise ValueError
        
        return optimizer

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, loss_before=None,input_samples=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        if loss_before is not None:
            loss=loss_before.detach()
        else:
            loss, sample_size, logging_output = closure()
            loss=loss.detach()

        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        # return loss, sample_size, logging_output

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



class SAM_Mask(SAM):
    def __init__(self, params, base_optimizer, model, keep_ratio=0.1, mask_iter_e=100, rho=0.05, **kwargs):
        self.model = model.cuda()
        self.keep_ratio = keep_ratio
        self.mask_iter_e = mask_iter_e
        assert(0.0 <= keep_ratio <= 1.0)
        super().__init__(params=params, 
                        base_optimizer=base_optimizer, 
                        rho=rho,
                        **kwargs)
        self.mask = {}

    def init_mask(self):
        for name, param in self.model.named_parameters():
            self.mask[name] = torch.zeros_like(param, dtype=torch.float32, requires_grad=False).cuda()
        
        self.remove_based_partial('bias')
        self.remove_based_partial('embed')
        self.remove_based_nntype(nn.BatchNorm1d)
        self.remove_based_nntype(nn.BatchNorm2d)


    def remove_weight(self, name):
        if name in list(self.mask.keys()):
            print('Removing `{}` (size:{};(param:{}).'.format(name, self.mask[name].shape, self.mask[name].numel()))
            self.mask.pop(name)

    def remove_based_nntype(self, nn_type):
        for name, module in self.model.named_modules():
            if isinstance(module, nn_type):
                self.remove_weight(name)
                self.remove_weight(name + '.weight')
                self.remove_weight(name + '.bias')

    def remove_based_partial(self, partial_name):
        for name in list(self.mask.keys()):
            if partial_name in name:
                print('Removing `{}` (size:{};(param:{}).'.format(name, self.mask[name].shape, self.mask[name].numel()))
                self.mask.pop(name)

    def set_fisher_mask(self, closure):
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if name in self.mask:
                fisher_dict[name] = torch.zeros_like(param, requires_grad=False).cuda()

        # closure()
        for name, param in self.model.named_parameters():
            if name in self.mask and param.grad != None:
                fisher_dict[name] += torch.square(param.grad).data
        self.model.zero_grad()
        
        # get topk mask
        param_shape = {}
        fisher_value = []
        all_param_size = 0
        for name, fisher_info in fisher_dict.items():
            if name in self.mask:
                param_shape[name] = fisher_info.shape
                fisher_value.append(fisher_info.view(-1))
                all_param_size += fisher_info.numel()
        
        fisher_value = torch.cat(fisher_value, 0)

        keep_num = int(all_param_size * self.keep_ratio)

        param_to_be_update = torch.sort(fisher_value, descending=True)[1][:keep_num]
        mask_position = torch.zeros_like(fisher_value, dtype=torch.float, requires_grad=False).cuda()
        mask_position[param_to_be_update] = 1

        # update to self.mask
        start_idx = 0
        for name, shape in param_shape.items():
            end_idx = start_idx + torch.prod(torch.tensor(shape))
            # self.mask[name] = copy.deepcopy(mask_position[start_idx: end_idx].reshape(shape)).cuda()
            self.mask[name] = mask_position[start_idx: end_idx].reshape(shape)
            self.mask[name].requires_grad = False
            start_idx = end_idx

    def mask_info(self):
        all_param = 0
        zero_param = 0
        nonzero_param = 0
        for name, mask_value in self.mask.items():
            all_param += mask_value.numel()
            nonzero_param += torch.sum(mask_value).item()
            zero_param += mask_value.numel() - torch.sum(mask_value).item()
        sparse_ratio = zero_param / float(all_param)
        info = 'Mask has {:.3f}Mb param to choose, {:.3f}Mb params fire, {:.3f}Mb params freeze, sparse ratio:{:.3f}'.format(all_param /1024. /1024.,
                                                                                                                            nonzero_param /1024. /1024., 
                                                                                                                            zero_param /1024. /1024., 
                                                                                                                            sparse_ratio)
        return [info, all_param, nonzero_param, zero_param, sparse_ratio]

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            # for p in group["params"]:
            for name, p in self.model.named_parameters():
                p.requires_grad = True 
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                if name in self.mask:
                    e_w.data = e_w.data * self.mask[name]
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()


class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, beta=1.0, gamma=1.0, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()


    def step(self, closure=None, loss_before=None,input_samples=None):
        # first forward-backward step
        if loss_before is not None:
            l_all_before=loss_before.clone().detach()
        else:
            loss, sample_size, logging_output, loss_all = closure(backward_grad_sync=False, forward_grad_sync=True,loss_backward=True, is_esam=True)
            l_all_before=loss_all.clone().detach()

        # first step to w + e(w) 
        self.first_step(zero_grad=True)

        with torch.no_grad():
            _, _, logging_output, loss_all_after = closure(backward_grad_sync=False, forward_grad_sync=True,loss_backward=False, is_esam=True)
            instance_sharpness = loss_all_after - l_all_before

            if "n_mask" in logging_output.keys():
                n_mask=logging_output["n_mask"]
                # print(n_mask)
                input_samples["id"]=input_samples["id"].index_select(0,torch.tensor(n_mask).cuda())
                input_samples["query_tokens"]=[input_samples["query_tokens"][i] for i in n_mask]
                input_samples["query_masks"]=[input_samples["query_masks"][i] for i in n_mask]
                input_samples["candidate_tokens"]=[input_samples["candidate_tokens"][i] for i in n_mask]
                input_samples["candidate_masks"]=[input_samples["candidate_masks"][i] for i in n_mask]
                input_samples["labels"]=[input_samples["labels"][i] for i in n_mask]

            prob = self.gamma
            sample_size=instance_sharpness.shape[0]
            if prob >=0.99:
                indices = torch.full([sample_size], 1, dtype=bool).cuda()
            else:
                try:
                    position = int(sample_size * prob)
                    cutoff,_ = torch.topk(instance_sharpness, position)
                    cutoff = cutoff[-1]
                    indices = instance_sharpness >= cutoff 
                except:
                    indices = torch.full([sample_size], 1, dtype=bool).cuda()

        if "net_input1" in input_samples.keys() and "net_input2" in input_samples.keys():
            # for name in ["net_input1","net_input2","net_input3","net_input4","net_input5"]:
            for name in input_samples.keys():
                if  "net_input" in name:
                    for key, value in input_samples[name].items():
                        if key == "src_lengths":
                            input_samples[name][key]=torch.tensor(value)[indices].tolist()
                        else:
                            input_samples[name][key]=value[indices]
            input_samples["target"]=input_samples["target"][indices]
            input_samples["id"]=input_samples["id"][indices]

        elif "net_input" in input_samples.keys():
            # print(indices)
            for key, value in input_samples["net_input"].items():
                input_samples["net_input"][key]=value[indices]
            input_samples["target"]=input_samples["target"][indices]
            input_samples["id"]=input_samples["id"][indices]

        elif "query_tokens" in input_samples.keys() and "n_mask" not in logging_output.keys():
            input_samples["id"]=input_samples["id"][indices]
            input_samples["query_tokens"]=input_samples["query_tokens"][indices]
            input_samples["query_masks"]=input_samples["query_masks"][indices]
            input_samples["candidate_tokens"]=input_samples["candidate_tokens"][indices]
            input_samples["candidate_masks"]=input_samples["candidate_masks"][indices]
        
        elif "n_mask" in logging_output.keys():
            n_mask=indices.nonzero().view(-1).tolist()
            input_samples["id"]=input_samples["id"].index_select(0,torch.tensor(n_mask).cuda())
            input_samples["query_tokens"]=[input_samples["query_tokens"][i] for i in n_mask]
            input_samples["query_masks"]=[input_samples["query_masks"][i] for i in n_mask]
            input_samples["candidate_tokens"]=[input_samples["candidate_tokens"][i] for i in n_mask]
            input_samples["candidate_masks"]=[input_samples["candidate_masks"][i] for i in n_mask]
            input_samples["labels"]=[input_samples["labels"][i] for i in n_mask]

        else:
            print("error, samsgd.py")
            assert 0
        
        closure(backward_grad_sync=True, forward_grad_sync=False,loss_backward=True, is_esam=True, input_samples=input_samples)
        # second step
        self.second_step(True)
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class ESAM_Mask(ESAM):
    def __init__(self, params, base_optimizer, model, num_samples=32, keep_ratio=0.1, mask_iter_e=1, rho=0.05, beta=0.5, gamma=0.5, adaptive=False, **kwargs):
        self.model = model.cuda()
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.mask_iter_e = mask_iter_e
        assert(0.0 <= keep_ratio <= 1.0)
        super().__init__(params=params, 
                        base_optimizer=base_optimizer, 
                        rho=rho,
                        beta=beta,
                        gamma=gamma, 
                        adaptive=adaptive, **kwargs)
        self.mask = {}

    def init_mask(self):
        for name, param in self.model.named_parameters():
            self.mask[name] = torch.zeros_like(param, dtype=torch.float32, requires_grad=False).cuda()
        
        self.remove_based_partial('bias')
        self.remove_based_partial('embed')
        self.remove_based_nntype(nn.BatchNorm1d)
        self.remove_based_nntype(nn.BatchNorm2d)


    def remove_weight(self, name):
        if name in list(self.mask.keys()):
            print('Removing `{}` (size:{};(param:{}).'.format(name, self.mask[name].shape, self.mask[name].numel()))
            self.mask.pop(name)

    def remove_based_nntype(self, nn_type):
        for name, module in self.model.named_modules():
            if isinstance(module, nn_type):
                self.remove_weight(name)
                self.remove_weight(name + '.weight')
                self.remove_weight(name + '.bias')

    def remove_based_partial(self, partial_name):
        for name in list(self.mask.keys()):
            if partial_name in name:
                print('Removing `{}` (size:{};(param:{}).'.format(name, self.mask[name].shape, self.mask[name].numel()))
                self.mask.pop(name)

    def set_fisher_mask(self, closure):
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if name in self.mask:
                fisher_dict[name] = torch.zeros_like(param, requires_grad=False).cuda()

        # closure()
        for name, param in self.model.named_parameters():
            if name in self.mask and param.grad != None:
                fisher_dict[name] += torch.square(param.grad).data
        self.model.zero_grad()
        
        # get topk mask
        param_shape = {}
        fisher_value = []
        all_param_size = 0
        for name, fisher_info in fisher_dict.items():
            if name in self.mask:
                param_shape[name] = fisher_info.shape
                fisher_value.append(fisher_info.view(-1))
                all_param_size += fisher_info.numel()
        
        fisher_value = torch.cat(fisher_value, 0)

        keep_num = int(all_param_size * self.keep_ratio)

        param_to_be_update = torch.sort(fisher_value, descending=True)[1][:keep_num]
        mask_position = torch.zeros_like(fisher_value, dtype=torch.float, requires_grad=False).cuda()
        mask_position[param_to_be_update] = 1

        # update to self.mask
        start_idx = 0
        for name, shape in param_shape.items():
            end_idx = start_idx + torch.prod(torch.tensor(shape))
            # self.mask[name] = copy.deepcopy(mask_position[start_idx: end_idx].reshape(shape)).cuda()
            self.mask[name] = mask_position[start_idx: end_idx].reshape(shape)
            self.mask[name].requires_grad = False
            start_idx = end_idx

    def mask_info(self):
        all_param = 0
        zero_param = 0
        nonzero_param = 0
        for name, mask_value in self.mask.items():
            all_param += mask_value.numel()
            nonzero_param += torch.sum(mask_value).item()
            zero_param += mask_value.numel() - torch.sum(mask_value).item()
        sparse_ratio = zero_param / float(all_param)
        info = 'Mask has {:.3f}Mb param to choose, {:.3f}Mb params fire, {:.3f}Mb params freeze, sparse ratio:{:.3f}'.format(all_param /1024. /1024.,
                                                                                                                            nonzero_param /1024. /1024., 
                                                                                                                            zero_param /1024. /1024., 
                                                                                                                            sparse_ratio)
        return [info, all_param, nonzero_param, zero_param, sparse_ratio]

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            # for p in group["params"]:
            for name, p in self.model.named_parameters():
                p.requires_grad = True 
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                if name in self.mask:
                    e_w.data = e_w.data * self.mask[name]
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

from torch.nn.modules.batchnorm import _BatchNorm
import contextlib

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer,model, gsam_alpha=0.4, rho_scheduler=None, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model.cuda()
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        # self.rho=rho_scheduler
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha
        
        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        # self.rho_t=self.rho
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_( vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        

    @torch.no_grad()
    def step(self, closure=None, loss_before=None,input_samples=None):
        closure = torch.enable_grad()(closure)
        # get gradient
        # outputs, loss_value = closure()
        if loss_before is not None:
            loss=loss_before.detach()
        else:
            loss, sample_size, logging_output = closure()
            loss=loss.detach()
        
        with self.maybe_no_sync():
            # get gradient

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            closure()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
            
        # synchronize gradients across workers
        self._sync_grad()    

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

class GSAM_Mask(GSAM):
    def __init__(self, params, base_optimizer, model, keep_ratio=0.1, mask_iter_e=100, rho_scheduler=None,adaptive=False, **kwargs):
        self.model = model.cuda()
        self.keep_ratio = keep_ratio
        self.mask_iter_e = mask_iter_e
        assert(0.0 <= keep_ratio <= 1.0)
        super().__init__(params=params, 
                        model=model,
                        base_optimizer=base_optimizer, 
                        rho_scheduler=rho_scheduler,
                        adaptive=adaptive, **kwargs)
        self.mask = {}

    def init_mask(self):
        for name, param in self.model.named_parameters():
            self.mask[name] = torch.zeros_like(param, dtype=torch.float32, requires_grad=False).cuda()
        
        self.remove_based_partial('bias')
        # self.remove_based_partial('embed')
        self.remove_based_nntype(nn.BatchNorm1d)
        self.remove_based_nntype(nn.BatchNorm2d)


    def remove_weight(self, name):
        if name in list(self.mask.keys()):
            print('Removing `{}` (size:{};(param:{}).'.format(name, self.mask[name].shape, self.mask[name].numel()))
            self.mask.pop(name)

    def remove_based_nntype(self, nn_type):
        for name, module in self.model.named_modules():
            if isinstance(module, nn_type):
                self.remove_weight(name)
                self.remove_weight(name + '.weight')
                self.remove_weight(name + '.bias')

    def remove_based_partial(self, partial_name):
        for name in list(self.mask.keys()):
            if partial_name in name:
                print('Removing `{}` (size:{};(param:{}).'.format(name, self.mask[name].shape, self.mask[name].numel()))
                self.mask.pop(name)


    def set_fisher_mask(self, closure):
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if name in self.mask:
                fisher_dict[name] = torch.zeros_like(param, requires_grad=False).cuda()

        # closure()
        for name, param in self.model.named_parameters():
            if name in self.mask and param.grad != None:
                fisher_dict[name] += torch.square(param.grad).data
        self.model.zero_grad()
        
        # get topk mask
        param_shape = {}
        fisher_value = []
        all_param_size = 0
        for name, fisher_info in fisher_dict.items():
            if name in self.mask:
                param_shape[name] = fisher_info.shape
                fisher_value.append(fisher_info.view(-1))
                all_param_size += fisher_info.numel()
        
        fisher_value = torch.cat(fisher_value, 0)

        keep_num = int(all_param_size * self.keep_ratio)
        assert keep_num > 0

        param_to_be_update = torch.topk(fisher_value, keep_num)[1]
        mask_position = torch.zeros_like(fisher_value, dtype=torch.float, requires_grad=False).cuda()
        mask_position[param_to_be_update] = 1
        assert fisher_value.numel() == self.mask_info()[1]

        # update to self.mask
        start_idx = 0
        for name, shape in param_shape.items():
            end_idx = start_idx + torch.prod(torch.tensor(shape))
            # self.mask[name] = copy.deepcopy(mask_position[start_idx: end_idx].reshape(shape)).cuda()
            self.mask[name] = mask_position[start_idx: end_idx].reshape(shape)
            self.mask[name].requires_grad = False
            start_idx = end_idx
        assert start_idx == len(mask_position)


    def mask_info(self):
        all_param = 0
        zero_param = 0
        nonzero_param = 0
        for name, mask_value in self.mask.items():
            all_param += mask_value.numel()
            nonzero_param += torch.sum(mask_value).item()
            zero_param += mask_value.numel() - torch.sum(mask_value).item()
        sparse_ratio = zero_param / float(all_param)
        info = 'Mask has {:.3f}Mb param to choose, {:.3f}Mb params fire, {:.3f}Mb params freeze, sparse ratio:{:.3f}'.format(all_param /1024. /1024.,
                                                                                                                            nonzero_param /1024. /1024., 
                                                                                                                            zero_param /1024. /1024., 
                                                                                                                            sparse_ratio)
        return [info, all_param, nonzero_param, zero_param, sparse_ratio]

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps) / self.keep_ratio

            for name, p in self.model.named_parameters():
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                if name in self.mask:
                    e_w.data = e_w.data * self.mask[name]
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w