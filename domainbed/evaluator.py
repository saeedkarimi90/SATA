import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader
import random
import copy
from copy import deepcopy
from domainbed import sata
import numpy as np
import os.path as osp

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def accuracy_from_loader(TTA,algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0
    
    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
    
        
        if TTA:
            logits = algorithm(x)   
            loss = F.cross_entropy(logits, y).item()
          

            
        else:
            with torch.no_grad():
                algorithm.featurizer.network.set_state(0) 
                logits = algorithm.network(x)   #predict
                loss = F.cross_entropy(logits, y).item()
                
        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss
 
        
        
def accuracy(TTA,algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(TTA, algorithm, loader, weights, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]
        
        
    def evaluate(self, algorithm, TTA = False, ret_losses=True):   # ret_losses = False
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}
        #----------------------------
        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])
            
            is_test = env_num in self.test_envs
            #---------------------------------------------------------
            
            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue
           
            
            if is_test:
                acc, loss = accuracy(TTA, algorithm, loader_kwargs, weights, debug=self.debug)
                accuracies[name] = acc
                losses[name] = loss
                summaries["test_" + inout] += acc / n_test_envs
        
        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
