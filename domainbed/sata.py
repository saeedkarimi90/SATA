import copy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import random
import torch.nn.functional as F
class SATA(nn.Module):

    def __init__(self, model, optimizer, steps = 1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.orginal = copy.deepcopy(model)
        self.C = model.num_classes
        self.freq = (torch.ones(self.C)/self.C).to(next(model.parameters()).device)
        self.num_domains = model.num_domains
        
        
        configure_model(model)
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        
        if self.episodic:
            self.reset()
            
        else:
            for step in range(self.steps):
                outputs = forward_and_adapt(self, x, self.model,self.orginal, self.optimizer)
        
         
        #return self.model.predict2(x)    
        return outputs 

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    ents = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def obtain_reliable_samples(orginal, x, num_domains):
    ys = []
    DSEoutputs = []
    
    for step in range(num_domains + 1): 
        orginal.featurizer.network.set_state(step)
        outputs = orginal.network(x)
        ys.append(outputs.detach().clone().argmax(1))
        if step > 0:
            DSEoutputs.append(outputs.detach().clone().softmax(1))
        del outputs
        
    with torch.no_grad():
        y0 = ys[0] + 1
        y1 = ys[1] + 1
        y2 = ys[2] + 1
        y3 = ys[3] + 1
        reliable = (y0 == y1)
        reliable = (reliable * y0 == y2).to(torch.long)
        reliable = (reliable * y0 == y3).to(torch.long)
    
    return reliable, DSEoutputs

def compute_entropy_loss(reliable, outputs):
    ent = (reliable * softmax_entropy(outputs)).mean(0)
    return ent

def compute_diversity_loss(self, outputs, DSEoutputs, reliable, num_domains):
    self.freq = .9 * self.freq + .1 * outputs.detach().mean(0)
    pred_label = outputs.detach().clone().argmax(1) 
    diverse_weight = 1 / self.freq[pred_label]
    sum = diverse_weight.sum()
    diverse_weight = diverse_weight / sum
    
    soft_outputs = outputs.softmax(1)
    unreliable = 1 - reliable 
    ConsLoss = 0 
    for s in range(num_domains):
        diff_outputs = (soft_outputs - DSEoutputs[s]) * unreliable[:, None] * diverse_weight[:, None]
        ConsLoss += diff_outputs.norm(dim=1, p=1).mean()
    
    return ConsLoss

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(self, x, model, orginal, optimizer):
    """Forward and adapt model on batch of data."""
    reliable, DSEoutputs = obtain_reliable_samples(orginal, x, self.num_domains)
    
    model.featurizer.network.set_state(0) 
    outputs = model.network(x)

    ent = compute_entropy_loss(reliable, outputs)
    ConsLoss = compute_diversity_loss(self, outputs, DSEoutputs, reliable, self.num_domains)
    
    loss = ent + .1 * ConsLoss / 3
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return outputs

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with SATA."""
   
    model.train()
    #for g in model.optimizer.param_groups:
    #    g['lr'] *= .1
    model.requires_grad_(True) #pacs = True
       
    for idx, m in enumerate(model.featurizer.network.layer3.modules()): 
        m.requires_grad_(False) 
       
    for idx, m in enumerate(model.featurizer.network.layer4.modules()): 
        m.requires_grad_(False)
        
    model.classifier.requires_grad_(False)


    return model
