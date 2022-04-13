import numpy as np
import torch

def gmean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1. / len(a))

def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False

def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True

def get_next_batch(loaders, iters, src, device):
    try:
        inputs, labels = next(iters[src])
    except StopIteration:
        iters[src] = iter(loaders[src])
        inputs, labels = next(iters[src])
    return inputs.to(device), labels.to(device)

def get_gate_label(gate_out, idx, device):
    labels = torch.full(gate_out.size()[:-1], idx, dtype=torch.long)
    labels = labels.to(device)
    return labels
