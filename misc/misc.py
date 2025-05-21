import torch
import random
import importlib
import numpy as np
import torch.nn as nn


def randAB(a=0, b=1):
    return np.random.rand() * (b - a) + a


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def setSeedGlobal(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nameSplit(cfgname, offset=1):
    tokens = cfgname.split(".")
    return ".".join(tokens[0 : offset]), tokens[offset:]


def getModule(prefix, modulePath):
    module = importlib.import_module(prefix)
    tokens = modulePath.split(".")
    tags = []
    flip = False
    for token in tokens:
        if flip:
            tags.append(token)
            continue
        try:
            module = importlib.import_module(".{}".format(token), package=prefix)
            prefix = "{}.{}".format(prefix, token)
        except:
            flip = True
            tags.append(token)
    return module, prefix, tags
