
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class SRT(Module):
    def __init__(self):
        super().__init__()
