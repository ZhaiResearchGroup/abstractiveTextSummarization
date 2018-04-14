import sys
sys.path.append("..")
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, hidden, encoder_output):
        return torch.bmm(hidden, encoder_output)