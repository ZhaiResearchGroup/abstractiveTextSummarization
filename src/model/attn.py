import sys
sys.path.append("..")
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    def __init__(self, input_size, output_size):
        super(Attn, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size

        self.vec2energy = nn.Linear(self.input_size, output_size)
        self.energy2out = nn.Linear(output_size, 1)

    def forward(self, hidden, encoder_outputs):
        return torch.bmm(encoder_outputs.transpose(0, 1), hidden.unsqueeze(2)).squeeze().unsqueeze(0)

    def score(self, hidden, encoder_output):
        energy = self.vec2energy(torch.cat((hidden, encoder_output), 1))
        energy.data.test_name = "energies"
        return self.energy2out(energy)