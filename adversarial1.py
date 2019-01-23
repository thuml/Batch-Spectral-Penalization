import torch
import numpy as np
from torch.autograd import Variable


class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                self.high - self.low) + self.low)
        return -coeff * gradOutput



