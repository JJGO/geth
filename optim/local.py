import torch
from torch.optim import Optimizer
import torch.optim
from ..comm import communicate


class LocalOptim(Optimizer):

    def __init__(self, inner_optim, frequency=1, **kwargs):
        if isinstance(inner_optim, str):
            inner_optim = getattr(torch.optim, inner_optim)(**kwargs)
        assert isinstance(inner_optim, Optimizer)

        self.optim = inner_optim
        self.frequency = frequency
        self._counter = 0

    @torch.no_grad()
    def step(self):

        self.optim.step()

        self._counter += 1
        if self._counter % self.frequency == 0:
            self._counter = 0

            self._avg_parameters()

    def _avg_parameters(self):


    def set_frequency(self, frequency):
        self._avg_parameters()
        self._counter = 0
        self.frequency = frequency
