import torch
import torch.optim
import torch.distributed as dist

from torch.optim import Optimizer

from ..comm import communicate


class LocalOptim(Optimizer):

    def __init__(self, params, inner_optim, frequency=1, **kwargs):
        if isinstance(inner_optim, str):
            inner_optim = getattr(torch.optim, inner_optim)(params, **kwargs)
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
        params = []
        world_size = dist.get_world_size()  # Get world size
        for group in self.optim.param_groups:
            for p in group['params']:
                p.data.div_(world_size)
                params.append(p.data)
        communicate(params, dist.all_reduce)

    def set_frequency(self, frequency):
        self._avg_parameters()
        self._counter = 0
        self.frequency = frequency

    def __str__(self):
        # return f"{self.__class__.__name__}\n" + str(self.optim)
        return repr(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(frequency={self.frequency})\n{repr(self.optim)}"

    def zero_grad(self):
        self.optim.zero_grad()

    def state_dict(self):
        state_dict = self.optim.state_dict()
        state_dict['frequency'] = self.frequency
        return state_dict

    def load_state_dict(self, state_dict):
        self.frequency = state_dict['frequency']
        self.optim.load_state_dict(state_dict)
