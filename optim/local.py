import torch
import torch.optim
import torch.distributed as dist

from torch.optim import Optimizer

from ..comm import communicate


class LocalOptim(Optimizer):
    def __init__(
        self, params, inner_optim, frequency=1, momentum_buffer="local", **kwargs
    ):
        if isinstance(inner_optim, str):
            inner_optim = getattr(torch.optim, inner_optim)(params, **kwargs)
        assert isinstance(inner_optim, Optimizer)

        self.optim = inner_optim
        self.frequency = frequency
        self._counter = 0
        self.momentum_buffer = momentum_buffer

        assert momentum_buffer in (
            "local",
            "sync",
            "zero",
        ), f"Momentum buffer parameter must be one of: local, sync or zero"
        if momentum_buffer == "sync":
            assert isinstance(inner_optim, torch.optim.SGD)
            # TODO: other momentum optimizers?

    @torch.no_grad()
    def step(self, closure=None):

        self.optim.step(closure)

        self._counter += 1
        if self._counter % self.frequency == 0:
            self._counter = 0
            self.synchronize()

    def synchronize(self):
        self.avg_parameters()

        # Momentum Buffers
        if self.frequency > 1:
            if self.momentum_buffer == "sync":
                self.avg_parameters("momentum_buffer")
            elif self.momentum_buffer == "zero":
                for pg in self.optim.param_groups:
                    # Zero out momentum buffers
                    pg["momentum"] *= 0.0

    def avg_parameters(self, kind="params"):
        params = []
        world_size = dist.get_world_size()  # Get world size
        for group in self.optim.param_groups:
            for p in group["params"]:
                if kind == "params":
                    p.data.div_(world_size)
                    params.append(p.data)
                elif kind == "momentum_buffer":
                    param_state = self.optim.state[p]
                    if "momentum_buffer" in param_state:
                        buf = param_state["momentum_buffer"]
                        buf.data.div_(world_size)
                        params.append(buf.data)
        communicate(params, dist.all_reduce)

    def set_frequency(self, frequency):
        self.avg_parameters()
        self._counter = 0
        self.frequency = frequency

    def __str__(self):
        # return f"{self.__class__.__name__}\n" + str(self.optim)
        return repr(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(frequency={self.frequency})\n{repr(self.optim)}"
        )

    def zero_grad(self):
        self.optim.zero_grad()

    def state_dict(self):
        state_dict = self.optim.state_dict()
        state_dict["frequency"] = self.frequency
        return state_dict

    def load_state_dict(self, state_dict):
        self.frequency = state_dict.get("frequency", 1)
        self.optim.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults
