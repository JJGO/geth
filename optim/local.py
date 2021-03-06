import torch
import torch.optim
import torch.distributed as dist

from torch.optim import Optimizer

from pylot.optim.lars import SGDLARS

from ..comm import communicate


class OptimWrapper(Optimizer):
    def __init__(self, params, inner_optim, **kwargs):
        if isinstance(inner_optim, str):
            inner_optim = getattr(torch.optim, inner_optim)(params, **kwargs)
        self.optim = inner_optim

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def state(self):
        return self.optim.state

    @property
    def defaults(self):
        return self.optim.defaults

    def zero_grad(self):
        self.optim.zero_grad()


class LocalOptim(OptimWrapper):
    def __init__(
        self,
        params,
        inner_optim,
        frequency=1,
        momentum_buffer="local",
        sync_grads=False,
        **kwargs,
    ):
        if inner_optim == "SGDLARS":
            inner_optim = SGDLARS(params, **kwargs)
        elif isinstance(inner_optim, str):
            inner_optim = getattr(torch.optim, inner_optim)(params, **kwargs)
        assert isinstance(inner_optim, Optimizer)

        self.optim = inner_optim
        self.frequency = frequency
        self._counter = 0
        # Whether to synchronize gradients or parameters of frequency =1
        self.momentum_buffer = momentum_buffer
        self.sync_grads = sync_grads

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

        if self.frequency == 1 and self.sync_grads:
            self._all_reduce("grads")

        self.optim.step(closure)

        if not self.sync_grads or self.frequency > 1:
            self._counter += 1
            if self._counter % self.frequency == 0:
                # self._counter = 0
                self.synchronize()

    def synchronize(self):
        self._all_reduce("params")

        # Momentum Buffers
        if self.momentum_buffer == "sync":
            self._all_reduce("momentum_buffer")

        elif self.momentum_buffer == "zero":
            for pg in self.optim.param_groups:
                # Zero out momentum buffers
                pg["momentum"] *= 0.0

    def _all_reduce(self, kind):
        tensors = []
        world_size = dist.get_world_size()
        for group in self.optim.param_groups:
            for p in group["params"]:
                t = None
                if kind == "params":
                    t = p
                elif kind == "grads":
                    t = p.grad
                elif kind == "momentum_buffer":
                    param_state = self.optim.state[p]
                    if "momentum_buffer" in param_state:
                        t = param_state["momentum_buffer"]
                if t is not None:
                    t.data.div_(world_size)
                    tensors.append(t.data)

        communicate(tensors, dist.all_reduce)

    def set_frequency(self, frequency):
        self._all_reduce("params")
        self._counter = 0
        self.frequency = frequency

    def __str__(self):
        # return f"{self.__class__.__name__}\n" + str(self.optim)
        return repr(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(frequency={self.frequency})\n{repr(self.optim)}"
        )

    def state_dict(self):
        state_dict = self.optim.state_dict()
        state_dict["frequency"] = self.frequency
        state_dict["counter"] = self._counter
        return state_dict

    def load_state_dict(self, state_dict):
        self.frequency = state_dict.get("frequency", 1)
        self._counter = state_dict.get("counter", 0)
        self.optim.load_state_dict(state_dict)
