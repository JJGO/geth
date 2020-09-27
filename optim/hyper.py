import math

import torch.distributed as dist

from .local import LocalOptim
from ..comm import communicate
from ..graph import Hypercube


class HyperOptim(LocalOptim):
    def __init__(self, params, inner_optim, dim=1, **kwargs):
        super().__init__(params, inner_optim, **kwargs)
        self.dim = dim
        self.neighbors = None
        self.dim_counter = 0

    def _init_neighbors(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        Dim = int(math.log2(world_size))
        graph = Hypercube(Dim)
        faces = graph.nfaces(self.dim)
        neighbors = []
        for e in faces:
            g = dist.new_group(e)
            if rank in e:
                neighbors.append(g)
        self.neighbors = neighbors

    def set_dim(self, dim):
        self.dim = dim
        self.dim_counter = 0
        self._init_neighbors()

    def synchronize(self):
        self._sync("params")

        # Momentum Buffers
        if self.momentum_buffer == "sync":
            self._sync("momentum_buffer")

        elif self.momentum_buffer == "zero":
            for pg in self.optim.param_groups:
                # Zero out momentum buffers
                pg["momentum"] *= 0.0

    def _sync(self, kind):
        if self.neighbors is None:
            self._init_neighbors()
        tensors = []
        group_size = int(2 ** self.dim)
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
                    t.data.div_(group_size)
                    tensors.append(t.data)

        communicate(tensors, dist.all_reduce, group=self.neighbors[self.dim_counter])
        self.dim_counter = (self.dim_counter + 1) % len(self.neighbors)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["dim"] = self.dim
        state_dict["dim_counter"] = self.dim_counter
        return state_dict

    def load_state_dict(self, state_dict):
        self.dim = state_dict.get("dim", 1)
        self.dim_counter = state_dict.get("dim_counter", 0)
        super().load_state_dict(state_dict)
