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
        self.i = 0

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
        self.i = 0
        self._init_neighbors()

    def _all_reduce(self, kind):
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

        communicate(tensors, dist.all_reduce, group=self.neighbors[self.i])
        self.i = (self.i + 1) % len(self.neighbors)
