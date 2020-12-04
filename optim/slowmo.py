import torch
from pylot.util import delegates

from .local import LocalOptim


class SlowMo(LocalOptim):
    @delegates(to=LocalOptim.__init__)
    def __init__(
        self,
        parameters,
        inner_optim,
        slowmo_lr=1.0,
        slowmo_frequency=None,
        slowmo_momentum=0.5,
        slowmo_momentum_buffer=None,
        **kwargs
    ):
        super().__init__(parameters, inner_optim, **kwargs)
        self.slowmo_lr = slowmo_lr
        self.slowmo_momentum = slowmo_momentum
        # Only if it's different from LocalOpim frequency
        # TODO implement this
        self.slowmo_frequency = slowmo_frequency
        # TODO keep track of when old_param was set to avoid
        # using it when is older than current - frequency

    @torch.no_grad()
    def step(self, closure=None):
        super().step(closure=closure)

        frequency = self.slowmo_frequency or self.frequency

        if frequency > 1 and self._counter % frequency == 0:
            self.global_momentum_step()

    @torch.no_grad()
    def global_momentum_step(self):

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                param_state = self.state[p]

                if "old_param" not in param_state:
                    param_state["old_param"] = torch.clone(p.data).detach()
                    continue

                old_data = param_state["old_param"]

                if "global_momentum_buffer" not in param_state:
                    param_state["global_momentum_buffer"] = torch.zeros_like(p.data)

                buf = param_state["global_momentum_buffer"]
                (
                    buf.mul_(self.slowmo_momentum)
                    .sub_(p.data, alpha=1 / lr)
                    .add_(old_data.data, alpha=1 / lr)
                )

                old_data.add_(buf, alpha=-lr * self.slowmo_lr)
                p.data.copy_(old_data)

    def get_buffers(self):
        """Returns momentum/Adam buffers"""
        buffers = []
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                for buffer_name in ["momentum_buffer", "exp_avg", "exp_avg_sq"]:
                    if buffer_name in param_state:
                        buf = param_state[buffer_name]
                        buffers.append(buf)
        return buffers

    def reset_buffers(self):
        """Reset momentum/Adam buffers"""
        buffers = self.get_buffers()
        for buffer in buffers:
            buffer.zero_()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["slowmo_lr"] = self.slowmo_lr
        state_dict["slowmo_momentum"] = self.slowmo_momentum
        state_dict["slowmo_frequency"] = self.slowmo_frequency
        return state_dict

    def load_state_dict(self, state_dict):
        self.slowmo_lr = state_dict.get("slowmo_lr", 1.0)
        self.slowmo_momentum = state_dict.get("slowmo_momentum", 0.5)
        self.slowmo_frequency = state_dict.get("slowmo_frequency", None)
        super().load_state_dict(state_dict)
