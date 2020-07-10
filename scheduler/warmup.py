import math
import torch


class BaseWarmupScheduler:
    """Base class for all warmup schedules
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), f"{type(optimizer)} is not an Optimizer"
        self.optimizer = optimizer
        if isinstance(warmup_period, int):
            warmup_period = [warmup_period for _ in optimizer.param_groups]
        assert len(warmup_period) == len(
            optimizer.param_groups
        ), "Warmup period must be an integer or a list of integers with len of param_groups"
        self.warmup_period = warmup_period
        self.last_step = last_step
        self.target_lrs = self.get_lrs_optim()
        self.last_lrs = [None for _ in self.target_lrs]

    def get_lrs_optim(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, step=None):
        """Dampen the learning rates.
        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        new_lrs = self.get_lrs_optim()
        for i, _ in enumerate(self.target_lrs):
            if new_lrs[i] != self.last_lrs[i]:
                # optimizer lr changed without our intervention
                # need to change target
                self.target_lrs[i] = new_lrs[i]

        for i, group in enumerate(self.optimizer.param_groups):
            omega = self.warmup_factor(step, self.warmup_period[i])
            group["lr"] = self.target_lrs[i] * omega
            self.last_lrs[i] = group["lr"]

    def reset(self):
        # Resets to target LRs so, other LR schedulers do not read bad
        # LR values
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.target_lrs[i]

    def __enter__(self):
        self.step()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()

    def warmup_factor(self, step, warmup_period):
        raise NotImplementedError


class LinearWarmupScheduler(BaseWarmupScheduler):
    """Linear warmup schedule.
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): WarmupScheduler period
        last_step (int): The index of last step. (Default: -1)
    """

    def warmup_factor(self, step, warmup_period):
        return min(1.0, (step + 1) / warmup_period)


class ExponentialWarmupScheduler(BaseWarmupScheduler):
    """Exponential warmup schedule.
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Effective warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def warmup_factor(self, step, warmup_period):
        # 1-e**-3 ~= 0.95
        return 1.0 - math.exp(-3 * (step + 1) / warmup_period)
