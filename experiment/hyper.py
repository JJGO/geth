import torch

from pylot.experiment import VisionClassificationTrainExperiment as VCTE
from pylot.util import printc, dict_recursive_update, CUDATimer

from ..optim import HyperOptim

from .distributed import DistributedTrainExperiment


class HyperDTE(DistributedTrainExperiment):
    def build_train(
        self, optim, epochs, dim_switch=None, **optim_kwargs,
    ):

        self.dim_switch = dict(dim_switch)

    @property
    def checkpoint_path(self):
        return self.path / "checkpoints"

    def checkpoint(self, tag=None):
        VCTE.checkpoint(self, tag=tag)

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        assert isinstance(self.optim, HyperOptim)
        # TODO factor out in an appropriate place
        timing = self.get_param("log.timing", False)
        if timing:
            sync_timer = CUDATimer(unit="ms")
            self.optim._all_reduce = sync_timer.wrap(
                self.optim._all_reduce, label="t_sync"
            )

        if start == 0:
            self.optim.set_dim(1)
            printc("Starting training, resetting dim to 1", color="ORANGE")

        for epoch in range(start, end):
            printc(f"Start epoch {epoch}", color="YELLOW")
            self._epoch = epoch
            self.sync_before_epoch()
            if epoch in self.dim_switch:
                dim = self.dim_switch[epoch]
                self.optim.set_dim(dim)
                printc(
                    f"Reached epoch {epoch}, setting dim to {dim}", color="CYAN",
                )

            self.checkpoint(tag="last")
            self.log(epoch=epoch)
            self.log(frequency=self.optim.frequency)
            self.log(dim=self.optim.dim)
            self.train(epoch)
            if timing:
                self.log(sync_timer.measurements)
                sync_timer.reset()
            self.eval(epoch)

            if self.scheduler:
                self.scheduler.step()

            with torch.no_grad():
                for cb in self.epoch_callbacks:
                    cb(epoch)

            self.dump_logs()

    def eval(self, epoch=0):
        self.checkpoint(tag="eval")
        self.optim._all_reduce("params")
        if self.is_master:
            super().eval(epoch)
        self.reload(tag="eval")
        # self.load_model(self.checkpoint_path / "eval.pt")
        (self.checkpoint_path / "eval.pt").unlink()
