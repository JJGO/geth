import pathlib
import shutil
import yaml

import pandas as pd

import torch
from torch import nn
from torchvision.models import resnet

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pylot
from pylot.experiment import VisionClassificationTrainExperiment as VCTE, Experiment
from pylot.scheduler import WarmupScheduler
from pylot.util import printc, dict_recursive_update
from pylot.models.vision import cifar_resnet

from .. import callbacks
from .. import optim


class DistributedExperiment(Experiment):
    def __init__(self, env=None, cfg=None, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.env = env
        self.parent_path = self.path
        if self.env is not None:
            assert env["num_tasks"] == self.get_param("distributed.world_size")
            # Differs from TrainExperiment in specializing the path to the replica number
            self.path = self.parent_path / str(env["global_rank"])

    @property
    def is_master(self):
        return self.env["global_rank"] == 0

    def build_logging(self):
        super().build_logging()

        if self.is_master:
            for i in ("checkpoints", "logs.csv"):
                dst = self.parent_path / i
                if not dst.exists():
                    # src = self.path / "logs.csv" # ! Bad because it's an absolute symlink
                    dst.symlink_to(f"0/{i}")


class DistributedTrainExperiment(VCTE, DistributedExperiment):

    OPTIMS = [torch.optim, optim]
    CALLBACKS = [pylot.callbacks, callbacks]

    def __init__(self, env=None, **kwargs):
        DistributedExperiment.__init__(self, env=env, **kwargs)

        self.build_data(**self.cfg["data"])
        self.build_model(**self.cfg["model"])
        self.build_loss(**self.cfg["loss"])
        self.build_train(**self.cfg["train"])

    @property
    def checkpoint_path(self):
        return self.parent_path / "checkpoints"

    def build_dataloader(self, **dataloader_kwargs):
        # Need a distributed sampler for data parallel jobs
        if self.env is not None:
            train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                num_replicas=self.env["num_tasks"],
                rank=self.env["global_rank"],
            )
            shuffle = None
        else:
            train_sampler = None
            shuffle = True

        self.train_dl = DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            sampler=train_sampler,
            **dataloader_kwargs,
        )
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)

    def build_model(self, model, ddp=False, **model_kwargs):
        """
        Initialize resnet50 similarly to "ImageNet in 1hr" paper
          - Batch norm moving average "momentum" <-- 0.9
          - Fully connected layer <-- Gaussian weights (mean=0, std=0.01)
          - gamma of last Batch norm layer of each residual block <-- 0
        """
        super().build_model(model, **model_kwargs)
        self.ddp = ddp

        if isinstance(self.model, (resnet.ResNet, cifar_resnet.ResNet)):
            # Distributed training uses 4 tricks to maintain accuracy with much larger
            # batch sizes. See https://arxiv.org/pdf/1706.02677.pdf for more details
            for m in self.model.modules():
                # The last BatchNorm layer in each block needs to be initialized as zero gamma
                if isinstance(m, (resnet.BasicBlock, cifar_resnet.BasicBlock)):
                    num_features = m.bn2.num_features
                    m.bn2.weight = nn.Parameter(torch.zeros(num_features))
                if isinstance(m, resnet.Bottleneck):
                    num_features = m.bn3.num_features
                    m.bn3.weight = nn.Parameter(torch.zeros(num_features))

                # Linear layers are initialized by drawing weights from a
                # zero-mean Gaussian with stddev 0.01. In the paper it was only for
                # fc layer, but in practice this seems to give better accuracy
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)

    def checkpoint(self, tag=None):
        # Only master worker should checkpoint since all models are synchronized
        # at the epoch boundary
        if self.is_master:
            super().checkpoint(tag=tag)

    def to_device(self):
        super().to_device()
        if self.ddp:
            self.model = DDP(self.model)

    def sync_before_epoch(self):
        # Check all workers are on the same page
        device = self.device
        src = torch.Tensor([self._epoch]).to(device)
        dest = [torch.Tensor([-1]).to(device) for _ in range(self.env["num_tasks"])]
        dist.all_gather(dest, src)

        for i, val in enumerate(dest):
            other_epoch = val.item()
            if other_epoch != self._epoch:
                raise ValueError(
                    f"Node {i} is on epoch:{other_epoch} but I'm on epoch:{self._epoch}"
                )

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        for epoch in range(start, end):
            printc(f"Start epoch {epoch}", color="YELLOW")
            self._epoch = epoch
            self.sync_before_epoch()
            self.checkpoint(tag="last")
            # self.checkpoint(tag=f"{epoch:03d}")
            self.log(epoch=epoch)
            self.train(epoch)

            if isinstance(self.optim, optim.LocalOptim):
                self.optim.synchronize()
            self.eval(epoch)

            if self.scheduler:
                self.scheduler.step()

            with torch.no_grad():
                for cb in self.epoch_callbacks:
                    cb(epoch)

            self.dump_logs()
        self.checkpoint(tag=f"{end:03d}")

    def rescale_lr(self, factor):

        printc(f"Rescaling LR by {factor}", color="ORANGE")

        for pg in self.optim.param_groups:
            pg["lr"] /= factor

        if isinstance(self.scheduler, WarmupScheduler):
            self.scheduler.target_lrs = [
                lr / factor for lr in self.scheduler.target_lrs
            ]


class PostLocalDTE(DistributedTrainExperiment):
    def build_train(self, optim, epochs, switch_local, rescale_lr=None, **optim_kwargs):
        super().build_train(optim, epochs, **optim_kwargs)

    @property
    def checkpoint_path(self):
        return self.path / "checkpoints"

    def checkpoint(self, tag=None):
        VCTE.checkpoint(self, tag=tag)

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        switch_local = self.get_param("train.switch_local")
        frequency = self.get_param("train.optim.frequency")
        assert isinstance(self.optim, optim.LocalOptim)

        if start == 0:
            self.optim.set_frequency(1)
            printc("Starting training, resetting frequency to 1", color="ORANGE")

        for epoch in range(start, end):
            printc(f"Start epoch {epoch}", color="YELLOW")
            self._epoch = epoch
            self.sync_before_epoch()
            if epoch == switch_local:
                self.optim.set_frequency(frequency)
                printc(
                    f"Reached epoch {switch_local}, setting frequency to {frequency}",
                    color="ORANGE",
                )
                if self.get_param("train.rescale_lr", False):
                    self.rescale_lr(self.get_param("train.rescale_lr"))

            self.checkpoint(tag="last")
            self.log(epoch=epoch)
            self.log(frequency=self.optim.frequency)
            self.train(epoch)
            self.eval(epoch)

            if self.scheduler:
                self.scheduler.step()

            with torch.no_grad():
                for cb in self.epoch_callbacks:
                    cb(epoch)

            self.dump_logs()

    def eval(self, epoch=0):
        # TODO: Use last synchronization instead
        self.checkpoint(tag="eval")
        self.optim.synchronize()
        if self.is_master:
            super().eval(epoch)
        self.reload(tag="eval")
        # self.load_model(self.checkpoint_path / "eval.pt")
        (self.checkpoint_path / "eval.pt").unlink()


class PartialPostLocalDTE(PostLocalDTE):
    def __init__(
        self,
        base_exp=None,
        switch_local=None,
        frequency=None,
        momentum_buffer="zero",
        path=None,
        env=None,
        **kwargs,
    ):
        if path is not None:
            assert env is not None, "env must be specified on distributed execution"
            super.__init__(path=path, env=env)
        else:
            assert frequency is not None, "frequency must be specified"
            assert switch_local is not None, "switch_local must be specified"
            base_exp = pathlib.Path(base_exp)
            # Read Base Experiment config
            with (base_exp / "config.yml").open("r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            # LocalOptim preparations
            cfg["train"]["switch_local"] = switch_local
            cfg["train"]["optim"]["frequency"] = frequency
            cfg["train"]["optim"]["momentum_buffer"] = momentum_buffer
            if cfg["train"]["optim"]["optim"] != "LocalOptim":
                cfg["train"]["optim"]["inner_optim"] = cfg["train"]["optim"]["optim"]
                cfg["train"]["optim"]["optim"] = "LocalOptim"

            if kwargs:
                # For any other change like log.root, train.epochs, &c
                cfg = dict_recursive_update(cfg, kwargs)

            # Need to init experiment so .path exists
            super().__init__(**cfg, env=env)

            # Setup Last Checkpoint by copying from base experiment
            # Neat thing is that the checkpoint will have the epoch so
            # it's harder to shoot yourself in the foot.

            checkpoint_path = base_exp / "checkpoints"  # load checkpoint from master

            # Reminder that checkpoint callback tags with epoch number at the end of the epoch
            last = checkpoint_path / f"{switch_local:03d}.pt"
            assert last.exists(), f"Could not find {last}"

            N = self.get_param("distributed.world_size")
            for i in range(N):
                checkpoint_path = self.path / f"{i}/checkpoints"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(last, checkpoint_path / "last.pt")

            # Copy logs over
            df = pd.read_csv((base_exp / "logs.csv").as_posix())
            df = df[df.epoch < switch_local]
            (self.path / "0").mkdir(parents=True, exist_ok=True)
            df.to_csv(self.path / "0/logs.csv", index=False)

    # # FIXME Fix this ugly thing because of callback at end
    # def load(self, tag=None):
    #     super().load(tag=tag)
    #     self._epoch += 1


class ResumeLocalDTE(DistributedTrainExperiment):
    def __init__(
        self,
        sync_model_path=None,
        frequency=None,
        rescale_lr=None,
        momentum_buffer="local",
        path=None,
        env=None,
    ):
        if path is None:
            self.sync_model_path = pathlib.Path(sync_model_path)
            with (self.sync_model_path / "config.yml").open("r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            cfg["train"]["optim"]["frequency"] = frequency
            cfg["train"]["sync_model"] = sync_model_path
            cfg["experiment"]["type"] = self.__class__.__name__
            if rescale_lr is not None:
                # Should be rescaled down by world size
                cfg["train"]["rescale_lr"] = rescale_lr
            cfg["train"]["optim"]["momentum_buffer"] = momentum_buffer
            super().__init__(**cfg, env=env)
        else:
            super().__init__(path=path, env=env)

    def build_train(self, sync_model, optim, epochs, rescale_lr=None, **optim_kwargs):
        # Takes a model to use the epoch weights from
        # It then symlinks the synced parent checkpoints so they can be loaded
        super().build_train(optim, epochs, **optim_kwargs)

        sync_model_path = pathlib.Path(sync_model)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.get_param("train.epochs")):
            checkpoint = sync_model_path / "checkpoints" / f"{epoch:03d}.pt"
            link = self.checkpoint_path / f"{epoch:03d}-sync.pt"
            if not link.exists():
                link.symlink_to(checkpoint)

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        for epoch in range(start, end):
            printc(f"Start epoch {epoch}", color="YELLOW")
            self._epoch = epoch
            self.sync_before_epoch()
            self.checkpoint(tag="last")
            self.reload(f"{epoch:03d}-sync")
            self.prepare_optim()
            self.log(epoch=epoch)
            self.train(epoch)
            if isinstance(self.optim, optim.LocalOptim):
                self.optim.synchronize()
            self.eval(epoch)
            self.checkpoint(tag=f"{epoch+1:03d}-local")

            with torch.no_grad():
                for cb in self.epoch_callbacks:
                    cb(epoch)

            self.dump_logs()

    def prepare_optim(self):
        assert isinstance(
            self.optim, optim.LocalOptim
        ), f"Optim mush be LocalOptim, found {self.optim.__class__.__name__}"
        assert (
            self.optim.frequency == 1
        ), f"Loaded LocalOptim must have frequency 1, found {self.optim.frequency}"
        self.optim.frequency = self.get_param("train.optim.frequency")
        self.optim.momentum_buffer = self.get_param(
            "train.optim.momentum_buffer", "local"
        )

        if self.get_param("train.rescale_lr", False):
            self.rescale_lr(self.get_param("train.rescale_lr"))
