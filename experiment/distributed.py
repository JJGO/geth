import pathlib
import yaml

import torch
import torchvision.models
from torch import nn

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pylot.experiment import VisionClassificationTrainExperiment as VCTE, Experiment
from pylot.scheduler import WarmupScheduler
from pylot.util import printc

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


class DistributedTrainExperiment(VCTE, DistributedExperiment):

    OPTIMS = [torch.optim, optim]

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

        if isinstance(self.model, torchvision.models.ResNet):
            for m in self.model.modules():
                if isinstance(m, torchvision.models.resnet.BasicBlock):
                    num_features = m.bn2.num_features
                    m.bn2.weight = nn.Parameter(torch.zeros(num_features))
                if isinstance(m, torchvision.models.resnet.Bottleneck):
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
        if self.env["global_rank"] == 0:
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
            self.checkpoint(tag=f"{epoch:03d}")
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


class ResumeLocalDTE(DistributedTrainExperiment):
    def __init__(
        self, sync_model_path=None, frequency=None, lr=None, path=None, env=None
    ):
        if path is None:
            self.sync_model_path = pathlib.Path(sync_model_path)
            with (self.sync_model_path / "config.yml").open("r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            cfg["train"]["optim"]["frequency"] = frequency
            cfg["train"]["sync_model"] = sync_model_path
            cfg["experiment"]["type"] = self.__class__.__name__
            if lr is not None:
                cfg["train"]["optim"]["lr"] = lr
            super().__init__(**cfg, env=env)
        else:
            super().__init__(path=path, env=env)

    def build_train(self, sync_model, optim, epochs, **optim_kwargs):
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
