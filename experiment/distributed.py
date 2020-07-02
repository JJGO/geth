import torch
from tqdm import tqdm


import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pylot.experiment import TrainExperiment, Experiment
from pylot.util import printc, StatsMeter, CUDATimer
from pylot.metrics import correct
from ..optim import LocalOptim

from .. import optim


class DistributedExperiment(Experiment):

    def __init__(self, env=None, cfg=None, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.env = env
        if self.env is not None:
            assert env['num_tasks'] == self.get_param('distributed.world_size')
            # Differs from TrainExperiment in specializing the path to the replica number
            self.parent_path = self.path
            self.path = self.parent_path / str(env['global_rank'])

    @property
    def is_master(self):
        return self.env['global_rank'] == 0


class DistributedTrainExperiment(TrainExperiment, DistributedExperiment):

    OPTIMS = [torch.optim, optim]

    def __init__(self, env=None, **kwargs):
        DistributedExperiment.__init__(self, env=env, **kwargs)

        self.build_data(**self.cfg['data'])
        self.build_model(**self.cfg['model'])
        self.build_loss(**self.cfg['loss'])
        self.build_train(**self.cfg['train'])

    def build_dataloader(self, **dataloader_kwargs):
        # Need a distributed sampler for data parallel jobs
        if self.env is not None:
            train_sampler = DistributedSampler(dataset=self.train_dataset,
                                               num_replicas=self.env['num_tasks'],
                                               rank=self.env['global_rank'])

            self.train_dl = DataLoader(self.train_dataset, sampler=train_sampler, **dataloader_kwargs)
            self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)

    def build_model(self, model, weights=None, ddp=False, **model_kwargs):
        super().build_model(model, weights=weights, **model_kwargs)
        self.ddp = ddp

    def to_device(self):
        super().to_device()
        if self.ddp:
            self.model = DDP(self.model)

    def sync_before_epoch(self):
        # Check all workers are on the same page
        device = self.device
        src = torch.Tensor([self._epoch]).to(device)
        dest = [torch.Tensor([-1]).to(device) for _ in range(self.env['num_tasks'])]
        dist.all_gather(dest, src)

        for i, val in enumerate(dest):
            other_epoch = val.item()
            if other_epoch != self._epoch:
                raise ValueError(f"Node {i} is on epoch:{other_epoch} but I'm on epoch:{self._epoch}")

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        for epoch in range(start, end):
            printc(f"Start epoch {epoch}", color='YELLOW')
            self._epoch = epoch
            self.sync_before_epoch()
            self.checkpoint(tag='last')
            self.checkpoint(tag=f"{epoch:03d}")
            self.log(epoch=epoch)
            self.train(epoch)
            self.eval(epoch)

            with torch.no_grad():
                for cb in self.epoch_callbacks:
                    cb(self, epoch)

            self.dump_logs()

    def run_epoch(self, train, epoch=0):
        progress = self.get_param('log.progress', True)
        if train:
            self.model.train()
            phase = 'train'
            dl = self.train_dl
        else:
            self.model.eval()
            phase = 'val'
            dl = self.val_dl

        total_loss = StatsMeter()
        acc1 = StatsMeter()
        acc5 = StatsMeter()
        timer = CUDATimer(skip=10, unit='ms')

        if progress:
            epoch_progress = tqdm(dl)
            epoch_progress.set_description(f"{phase.capitalize()} Epoch {epoch}/{self.epochs}")
            epoch_iter = iter(epoch_progress)
        else:
            epoch_iter = iter(dl)

        with torch.set_grad_enabled(train):
            for _ in range(len(dl)):
                with timer("t_data"):
                    x, y = next(epoch_iter)
                    x, y = x.to(self.device), y.to(self.device)
                with timer("t_forward"):
                    yhat = self.model(x)
                    loss = self.loss_func(yhat, y)
                if train:
                    with timer("t_backward"):
                        loss.backward()
                    with timer("t_optim"):
                        self.optim.step()
                        self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                postfix = {'loss': total_loss.mean, 'top1': acc1.mean, 'top5': acc5.mean}

                for cb in self.batch_callbacks:
                    cb(self, postfix)

                if progress:
                    epoch_progress.set_postfix(postfix)

        if train:
            self.log(timer.measurements)
            if isinstance(self.optim, LocalOptim):
                self.optim.avg_parameters()

        self.log({
            f'{phase}_loss': total_loss.mean,
            f'{phase}_acc1': acc1.mean,
            f'{phase}_acc5': acc5.mean,
        })

        return total_loss.mean
