# import torch.distributed as dist

import torch
from tqdm import tqdm

from pylot.experiment import TrainExperiment, Experiment
from pylot.util import printc, StatsMeter, StatsTimer
from pylot.metrics import correct

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .. import optim


class DistributedExperiment(Experiment):
    pass


class DistributedTrainExperiment(TrainExperiment, DistributedExperiment):

    OPTIMS = [torch.optim, optim]

    def __init__(self, distributed=None, **kwargs):
        Experiment.__init__(self, **kwargs)
        self.distributed = distributed
        if self.distributed is not None:
            # Differs from TrainExperiment in specializing the path to the replica number
            self.parent_path = self.path
            self.path = self.parent_path / str(distributed['global_rank'])

        self.build_data(**self.cfg['data'])
        self.build_model(**self.cfg['model'])
        self.build_loss(**self.cfg['loss'])
        self.build_train(**self.cfg['train'])

    def build_dataloader(self, **dataloader_kwargs):
        # Need a distributed sampler
        num_replicas = self.distributed['num_tasks'] if self.distributed is not None else 1
        rank = self.distributed['global_rank'] if self.distributed is not None else 0
        train_sampler = DistributedSampler(dataset=self.train_dataset,
                                           num_replicas=num_replicas,
                                           rank=rank)

        self.train_dl = DataLoader(self.train_dataset, sampler=train_sampler, **dataloader_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)

    def sync_before_epoch(self):
        # Check all workers are on the same page
        device = self.device
        src = torch.Tensor([self._epoch]).to(device)
        dest = [torch.Tensor([-1]).to(device) for _ in range(self.distributed['num_tasks'])]
        dist.all_gather(dest, src)

        for i, val in enumerate(dest):
            other_epoch = val.item()
            if other_epoch != self._epoch:
                raise ValueError(f"Node {i} is on epoch:{other_epoch} but I'm on epoch:{self._epoch}")

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        try:
            for epoch in range(start, end):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.checkpoint()  # We checkpoint every epoch for reference
                self.eval(epoch)

                with torch.no_grad():
                    for cb in self.epoch_callbacks:
                        cb(self, epoch)

                self.log_epoch(epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')
            self.checkpoint(self.log_epoch_n - 1)

    def run_epoch(self, train, epoch=0):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            self.model.eval()
            prefix = 'val'
            dl = self.val_dl

        total_loss = StatsMeter()
        acc1 = StatsMeter()
        acc5 = StatsMeter()
        timer = StatsTimer()

        if self.cfg['log'].get('progress', True):
            epoch_progress = tqdm(dl)
            epoch_progress.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")
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

                if self.cfg['log'].get('progress', True):
                    epoch_progress.set_postfix(postfix)

        self.log({
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        }, timer.measurements)

        return total_loss.mean
