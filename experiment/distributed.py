# import torch.distributed as dist

import torch
from tqdm import tqdm  # TODO remove tqdm, unnecessary for SLURM stuff

from pylot.experiment import TrainExperiment, Experiment
from pylot.util import printc, StatsMeter, StatsTimer
from pylot.metrics import correct

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .. import optim


class DistributedExperiment(Experiment):
    pass


class DistributedTrainingExperiment(TrainExperiment, DistributedExperiment):

    OPTIMS = [torch.optim, optim]

    def __init__(self, distributed=None, **kwargs):
        super().__init__(**kwargs)
        self.distributed = distributed
        if self.distributed is not None:
            # Assign current path
            #       specialize logging using the cfg['distributed']['global_rank'] (just add a subpath)
            #       but make sure if resuming this is not done, i.e. the subpath should not be reflected in the self.cfg
            #       |- This should be fine, since for resuming the whole path is used?
            self.parent_path = self.path
            self.path = self.parent_path / str(distributed['global_rank'])

            self.build_data(**self.cfg['data'])
            self.build_model(**self.cfg['model'])
            self.build_loss(**self.cfg['loss'])
            self.biild_train(**self.cfg['train'])

    def build_dataloader(self, **dataloader_kwargs):
        train_sampler = DistributedSampler(dataset=self.train_dataset,
                                           num_replicas=self.distributed['num_tasks'],
                                           rank=self.distributed['global_rank'])

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, sampler=train_sampler, **dataloader_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dataloader_kwargs)

    def run_epochs(self, start=0, end=None):
        end = self.epochs if end is None else end
        try:
            for epoch in range(start, end):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.checkpoint()  # We checkpoint every epoch for reference
                self.eval(epoch)
                self.log_epoch(epoch)

                with torch.no_grad():
                    for cb in self.epoch_callbacks:
                        cb(self, epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')
            self.checkpoint(self.log_epoch_n-1)

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

        epoch_progress = tqdm(dl)
        epoch_progress.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")
        epoch_iter = iter(epoch_progress)

        timer = StatsTimer()

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
                epoch_progress.set_postfix(postfix)

                for cb in self.batch_callbacks:
                    cb(self, postfix)

        self.log({
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        }, timer.measurements)

        return total_loss.mean
