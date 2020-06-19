import pathlib

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn

from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchviz import make_dot

from .base import Experiment
from ..util import printc, summary, StatsMeter

from .. import models
from .. import datasets
from .. import callbacks


def allbut(mapping, keys):
    import copy
    mapping = copy.deepcopy(mapping)
    for k in keys:
        del mapping[k]
    return mapping


class TrainingExperiment(Experiment):

    def __init__(self, cfg=None, **kwargs):

        # Default children kwargs
        super().__init__(cfg, **kwargs)

        # Save params
        self.build_data(**self.cfg['data'])
        self.build_model(**self.cfg['model'])
        self.build_loss(**self.cfg['loss'])
        self.build_train(**self.cfg['train'])

    def run(self):
        self.freeze()
        printc(f"Running {str(self)}", color='YELLOW')
        self.to_device()
        self.build_logging()
        self.run_epochs()

    def build_data(self, dataset, **data_kwargs):

        if hasattr(datasets, dataset):
            constructor = getattr(datasets, dataset)
            kwargs = allbut(data_kwargs, ['dataloader', 'resize'])
            self.train_dataset = constructor(train=True, **kwargs)
            self.val_dataset = constructor(train=False, **kwargs)

        elif hasattr(tv_datasets, dataset):
            constructor = getattr(tv_datasets, dataset)
            self.train_dataset = constructor('/tmp/torch_data', train=True, download=True, transform=transforms.ToTensor())
            self.val_dataset = constructor('/tmp/torch_data', train=False, download=True, transform=transforms.ToTensor())

        else:
            raise ValueError(f"Dataset {dataset} is not recognized")

        if 'resize' in data_kwargs:
            ratio = data_kwargs['resize']
            self.train_dataset = ResizeDataset(self.train_dataset, ratio=ratio)
            self.val_dataset = ResizeDataset(self.val_dataset, ratio=ratio)

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **data_kwargs['dataloader'])
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **data_kwargs['dataloader'])

    def build_model(self, model, resume=None, **model_kwargs):
        # in_ch = self.train_dataset.shape[-1]
        # out_ch = self.train_dataset.outshape[-1]

        if hasattr(models, model):
            constructor = getattr(models, model)
            self.model = constructor(**model_kwargs)

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous['model_state_dict'])

    def build_loss(self, loss, flatten=True, crop=False, **loss_kwargs):
        if hasattr(nn, loss):
            loss_func = getattr(nn, loss)(**loss_kwargs)
        if flatten:
            loss_func = flatten_loss(loss_func)
        if crop:
            assert 'resize' in self.cfg['data']
            loss_func = crop_loss(loss_func, ratio=self.cfg['data']['resize'])

        self.loss_func = loss_func

    def build_train(self, optim, epochs, resume_optim=False, **optim_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-4},
            'Adam': {'betas': (.9, .99), 'lr': 3e-4}
        }

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim

        if resume_optim:
            assert "resume" in self.cfg['model'], "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        # Assume classification experiment

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

    def checkpoint(self):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint-{epoch}.pt')

    def build_logging(self):
        super().build_logging()

        # Sample a batch
        x, y = next(iter(self.train_dl))
        x, y = x.to(self.device), y.to(self.device)

        # Save model summary
        with open(self.path / 'summary.txt', 'w') as f:
            s = summary(self.model, x.shape[1:], echo=False)
            print(s, file=f)

        # Save model topology
        yhat = self.model(x)
        loss = self.loss_func(yhat, y)
        g = make_dot(loss)
        # g.format = 'svg'
        g.render(self.path / 'topology')

        # Callbacks
        cbs = self.cfg['log']['batch_callbacks']
        self.batch_callbacks = [getattr(callbacks, k)(self, **args) for c in cbs for k, args in c.items()]
        cbs = self.cfg['log']['epoch_callbacks']
        self.epoch_callbacks = [getattr(callbacks, k)(self, **args) for c in cbs for k, args in c.items()]

    def run_epochs(self):
        try:
            for epoch in range(self.epochs):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                # self.eval(epoch) TODO Uncomment
                self.log_epoch(epoch)

                with torch.set_grad_enabled(False):
                    for cb in self.epoch_callbacks:
                        cb(self, epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def run_epoch(self, train, epoch=0):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            prefix = 'val'
            dl = self.val_dl
            self.model.eval()

        total_loss = StatsMeter()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if train:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                total_loss.add(loss.item() / dl.batch_size)

                postfix = {'loss': total_loss.mean}
                for cb in self.batch_callbacks:
                    cb(self, postfix)

                epoch_iter.set_postfix(postfix)

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
        })

        return total_loss.mean

    def train(self, epoch=0):
        return self.run_epoch(True, epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch)
