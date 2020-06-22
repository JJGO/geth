import torch.distributed as dist

from .base import Experiment
from .train import TrainingExperiment


class DistributedExperiment(Experiment):
    pass


class DistributedTrainingExperiment(TraininingExperiment, DistributedExperiment):

    # TODO: specialize logging using the cfg['distributed']['global_rank'] (just add a subpath)
    #       but make sure if resuming this is not done, i.e. the subpath should not be reflected in the self.cfg
    # TODO: every k_iterations all_reduce the parameters and buffers
    # TODO: Initialize the Distributed Sampler correctly
    # TODO: Post-Local SGD params are
    #    - steps to take with constant averaging, (time to basin)
    #    - frequency of parameter averagining


