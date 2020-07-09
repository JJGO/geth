import os
import math
import pathlib
import yaml

import torch
import submitit
import torch.distributed as dist

from pylot.util import printc, setup_colored_traceback
from geth.experiment import DistributedExperiment


class CheckpointWrapper:
    def __init__(self, experiment_class, cluster_params):
        self.experiment_class = experiment_class
        self.cluster_params = cluster_params.copy()

    def __call__(self, **kwargs):
        setup_colored_traceback()

        # Log cluster params
        cp_str = ("\n" + yaml.dump(self.cluster_params)).replace("\n", "\n\t")
        printc("Running with cluster params" + cp_str, color="CYAN")

        if issubclass(self.experiment_class, DistributedExperiment):
            self.distributed_setup()
            kwargs["env"] = self.distributed

        self.experiment = self.experiment_class(**kwargs)
        self.save_cluster_params()

        if not self.experiment.path.exists():
            self.experiment.run()
        else:
            self.experiment.load()
            self.experiment.resume()

    def distributed_setup(self):
        job_env = submitit.JobEnvironment()
        master_node = job_env.hostnames[0]
        self.distributed = dict(
            global_rank=job_env.global_rank,
            local_rank=job_env.local_rank,
            num_nodes=job_env.num_nodes,
            num_tasks=job_env.num_tasks,
            node=job_env.node,
            master=master_node,
        )
        tcp = f"tcp://{master_node}:42029"
        # Init torch.distributed WORLD group
        dist.init_process_group(
            init_method=tcp,
            rank=job_env.global_rank,
            world_size=job_env.num_tasks,
            backend="nccl",
        )
        # GPU isolation
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job_env.local_rank)

    def checkpoint(self, **kwargs):
        printc("SubmitIt checkpoint", color="ORANGE")
        resubmit = CheckpointWrapper(self.experiment_class, self.cluster_params)
        path = self.experiment.path

        # Only the master node gets the checkpoint and requeue signal
        if issubclass(self.experiment_class, DistributedExperiment):
            # Need to use parent path for the whole resubmission
            path = self.experiment.parent_path

        self.experiment.checkpoint(tag="interrupt")

        return submitit.helpers.DelayedSubmission(resubmit, path=path.as_posix())

    def save_cluster_params(self):
        path = self.experiment.path
        if isinstance(self.experiment, DistributedExperiment):
            path = self.experiment.parent_path
            if not self.experiment.is_master:
                return
        with open(path / "cluster_params.yml", "w") as f:
            yaml.dump(self.cluster_params, f)


def submit_experiment(experiment, debug=False, cluster_params=None, **kwargs):
    _cluster_params = dict(
        timeout_min=60 * 8,
        slurm_partition="learnfair",
        gpus_per_node=1,
        cpus_per_task=10,
        tasks_per_node=1,
        nodes=1,
        slurm_mem="64G",
    )

    if issubclass(experiment, DistributedExperiment):
        # DistributedExperiments must have been initialized to prevent UID randomization
        nodes, gpus_per_node, tasks_per_node = default_cluster_params_distributed(
            kwargs["path"]
        )
        _cluster_params.update(
            nodes=nodes, gpus_per_node=gpus_per_node, tasks_per_node=tasks_per_node
        )

    if cluster_params is not None:
        _cluster_params.update(cluster_params)

    if debug:
        executor = submitit.LocalExecutor(folder="/tmp/submitit-logs")
    else:
        executor = submitit.AutoExecutor(folder="/checkpoint/jjgo/submitit-logs/")

    # PyTorch 1.5 specific issue: https://github.com/pytorch/pytorch/issues/37377
    if torch.__version__ == "1.5.0":
        os.environ["MKL_THREADING_LAYER"] = "GNU"  # Workaround

    executor.update_parameters(**_cluster_params)
    submit = CheckpointWrapper(experiment, _cluster_params)
    job = executor.submit(submit, **kwargs)
    print(f"Submitted job {job.job_id}")
    return job


def default_cluster_params_distributed(path):
    cfg_path = pathlib.Path(path) / "config.yml"
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    world_size = cfg["distributed"]["world_size"]
    nodes = math.ceil(world_size / 8)
    gpus_per_node = min(8, world_size)
    tasks_per_node = gpus_per_node
    assert (
        world_size % gpus_per_node == 0
    ), f"GPUs-per-node {gpus_per_node} does not evenly divide num tasks {world_size}"
    assert (
        world_size == nodes * tasks_per_node
    ), f"World size {world_size} does not equal to number of tasks {nodes}nodes x {tasks_per_node}tasks/node"
    return nodes, gpus_per_node, tasks_per_node
