import os

# PyTorch 1.5 specific issue: https://github.com/pytorch/pytorch/issues/37377
# if torch.__version__ >= "1.5.0":
os.environ["MKL_THREADING_LAYER"] = "GNU"  # Workaround

import collections
import math
import pathlib
import yaml
import hashlib

import torch
import torch.distributed as dist
import submitit

from pylot.util import printc, setup_colored_traceback, allbut
from geth.experiment import DistributedExperiment
from geth.util import get_tcp_interface_name


DEFAULT_CLUSTER_PARAMS = dict(
    timeout_min=60 * 3,
    slurm_partition="priority,learnfair",
    cpus_per_task=10,
    slurm_comment="MLSys 2021 Rebuttal 12/7",
    # gpus_per_node=1,
    # tasks_per_node=1,
    # nodes=1,
    slurm_mem="64G",
)


def deterministic_hash(s):
    m = hashlib.md5(s.encode())
    return int(m.hexdigest(), 16)


class CheckpointWrapper:
    def __init__(self, experiment_class, cluster_params):
        self.experiment_class = experiment_class
        self.cluster_params = cluster_params.copy()

    def __call__(self, **kwargs):
        # This makes looking at errors much nicer
        setup_colored_traceback()

        # os.umask(0o000)

        # from torch.utils.collect_env import get_pretty_env_info
        # print(get_pretty_env_info())

        # Log cluster params
        cp_str = ("\n" + yaml.dump(self.cluster_params)).replace("\n", "\n\t")
        printc("Running with cluster params" + cp_str, color="CYAN")

        if issubclass(self.experiment_class, DistributedExperiment):
            self.distributed_setup()
            kwargs["env"] = self.distributed

        with torch.cuda.device(self.distributed["local_rank"]):

            self.experiment = self.experiment_class(**kwargs)
            self.save_cluster_params()

            if not self.experiment.path.exists():
                self.experiment.run()
            else:
                self.experiment.load()
                self.experiment.resume()

    def distributed_setup(self):
        if self.cluster_params.get("use_ethernet", False):
            printc("Forcing ethernet communication", color="CYAN")
            os.environ["NCCL_SOCKET_IFNAME"] = get_tcp_interface_name(
                network_interface_type="ethernet"
            )
            os.environ["NCCL_IB_DISABLE"] = "1"

        job_env = submitit.JobEnvironment()
        master_node = job_env.hostnames[0]
        attrs = ["global_rank", "local_rank", "num_nodes", "num_tasks", "node"]
        self.distributed = {k: getattr(job_env, k) for k in attrs}
        self.distributed["master"] = master_node
        # Init torch.distributed WORLD group
        printc(f"Running with job_id: {job_env.job_id}", color="CYAN")
        port = 42000 + (deterministic_hash(job_env.job_id) % 10000)
        addr = f"tcp://{master_node}:{port}"
        printc(f"Initializing dist group at {addr}", color="CYAN")
        dist.init_process_group(
            init_method=addr,
            rank=job_env.global_rank,
            world_size=job_env.num_tasks,
            backend="nccl",
        )
        # GPU isolation
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(job_env.local_rank)

    def checkpoint(self, **_):
        printc("SubmitIt checkpoint", color="ORANGE")
        resubmit = CheckpointWrapper(self.experiment_class, self.cluster_params)
        path = self.experiment.path

        # Only the master node gets the checkpoint and requeue signal
        if issubclass(self.experiment_class, DistributedExperiment):
            # Need to use parent path for the whole resubmission
            path = self.experiment.parent_path
        else:
            # Only makes sense to checkpoint when not distributed
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


def auto_dist_params(world_size, as_dict=True):
    MAX_GPUS_PER_NODE = 8
    nodes = math.ceil(world_size / MAX_GPUS_PER_NODE)
    gpus_per_node = min(MAX_GPUS_PER_NODE, world_size)
    tasks_per_node = gpus_per_node

    assert (
        world_size % gpus_per_node == 0
    ), f"GPUs-per-node {gpus_per_node} does not evenly divide num tasks {world_size}"
    assert (
        world_size == nodes * tasks_per_node
    ), f"World size {world_size} does not equal to number of tasks {nodes}nodes x {tasks_per_node}tasks/node"

    if not as_dict:
        return nodes, gpus_per_node, tasks_per_node
    return dict(nodes=nodes, gpus_per_node=gpus_per_node, tasks_per_node=tasks_per_node)


# def prepare_experiment(experiment, debug=False, cluster_params=None, **kwargs):

#     if issubclass(experiment, DistributedExperiment):
#         # DistributedExperiments must have been initialized to prevent UID randomization
#         cfg = load_cfg(kwargs['path'])
#         world_size = cfg["distributed"]["world_size"]
#         _cluster_params.update(auto_dist_params(world_size))

#     if cluster_params is not None:
#         _cluster_params.update(cluster_params)

#     if debug:
#         executor = submitit.LocalExecutor(folder="/tmp/submitit-logs")
#     else:
#         executor = submitit.AutoExecutor(folder="/checkpoint/jjgo/submitit-logs/")

#     executor.update_parameters(**_cluster_params)
#     submit = CheckpointWrapper(experiment, _cluster_params)
#     return submit, kwargs


def prepare_dist_experiment(experiment, cfg, cluster_params=None):
    _cluster_params = DEFAULT_CLUSTER_PARAMS.copy()
    # DistributedExperiments need to be instantiated so the
    # path (UID) is known to all replicas and is not randomized
    exp = experiment(**cfg)
    print(f"Initialized experiment at {exp.path}")
    # Infer correct cluster params for the experiment given the world_size
    _cluster_params.update(auto_dist_params(exp.get_param("distributed.world_size")))
    # Update other cluster params like timeout
    if cluster_params is not None:
        assert not any(
            k in cluster_params for k in ["node", "gpus_per_node", "tasks_per_node"]
        ), "node, gpus_per_node, tasks_per_node are inferred from world_size"
        _cluster_params.update(cluster_params)
    submit = CheckpointWrapper(experiment, _cluster_params)
    return submit, {"path": exp.path}


def prepare_experiment(experiment, cfg, cluster_params=None):
    _cluster_params = DEFAULT_CLUSTER_PARAMS.copy()
    if cluster_params is not None:
        _cluster_params.update(cluster_params)

    if issubclass(experiment, DistributedExperiment):
        submit, kwargs = prepare_dist_experiment(experiment, cfg, _cluster_params)
        return submit, kwargs, submit.cluster_params

    submit = CheckpointWrapper(experiment, _cluster_params)
    return submit, cfg, submit.cluster_params


def get_executor(local=False, batch=None):
    if local:
        return submitit.LocalExecutor(folder="/tmp/submitit-logs")

    executor = submitit.AutoExecutor(folder="/checkpoint/jjgo/submitit-logs/")
    if batch is not None:
        assert isinstance(batch, int)
        executor.update_parameters(slurm_array_parallelism=batch)

    return executor


def submit_experiment(experiment, cfg, cluster_params=None, local=False):
    submit, kwargs, cluster_params = prepare_experiment(experiment, cfg, cluster_params)
    executor = get_executor(local=local)

    cluster_params["slurm_gres"] = f"gpu:volta:{cluster_params.pop('gpus_per_node')}"
    executor.update_parameters(**allbut(cluster_params, ["use_ethernet"]))
    job = executor.submit(submit, **kwargs)
    print(f"Submitted job {job.job_id}")
    return job


def batch_submit_experiments(experiments, batch, cluster_params=None):

    executor = get_executor(local=False, batch=batch)

    submissions = []
    _cluster_params = None
    for exp, cfg in experiments:
        submit, kwargs, cl_ps = prepare_experiment(exp, cfg, cluster_params)
        submissions.append((submit, kwargs))
        if _cluster_params is None:
            _cluster_params = cl_ps
        else:
            assert (
                _cluster_params == cl_ps
            ), "Found experiment with different cluster params"

    # _cluster_params["gpus_per_node"] = f"volta:{_cluster_params['gpus_per_node']}"
    _cluster_params["slurm_gres"] = f"gpu:volta:{_cluster_params.pop('gpus_per_node')}"
    executor.update_parameters(**allbut(_cluster_params, ["use_ethernet"]))

    jobs = []
    with executor.batch():
        for submit, kwargs in submissions:
            job = executor.submit(submit, **kwargs)
            jobs.append(job)

    return jobs
