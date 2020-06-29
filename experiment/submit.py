import submitit
import torch.distributed as dist

from pylot.util import printc
from geth.experiment import DistributedExperiment


class CheckpointWrapper:

    def __init__(self, experiment_class):
        self.experiment_class = experiment_class

    def __call__(self, **kwargs):

        if issubclass(self.experiment_class, DistributedExperiment):
            job_env = submitit.JobEnvironment()
            master_node = job_env.hostnames[0]
            kwargs['distributed'] = dict(
                global_rank=job_env.global_rank,
                local_rank=job_env.local_rank,
                num_nodes=job_env.num_nodes,
                num_tasks=job_env.num_tasks,
                node=job_env.node,
                master=master_node
            )
            self.distributed = kwargs['distributed']
            tcp = f'tcp://{master_node}:42029'
            dist.init_process_group(init_method=tcp, rank=job_env.global_rank, world_size=job_env.num_tasks, backend='nccl')

        self.experiment = self.experiment_class(**kwargs)

        if not self.experiment.path.exists():
            self.experiment.run()
        else:
            self.experiment.load()
            self.experiment.resume()

    def checkpoint(self, **kwargs):
        printc("SubmiIt checkpoint", color='ORANGE')
        self.experiment.checkpoint()
        resubmit = CheckpointWrapper(self.experiment_class)
        # Need to use parent path for the whole resubmission
        path = self.experiment.path.as_posix()
        # Only master resubmits
        if issubclass(self.experiment_class, DistributedExperiment):
            path = self.experiment.parent_path.as_posix()
            if self.experiment.distributed['global_rank'] != 0:
                return
        return submitit.helpers.DelayedSubmission(resubmit, path=path)


def submit_experiment(cluster_params, experiment, debug=False, **kwargs):
    if debug:
        executor = submitit.LocalExecutor(folder='/tmp/submitit-logs')
    else:
        executor = submitit.AutoExecutor(folder='/checkpoint/jjgo/submitit-logs/')
    executor.update_parameters(**cluster_params)
    submit = CheckpointWrapper(experiment)
    job = executor.submit(submit, **kwargs)
    return job
