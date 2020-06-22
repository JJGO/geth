import submitit
import torch.distributed as dist

from .distributed import DistributedExperiment


class CheckpointWrapper:

    def __init__(self, experiment):
        self.experiment = experiment

    def __call__(self, cfg=None, **kwargs):

        if issubclass(self.experiment, DistributedExperiment):
            job_env = submitit.JobEnvironment()
            master_node = job_env.hostnames[0]
            kwargs['distributed'] = {
                'global_rank': job_env.global_rank,
                'local_rank': job_env.local_rank,
                'num_nodes': job_env.num_nodes,
                'num_tasks': job_env.num_tasks,
                'node': job_env.node,
                'master': master_node
            }
            tcp = f'tcp://{master_node}:42029'
            dist.init_process_group(init_method=tcp, rank=job_env.global_rank, world_size=job_env.num_tasks, backend='nccl')

        self.experiment = self.experiment(cfg, **kwargs)

        if not self.experiment.path.exists():
            self.experiment.run()
        else:
            self.experiment.load()
            self.experiment.resume()

    def checkpoint(self, checkpointpath: str) -> submitit.helpers.DelayedSubmission:

        self.experiment.checkpoint()
        resubmit = CheckpointWrapper(self.experiment)
        return submitit.helpers.DelayedSubmission(resubmit, cfg=self.experiment.cfg)


def submit_experiment(experiment, cluster_params, **kwargs):

    executor = submitit.AutoExecutor(folder='/checkpoint/jjgo/logs/')
    executor.update_params(**cluster_params)
    submit = CheckpointWrapper(experiment)
    job = executor.submit(submit, **kwargs)
    return job
