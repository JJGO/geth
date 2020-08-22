import itertools

import numpy as np
from scipy.stats import variation
from scipy.spatial.distance import cosine

import torch
import torch.distributed as dist

from pylot.util import CSVLogger, printc


def SynchronizationAnalysis(exp, samples=5):

    if exp.is_master:
        logger = CSVLogger(exp.parent_path / "analysis.csv")

    def SynchronizationAnalysisCallback(train, epoch, iteration, postfix):
        if not train:
            return
        # printc(f"Analysis Callback @ {epoch}.{iteration}", color='RED')
        exp.checkpoint(tag="tmp")
        tmp = exp.checkpoint_path / "tmp.pt"
        tmp_grad = exp.checkpoint_path / "tmp-grad.pt"

        # printc("Made temp checkpoints", color='RED')

        with torch.set_grad_enabled(True):
            for x, y in itertools.islice(exp.val_dl, samples):
                x, y = x.to(exp.device), y.to(exp.device)
                yhat = exp.model(x)
                loss = exp.loss_func(yhat, y)
                loss.backward()

        # printc("Ran eval", color="RED")

        grads = {name: tensor.grad for name, tensor in exp.model.named_parameters()}
        torch.save(grads, tmp_grad)
        exp.optim.zero_grad()

        # printc("Computed grads, waiting for group", color='RED')

        dist.barrier()

        if exp.is_master:
            # printc(f"I'm master, loading weights, grads", color='GREEN')
            N = exp.get_param("distributed.world_size")

            state = {}
            state["weights"] = [
                torch.load(exp.parent_path / f"{i}/checkpoints/tmp.pt")[
                    "model_state_dict"
                ]
                for i in range(N)
            ]
            state["grads"] = [
                torch.load(exp.parent_path / f"{i}/checkpoints/tmp-grad.pt")
                for i in range(N)
            ]

            for param in state["grads"][0]:

                dists = {}

                values = {
                    s: [state[s][i][param].detach().cpu().numpy() for i in range(N)]
                    for s in ("weights", "grads")
                }

                for s in values:

                    dists[f"{s}_cos"] = [
                        cosine(x1.flatten(), x2.flatten())
                        for x1, x2 in itertools.combinations(values[s], 2)
                    ]

                    mean = np.mean(values[s], axis=0)

                    dists[f"{s}_l1"] = [
                        np.linalg.norm((x - mean).flatten(), ord=1) for x in values[s]
                    ]
                    dists[f"{s}_l2"] = [
                        np.linalg.norm((x - mean).flatten(), ord=2) for x in values[s]
                    ]

                    for k in list(dists.keys()):
                        dists[f"{k}_mean"] = np.mean(dists[k])
                        dists[f"{k}_std"] = np.std(dists[k])

                    dists[f"{s}_mean"] = mean.mean()

                logger.set(epoch=epoch, iteration=iteration, param=param, **dists)
                logger.flush()

        dist.barrier()
        tmp.unlink()
        tmp_grad.unlink()

    return SynchronizationAnalysisCallback
