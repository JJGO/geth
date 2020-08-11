import itertools

import numpy as np
from scipy.stats import variation
from scipy.spatial.distance import cosine

import torch
import torch.distributed as dist

from pylot.util import CSVLogger

# fmt: off
def SynchronizationAnalysis(exp):

    if exp.is_master:
        logger = CSVLogger(exp.parent_path / "analysis.csv")

    def SynchronizationAnalysisCallback(epoch, iteration, postfix):
        exp.checkpoint(tag="tmp")
        tmp = exp.checkpoint_path / "tmp.pt"
        tmp_grad = exp.checkpoint_path / "tmp-grad.pt"

        with torch.set_grad_enabled(True):
            for x, y in exp.val_dl:
                x, y = x.to(exp.device), y.to(exp.device)
                yhat = exp.model(x)
                loss = exp.loss_func(yhat, y)
                loss.backward()

        grads = {name: tensor.grad for name, tensor in exp.model.named_parameters()}
        torch.save(grads, tmp_grad)
        exp.optim.zero_grad()

        dist.barrier()

        if exp.is_master:
            N = exp.get_param("distributed.world_size")

            weights = [torch.load(exp.parent_path / f"{i}/checkpoints/tmp.pt")["model_state_dict"] for i in range(N)]
            grads = [torch.load(exp.parent_path / f"{i}/checkpoints/tmp-grad.pt") for i in range(N)]

            weights = {param: np.array([weights[i][param].detach().cpu().numpy() for i in range(N)]) for param in weights[0]}
            grads = {param: np.array([grads[i][param].detach().cpu().numpy() for i in range(N)]) for param in grads[0]}

            for param in weights:
                cv_param = variation(weights[param], axis=0)
                cos_param = [cosine(w1.flatten(), w2.flatten()) for w1, w2 in itertools.combinations(weights[param], 2)]
                if param in grads:
                    cv_grad = variation(grads[param], axis=0)
                    cos_grad = [cosine(g1.flatten(), g2.flatten()) for g1, g2 in itertools.combinations(grads[param], 2)]

                logger.set(
                    epoch=epoch,
                    iteration=iteration,
                    param=param,
                    cv_param_mean=cv_param.mean(),
                    cv_param_std=cv_param.std(),
                    cv_grad_mean=cv_grad.mean(),
                    cv_grad_std=cv_grad.std(),
                )

                if param in grads:
                    logger.set(
                        cos_param_mean=np.mean(cos_param),
                        cos_param_std=np.std(cos_param),
                        cos_grad_mean=np.mean(cos_grad),
                        cos_grad_std=np.std(cos_grad)
                    )

            logger.flush()

        dist.barrier()
        tmp.unlink()
        tmp_grad.unlink()

    return SynchronizationAnalysisCallback

# fmt: on
