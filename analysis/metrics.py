from functools import lru_cache
import json
import pandas as pd
import importlib.resources

with importlib.resources.path("geth.measurements", "times_per_epoch.csv") as p:
    measured_times_per_epoch = pd.read_csv(p)

PARAMS_MODEL = {"resnet50": 25557032, "resnet101": 44549160}

SAMPLES_DATASET = {
    "ImageNet": 1281167,
    "TinyImageNet": 100000,
}


def effective_batch_size(batch_size, world_size):
    return batch_size * world_size


def norm_lr(lr, effective_batch_size):
    return lr / effective_batch_size


def compute_time(changes, epochs, times, init_val=1):
    """
    input:
        changes     List[Tuple]
            Sequence of changes in #steps between synchs,
            applied at the beggining of the epochs
            [(epoch_i, #steps between synchs)]
        epochs      int
            Number of epochs
        times       Dict[int:int]
            Time per epoch for each T
        init_val    int
            Default initial value for changes.
            Overwritten if the first entry in changes is (0, _)
    output:
        synchs      int
    """
    assert epochs > 0, f"epochs={epochs} should be >0"
    assert init_val > 0, f"init_val={init_val} should be >0"

    if len(changes) == 0:
        changes = [(0, init_val)]
    elif changes[0][0] != 0:
        changes = [(0, init_val)] + changes

    epoch_diffs = [changes[i][0] - changes[i - 1][0] for i in range(1, len(changes))]
    epoch_diffs.append(epochs - changes[-1][0])

    return sum([d * times[t] for d, t in zip(epoch_diffs, [x for _, x in changes])])


def compute_synchs(changes, epochs, steps, init_val=1):
    """
    input:
        changes     List[Tuple]
            Sequence of changes in #steps between synchs,
            applied at the beggining of the epochs
            [(epoch_i, #steps between synchs)]
        epochs      int
            Number of epochs
        steps       int
            Number of steps per epoch
        init_val    int
            Default initial value for changes.
            Overwritten if the first entry in changes is (0, _)
    output:
        synchs      int
    """
    assert epochs > 0, f"epochs={epochs} should be >0"
    assert steps > 0, f"steps={steps} should be >0"
    assert init_val > 0, f"init_val={init_val} should be >0"

    if len(changes) == 0:
        d = steps * epochs
        return d // init_val + (d % init_val > 0)

    elif changes[0][0] != 0:
        d, t = steps * changes[0][0], init_val
        total = d // t + (d % t > 0)
    else:
        total = 0

    epoch_diffs = [
        steps * (changes[i][0] - changes[i - 1][0]) for i in range(1, len(changes))
    ]
    epoch_diffs.append(steps * (epochs - changes[-1][0]))
    total += sum(
        d // t + (d % t > 0) for d, t in zip(epoch_diffs, [x for _, x in changes])
    )
    return total


def n_syncs(dataset, freq_switch, epochs, effective_batch_size):

    return compute_synchs(
        [[0, 1], *freq_switch], epochs, SAMPLES_DATASET[dataset] / effective_batch_size
    )


# def n_params(model):
#     return PARAMS_MODEL.get(model, None)


@lru_cache(maxsize=None)
def n_params(path):
    summary_path = path / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            n = json.load(f)["total_params"]
        return n
    return None


def frequency(frequency, freq_switch):
    if pd.isna(freq_switch):
        return (frequency,)
    #         return frequency
    return tuple([f for _, f in freq_switch])


def switch_local(switch_local, freq_switch):
    if pd.isna(freq_switch):
        return (switch_local,)
    #         return switch_local
    return tuple([s for s, _ in freq_switch])


def freq_switch(freq_switch, frequency, switch_local):
    if not pd.isna(freq_switch):
        return freq_switch
    return tuple([(s, f) for s, f in zip(switch_local, frequency)])


def communication_cost(n_params, n_syncs):
    return n_params * n_syncs * 2


@lru_cache(maxsize=None)
def time_per_epoch(model, world_size, frequency):
    df = measured_times_per_epoch
    q = df.query(
        f"model == '{model}' & world_size == {world_size} & frequency == {frequency}"
    )
    return q.t_epoch.median()


def wall_time(model, world_size, freq_switch, epochs, frequency, switch_local):
    if freq_switch is None:
        freq_switch = [(switch_local, frequency)]
    try:
        times = {
            f: time_per_epoch(model, world_size, f)
            for f in [1] + [ff for _, ff in freq_switch]
        }
        return compute_time(list(map(list, freq_switch)), epochs, times)
    except KeyError:
        return None


def speedup(wall_time, model, world_size, epochs):
    baseline = time_per_epoch(model, world_size, 1) * epochs
    return baseline / wall_time


def batch_sync(batch_size, world_size, frequency):
    if isinstance(frequency, (int, float)):
        return batch_size * world_size * frequency
    if len(frequency) > 0:
        return -1
    return batch_size * world_size * frequency[0]


def folder(path):
    return path.parent.name


def filters(model, filters):
    if model.startswith("fresnet") and pd.isna(filters):
        return (64, 128, 256, 512)
    return filters
