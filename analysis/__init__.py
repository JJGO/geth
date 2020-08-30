def effective_batch_size(batch_size, world_size):
    return batch_size * world_size


def norm_lr(lr, effective_batch_size):
    return lr / effective_batch_size
