"""
Communication utils, based on the fairinternal/multi-agent-opt-pytorch implementation
"""

import collections
import torch
import torch.distributed as dist


def flatten_tensors(tensors):
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def group_by_dtype(tensors):
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def communicate(tensors, communication_op, group=dist.group.WORLD):
    """
    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for dtype in tensors_by_dtype:
        flat_tensor = flatten_tensors(tensors_by_dtype[dtype])
        communication_op(tensor=flat_tensor, group=group)
        for f, t in zip(
            unflatten_tensors(flat_tensor, tensors_by_dtype[dtype]),
            tensors_by_dtype[dtype],
        ):
            t.copy_(f)
