import torch
from torch import nn
from torchvision.models import resnet


def resnet_goyal_init(model):
    """
    Initialize resnet50 similarly to "ImageNet in 1hr" paper
      - Batch norm moving average "momentum" <-- 0.9
      - Fully connected layer <-- Gaussian weights (mean=0, std=0.01)
      - gamma of last Batch norm layer of each residual block <-- 0
    """
    # Distributed training uses 4 tricks to maintain accuracy with much larger
    # batch sizes. See https://arxiv.org/pdf/1706.02677.pdf for more details

    assert isinstance(model, resnet.ResNet)

    for m in model.modules():
        # The last BatchNorm layer in each block needs to be initialized as zero gamma
        if isinstance(m, resnet.BasicBlock):
            num_features = m.bn2.num_features
            m.bn2.weight = nn.Parameter(torch.zeros(num_features))
        if isinstance(m, resnet.Bottleneck):
            num_features = m.bn3.num_features
            m.bn3.weight = nn.Parameter(torch.zeros(num_features))

        # Linear layers are initialized by drawing weights from a
        # zero-mean Gaussian with stddev 0.01. In the paper it was only for
        # fc layer, but in practice this seems to give better accuracy
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
    return model
