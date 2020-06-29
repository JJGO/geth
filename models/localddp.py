from torch import nn

class LocalDDP(nn.Module):

    def __init__(self, module):
        self.module = module

    def forward(self, input):
        return self.module.forward(input)

    def average(self):
        # TODO Average parameters & buffers across replicas
       raise NotImplementedError("TODO")

       # All reduce params

       # All reduce buffers
