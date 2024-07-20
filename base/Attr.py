import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    embed_dims = [('driverID', 24000, 16), ('weekID', 7, 3), ('timeID', 1440, 8)]

    def __init__(self):
        super(Net, self).__init__()
        print("Model:: def __init__(self):")
        # whether to add the two ends of the path into Attribute Component
        self.build()
        print("Model22:: def __init__(self):")

    def build(self):
        for name, dim_in, dim_out in Net.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        print("Model333:: def __init__(self):")
        sz = 0
        for name, dim_in, dim_out in Net.embed_dims:
            sz += dim_out
        # append total distance
        return sz + 1

    def forward(self, attr):
        em_list = []
        print("Model444:: def __init__(self):")
        for name, dim_in, dim_out in Net.embed_dims:
            print("Model555:: def __init__(self):")
            embed = getattr(self, name + '_em')
            print("Model666:: def __init__(self):")
            attr_t = attr[name].view(-1, 1)
            print("Model777:: def __init__(self):")
            attr_t = torch.squeeze(embed(attr_t))
            print("Model888:: def __init__(self):")
            em_list.append(attr_t)
            print("Model999:: def __init__(self):")
        dist = utils.normalize(attr['dist'], 'dist')
        em_list.append(dist.view(-1, 1))

        return torch.cat(em_list, dim = 1)
