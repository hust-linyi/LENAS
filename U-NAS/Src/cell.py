import torch
import torch.nn as nn
from prim_ops import OPS, DownOps, UpOps, NormOps, ConvOps
# from helper import dim_assert
import pdb


class MixedOp(nn.Module):
    def __init__(self, channels, stride, isConnect=False, transposed=False):
        '''
        channels: in_channels == out_channels for MixedOp
        '''
        super().__init__()
        self._ops = nn.ModuleList()
        self.stride = stride
        self.isConnect = isConnect
        if stride == 1:
            primitives = NormOps
        else:
            primitives = UpOps if transposed else DownOps
        for pri in primitives:
            op = OPS[pri](channels)
            self._ops.append(op)

    def forward(self, x, alpha1, alpha2, alpha3):

        if self.stride == 1:
            if self.isConnect == True:
                res = sum([w * op(x) for w, op in zip(alpha2, self._ops)])
            else:
                res = sum([w * op(x) for w, op in zip(alpha1, self._ops)]) # debug: dim_assert
        else:
            res = sum([w * op(x) for w, op in zip(alpha3, self._ops)]) # debug: dim_assert
        return res

class Cell(nn.Module):
    #def __init__(self, n_nodes, c0, c1, c_node, downward=True):
    def __init__(self, n_nodes, c0, c_node, downward=True):
        '''
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        c_node: out_channels for each node.
        downward: If True, this is a downward block, otherwise, an upward block.
        '''
        super().__init__()

        self.n_nodes = n_nodes
        self.c_node = c_node

        #print(c0, "c0", c_node, "c_node")
        self.preprocess0 = ConvOps(c0, c_node, kernel_size=3, stride=1, ops_order='act_weight_norm')
        self._ops = nn.ModuleList()

        for _ in range(n_nodes-2):
            self._ops.append(MixedOp(c_node, stride=1))
        
        self._ops.append(MixedOp(c_node, stride=1, isConnect=True))

        if downward:
            self._ops.append(MixedOp(c_node, stride=2))
        else:
            self._ops.append(MixedOp(c_node, stride=2, transposed=True))

        #print(self._ops)


    @property
    def out_channels(self):
        return self.n_nodes * self.c_node

    #def forward(self, x0, x1, alpha1, alpha2):
    def forward(self, x0, alpha1, alpha2, alpha3):
        '''
        x0, x1: Inputs to a cell
        alpha1: Weights for MixedOp with stride == 1
        alpha2: Weights for MixedOp with stride == 2
        '''
        
        x0 = self.preprocess0(x0)
        
        x1 = x0

        #print(alpha1, alpha2, "cell")

        
        for i in range(self.n_nodes-2):
            x0 = self._ops[i](x0, alpha1[i], alpha2[i], alpha3[i])
        
        x1 = self._ops[-2](x0, alpha1[-2], alpha2[-2], alpha3[-2])
        
        
        output = self._ops[-1](sum([x0, x1]), alpha1[-1], alpha2[-1], alpha3[-1])

        return output
        
        '''
        for i in range(self.n_nodes):
            x0 = self._ops[i](x0, alpha1[i], alpha2[i])
        
        return x0
        '''


class BottleNeck(nn.Module):
    def __init__(self, c0, c_node, upsample=False):

        super().__init__()
        print(c0, "c0", c_node, "c_node")

        self.conv1 = ConvOps(c0, c0)
        self.conv2 = ConvOps(c0, c_node)
        if upsample:
            self.conv3 = ConvOps(c_node, c_node, stride=2)
        else:
            self.conv3 = ConvOps(c_node, c_node)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x