import torch
import torch.nn as nn
from prim_ops import OPS, ConvOps
# from helper import dim_assert
import pdb
from genotype import Genotype
import torch.nn.functional as F
from torch.autograd import Function

FLAG_DEBUG = False

class SearchedCell(nn.Module):
    def __init__(self, n_nodes, c0, c_node, gene, downward=True):
        '''
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        c_node: out_channels for each node.
        gene: Genotype, searched architecture of a cell
        downward: If True, this is a downward block, otherwise, an upward block.
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.c_node = c_node
        self.genolist = gene.down if downward else gene.up
        self.downward = downward
        
        
        if downward:
            self.preprocess0 = ConvOps(c0, c_node, kernel_size=3, stride=1, ops_order='act_weight_norm')
            
            self._ops = nn.ModuleList([OPS[i[1]](c_node) for i in self.genolist])
        else:
            self.preprocess0 = ConvOps(c0, c0//2, kernel_size=3, stride=1, ops_order='act_weight_norm')

            self._ops = nn.ModuleList([OPS[i[1]](c0//2) if i != len(self.genolist)-1 else OPS[i[1]](c_node) for i in self.genolist])


            self.down_channel = ConvOps(c0//2, c_node, kernel_size=1, stride=1, ops_order='act_weight_norm')

        
        return

    @property
    def out_channels(self):
        return self.n_nodes * self.c_node

    def forward(self, x0):

        x0 = self.preprocess0(x0) # cancel preprocess0
        x1 = x0
        i = 0

        for i in range(self.n_nodes-2):
            x0 = self._ops[i](x0)
        
        x1 = self._ops[-2](x1)
        
        output = self._ops[-1](sum([x0, x1]))

        if self.downward == False:
            output = self.down_channel(output)

        return output
        '''
        for i in range(self.n_nodes):
            x0 = self._ops[i](x0)
        
        
        #output = self._ops[-1](x0)

        return x0
        '''

class BottleNeck(nn.Module):
    def __init__(self, c0, c_node, upsample=False):

        super().__init__()
        #print(c0, "c0", c_node, "c_node")

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
            

class SearchedNet(nn.Module):
    def __init__(self, in_channels, init_n_kernels, out_channels, depth, n_nodes, channel_change,
                 gene):
        '''
        This class defines the U-shaped architecture. I take it as the kernel of NAS. 
        in_channels: How many kinds of MRI modalities being used.
        init_n_kernels: Number of kernels for the nodes in the first cell.
        out_channels: How many kinds of tumor labels.
        depth: Number of downward cells. For upward, it has depth+1 cells.
        n_nodes: Number of nodes in each cell.
        channel_change: If True, channel size expands and shrinks in double during downward and upward forwarding.
        gene: searched cell.
        '''
        super().__init__()
        c0 = init_n_kernels # channel0, channel1, the number of kernels.
        c_node = init_n_kernels

        self.fc_out = True
        self.pool_out = 'max'
        self.out_dims = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
        self.fc_layers = self._make_fc()

        self.stem0 = ConvOps(in_channels, c0, kernel_size=3, ops_order='weight_norm')
        self.single_conv = nn.Conv3d(c0, c0, kernel_size=3, padding=1)

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()
        self.bottle = nn.ModuleList()

        down = [c0]
        for i in range(depth):
            c_node = 2 * c_node if channel_change else c_node  # double the number of filters
            down_cell = SearchedCell(n_nodes, c0, c_node, gene)

            self.down_cells += [down_cell]
            
            c0 = c_node

            down.append(c_node)
        
        #self.bottle += [BottleNeck(c0, c_node)]

        c_node //= 2
        
        #down.pop()

        for i in range(depth):
            #print(c0, c_node, down[-1])
            up_cell = SearchedCell(n_nodes, c0+down.pop(), c_node, gene, downward=False)
            #print(up_cell)
            self.up_cells += [up_cell]
            c0 = c_node
            c_node = c_node // 2 if channel_change else c_node  # halve the number of filters
            
        self.last_conv = nn.Sequential(nn.Conv3d(c0*2, c0, kernel_size=1, padding=0, bias=True), nn.Conv3d(c0, 1, kernel_size=1, padding=0, bias=True))

    def _make_fc(self):
        if self.pool_out == "avg":
            layers = [
                nn.AdaptiveAvgPool1d(output) for output in self.out_dims
            ]
        elif self.pool_out == "max":
            layers = [
                nn.AdaptiveMaxPool1d(output) for output in self.out_dims
            ]
        return nn.Sequential(*layers)

    def _add_feature(self, x, feature_maps, fc_layer=False):
        if self.fc_out:
            out = self.fc_layers[fc_layer](x.view(x.size(0), x.size(1), -1))
            if self.pool_out == "max":
                out, _ = out.max(dim=1)
            else:
                out = out.mean(dim=1)
            fc_layer += 1
            feature_maps.append(out)
        else:
            feature_maps.append(x.view(x.size(0), -1))                                  

    def forward(self, x):
        feature_maps = []

        s0 = self.stem0(x)
        s0 = self.single_conv(s0)
        s0 = self.single_conv(s0)
        s = [s0]
        down_outputs = [s0] # 16,128,128,128

        self._add_feature(s0, feature_maps, 0)
        # print(s0.view(s0.size(0), -1).shape)

        for i, cell in enumerate(self.down_cells):
            s0 = cell(s0)
            self._add_feature(s0, feature_maps, i+1)
            # print(s0.view(s0.size(0), -1).shape)
            # feature_maps.append(s0.view(s0.size(0), -1))    
            s.append(s0)
        if FLAG_DEBUG:
            print('x.shape = ',x.shape)
            for i in down_outputs: 
                print(i.shape)

        for i, bot in enumerate(self.bottle):
            s0 = bot(s0)
            # print(s0.view(s0.size(0), -1).shape)
            self._add_feature(s0, feature_maps, i + len(self.down_cells) +1)
            # print(s0.shape)
            # feature_maps.append(s0.view(s0.size(0), -1))
        
        #print(len(self.up_cells))
        for i, cell in enumerate(self.up_cells):
            s1 = s.pop()
            s0 = cell(torch.cat([s0, s1], 1))
            # print(s0.view(s0.size(0), -1).shape)
            self._add_feature(s0, feature_maps, i + len(self.down_cells) + len(self.bottle) +1)
            # print(s0.shape)
            # feature_maps.append(s0.view(s0.size(0), -1))
            if FLAG_DEBUG:
                print(s0.shape)

        s1 = s.pop()
        output = self.last_conv(torch.cat([s0,s1], 1))
        self._add_feature(output, feature_maps, len(self.down_cells) + len(self.bottle) + len(self.up_cells) +1)

        return output, feature_maps
        
        # What is fc_out?


class GradReverse(Function):
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        return (-grad_output)

def grad_reverse(x):
    return GradReverse()(x)

class Discriminator(nn.Module):
    def __init__(self, outputs_size, K = 2):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv2 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv3 = nn.Conv2d(in_channels=outputs_size, out_channels=2, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = x[:,:,None,None]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = nn.Sigmoid()(out)
        return out

class Discriminators(nn.Module):
    def __init__(self, output_dims=[1000,1000,1000,1000,1000,1000,1000,1000,1000,1000], grl=False):
        super(Discriminators, self).__init__()
        #self.discriminators = [Discriminator(i) for i in output_dims]
        self.discriminators = nn.ModuleList([Discriminator(i) for i in output_dims])
        self.grl = grl
    
    def forward(self, x):
        if self.grl == True:
            out = [self.discriminators[i](grad_reverse(x[i])) for i in range(len(self.discriminators))]
        else:
            out = [self.discriminators[i](x[i]) for i in range(len(self.discriminators))]

        return out