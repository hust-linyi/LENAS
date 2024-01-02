from collections import namedtuple
import numpy as np
from prim_ops import DownOps, UpOps, NormOps, ConnectOps
import pdb

Genotype = namedtuple('Genotype', ['down','up'])
'''
Genotype saves the searched downward cell and upward cell
'''

class GenoParser:
    def __init__(self, n_nodes):
        '''
        This is the class for genotype operations.
        n_nodes: How many nodes in a cell.
        '''
        self.n_nodes = n_nodes
        
    def parse(self, alpha1, alpha2, alpha3, downward=True):
        '''
        alpha1: Weights for MixedOps with stride=1
        alpha2: Weights for MixedOps with stride=2
        Note these two matrix are softmaxed as same as in nas.KernelNet().
        Each MixedOp would keep the Op with the highest alpha value.
        For each node, two edges with the highest alpha values (throughout all stride==1 and stride==2 edges) 
        are kept as the inputs.
        '''
        gene = []
        res = []

        for i in range(self.n_nodes):
            if i < self.n_nodes - 2:
                argmax = np.argmax(alpha1[i])
                res.append((alpha1[i][argmax], NormOps[argmax]))
            elif i == self.n_nodes - 2:
                argmax = np.argmax(alpha2[i])
                res.append((alpha2[i][argmax]*len(ConnectOps)/len(NormOps), ConnectOps[argmax]))
            else:
                if downward:
                    argmax = np.argmax(alpha3[i])
                    res.append((alpha3[i][argmax]*len(DownOps)/len(NormOps), DownOps[argmax]))
                else:
                    argmax = np.argmax(alpha3[i])
                    res.append((alpha3[i][argmax]*len(UpOps)/len(NormOps), UpOps[argmax]))
            
            gene.append(res)

        gene.sort()
        print(gene[-1])

        return gene[-1]
