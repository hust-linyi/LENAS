import torch
import torch.nn as nn
from prim_ops import OPS, ConvOps
from genotype import Genotype
from model import SearchedNet
import pickle
import os

teachers = []
teachers_names = ['NAS_28', 'NAS_20', 'NAS_27', 'NAS_37']

model_dir = "YOUR_ROOT/RTDosePrediction/Teachers"

for name in teachers_names:
    model_path = os.path.join(model_dir, name) + '.pkl'
    geno_path = os.path.join(model_dir, name) + '_geno.pkl'

    with open(geno_path, 'rb') as f:
        gene = eval(pickle.load(f)[0])

        model = SearchedNet(in_channels=9, 
                            init_n_kernels=16, 
                            out_channels=1, 
                            depth=4, 
                            n_nodes=4, 
                            channel_change=True,
                            gene=gene)
        model.load_state_dict(torch.load(model_path)['network_state_dict'])
        model.cuda()
        
        teachers.append(model)
