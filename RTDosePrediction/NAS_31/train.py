# -*- encoding: utf-8 -*-
import os
import sys
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

import argparse

from DataLoader.dataloader_OpenKBP_C3D import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer
from searched import SearchedNet
from online_evaluation import online_evaluation
from loss import Loss
import pickle
from genotype import Genotype
from helper  import calc_param_size
# from torchviz import make_dot
# import torch 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[1, 0],
                        help='list_GPU_ids for training (default: [1, 0])')
    parser.add_argument('--max_iter',  type=int, default=80000,
                        help='training iterations(default: 80000)')
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'NAS_31'
    trainer.setting.output_dir = 'YOUR_ROOT/Experiment/RTDosePrediction/Output/NAS_31'
    list_GPU_ids = args.list_GPU_ids

    geno_file = 'best_genotype.pkl'
    with open(geno_file, 'rb') as f:
        gene = eval(pickle.load(f)[0])

    trainer.setting.network = SearchedNet(in_channels=9, 
                              init_n_kernels=16, 
                              out_channels=1, 
                              depth=4, 
                              n_nodes=4,
                              channel_change=True,
                              gene=gene)
    
    '''
    trainer.setting.network.cuda()
    y = trainer.setting.network(torch.rand(1, 9, 128, 128, 128).cuda())
    g = make_dot(y)
    g.render('model', view=False)
    '''
    
    print('Param size = {:.3f} MB'.format(calc_param_size(trainer.setting.network)))

    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=4
    )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 3e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
