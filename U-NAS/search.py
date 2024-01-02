import pdb
import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
from loss import L1Loss, Diversityloss
from helper import calc_param_size, print_red
from nas import ShellNet
import sys
from torch.optim import Adam
from adabound import AdaBound
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
import pickle
import shutil
from dataloader import get_loader, read_data, val_transform, pre_processing
import numpy as np
from evaluate import *
from searched import SearchedNet
from genotype import Genotype


sys.path.append('./NetworkTrainer')
from NetworkTrainer.network_trainer import *


DEBUG_FLAG = False

class Base:
    '''
    Base class for Searching and Training
    jupyter: if True, run in Jupyter Notebook, otherwise in shell.
    for_search: if True, for search, otherwise for training. Notice patch_search could be different from patch_training.
    for_final_training: if False, for k-fold-cross-val, otherwise final training will use the whole training dataset.
    '''
    def __init__(self, jupyter=True, for_search=True, for_final_training=False):
        self.jupyter = jupyter
        self.for_search = for_search
        self.for_final_training = for_final_training
        self._init_config()
        self._init_log()
        self._init_device()
        #self._init_dataset()
    
    def _init_log(self):
        try:
            os.mkdir(self.config['search']['log_path'])
        except FileExistsError:
            pass
        
    def _init_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',type=str,default='config.yml',
                            help='Configuration file to use')
        if self.jupyter: # for jupyter notebook
            self.args = parser.parse_args(args=[])
        else:  # for shell
            self.args = parser.parse_args()
        with open(self.args.config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        print('data[patch_overlap] =', self.config['data']['patch_overlap'])
        print('search[patch_shape] =', self.config['search']['patch_shape'])
        print('train[patch_shape] =', self.config['train']['patch_shape'])
        print('train[epochs] =', self.config['train']['epochs'])
        print('data[inclusive_label] =', self.config['data']['inclusive_label'])
        print('data[both_ps] =', self.config['data']['both_ps'])
        return
        
    def _init_device(self):
        if self.config['search']['gpu'] and torch.cuda.is_available() :
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        else:
            print_red('No gpu devices available!, we will use cpu')
            self.device = torch.device('cpu')
        return


class Searching(Base):
    '''
    Searching process
    jupyter: if True, run in Jupyter Notebook, otherwise in shell.
    new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
    '''
    def __init__(self, jupyter=False, new_lr=False):
        super().__init__(jupyter=jupyter)
        self._init_model()
        self.check_resume(new_lr=new_lr)
        self.batch_size_train=self.config['data']['batch_size_train']
        self.batch_size_val=self.config['data']['batch_size_val']
        self.train_epoch, self.val_epoch = get_loader()
        
    
    def _init_model(self):
        self.model = ShellNet(in_channels=9, 
                              init_n_kernels=self.config['search']['init_n_kernels'], 
                              out_channels=1, 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'],
                              normal_w_share=self.config['search']['normal_w_share'], 
                              channel_change=self.config['search']['channel_change']).to(self.device)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))
        self.L1loss = L1Loss().to(self.device)
        self.divloss = Diversityloss().to(self.device)

        self.optim_shell = Adam(self.model.alphas()) # lr=3e-4
        self.optim_kernel = Adam(self.model.kernel.parameters())
        self.shell_scheduler = CosineAnnealingLR(self.optim_shell, T_max=self.config['search']['epochs'], eta_min=1e-7)
        self.kernel_scheduler = CosineAnnealingLR(self.optim_kernel, T_max=self.config['search']['epochs'], eta_min=1e-7)

    def check_resume(self, new_lr=False):
        self.last_save = self.config['search']['last_save']
        self.best_shot = self.config['search']['best_shot']
        if os.path.exists(self.last_save):
            state_dicts = torch.load(self.last_save, map_location=self.device)
            self.epoch = state_dicts['epoch'] + 1
            self.geno_count = state_dicts['geno_count']
            self.history = state_dicts['history']
            self.model.load_state_dict(state_dicts['model_param'])
            if not new_lr:
                self.optim_shell.load_state_dict(state_dicts['optim_shell'])
                self.optim_kernel.load_state_dict(state_dicts['optim_kernel'])
                self.shell_scheduler.load_state_dict(state_dicts['shell_scheduler'])
                self.kernel_scheduler.load_state_dict(state_dicts['kernel_scheduler'])
            self.best_val_loss = state_dicts['best_loss']
        else:
            self.epoch = 0
            self.geno_count = Counter()
            self.history = defaultdict(list)
            self.best_val_loss = 1.0

    def search(self):
        '''
        Return the best genotype in tuple:
        (best_gene: str(Genotype), geno_count: int)
        '''
        geno_file = self.config['search']['geno_file']
        if os.path.exists(geno_file):
            print('{} exists.'.format(geno_file))
            with open(geno_file, 'rb') as f:
                return pickle.load(f)

        best_gene = None
        best_geno_count = self.config['search']['best_geno_count']
        n_epochs = self.config['search']['epochs']
        for epoch in range(n_epochs):
            is_best = False
            gene = self.model.get_gene()
            self.geno_count[str(gene)] += 1
            if self.geno_count[str(gene)] >= best_geno_count:
                print('>= best_geno_count: ({})'.format(best_geno_count))
                best_gene = (str(gene), best_geno_count)
                break

            shell_loss, kernel_loss = self.train()
            val_loss = self.validate(epoch)
            self.shell_scheduler.step(shell_loss)
            self.kernel_scheduler.step(val_loss)
            self.history['shell_loss'].append(shell_loss)
            self.history['kernel_loss'].append(kernel_loss)
            self.history['val_loss'].append(val_loss)
            
            if val_loss < self.best_val_loss:
                is_best = True
                self.best_val_loss = val_loss
            
            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'geno_count': self.geno_count,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim_shell': self.optim_shell.state_dict(),
                'optim_kernel': self.optim_kernel.state_dict(),
                'kernel_scheduler': self.kernel_scheduler.state_dict(),
                'shell_scheduler': self.kernel_scheduler.state_dict(),
                'best_loss': self.best_val_loss
            }
            torch.save(state_dicts, self.last_save)
            
            if is_best:
                shutil.copy(self.last_save, self.best_shot)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch >= 1:
                break
                
        if best_gene is None:
            gene = str(self.model.get_gene())
            self.geno_count[gene] += 1
            best_gene = (gene, self.geno_count[gene])
        with open(geno_file, 'wb') as f:
            pickle.dump(best_gene, f)
        return best_gene
        
    
    def train(self):
        '''
        Searching | Training process
        To do optim_shell.step() and optim_kernel.step() in turn.
        '''
        self.model.train()
        y = self.model(torch.rand(1, 9, 128, 128, 128).cuda())
        n_steps = 50
        sum_loss = 0
        sum_val_loss = 0

        geno_file = 'ref_genotype.pkl'
        with open(geno_file, 'rb') as f:
            gene = eval(pickle.load(f)[0])

        with torch.no_grad():
            ref = SearchedNet(in_channels=9, 
                            init_n_kernels=16, 
                            out_channels=1, 
                            depth=4, 
                            n_nodes=4,
                            channel_change=True,
                            gene=gene)
        
            ref.load_state_dict(torch.load('best_val_evaluation_index.pkl', map_location='cpu')['network_state_dict'])
            ref = ref.cuda()

        with tqdm(self.train_epoch, total = n_steps,
                  desc = 'Searching | Epoch {} | Training'.format(self.epoch)) as pbar:
            for step, list_loader_output in enumerate(pbar):
                x = list_loader_output[0]
                y_truth = list_loader_output[1:]
                y_truth = torch.Tensor(np.array([item.numpy() for item in y_truth]))

                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)
                try:
                    val_list = next(iter(self.val_epoch))
                    val_x, val_y_truth = val_list[0], val_list[1:]
                    val_y_truth = torch.Tensor(np.array([item.numpy() for item in val_y_truth]))

                except StopIteration:
                    pass

                val_x = torch.as_tensor(val_x, device=self.device, dtype=torch.float)
                val_y_truth = torch.as_tensor(val_y_truth, device=self.device, dtype=torch.float)

                # optim_shell
                ref_y_pred = ref(val_x)

                self.optim_shell.zero_grad()
                val_y_pred = self.model(val_x)
                val_loss = self.L1loss(val_y_pred, val_y_truth) + 10*self.divloss(val_y_pred, ref_y_pred)
                sum_val_loss += val_loss.item()
                val_loss.backward()
                self.optim_shell.step()
                
                # optim_kernel
                self.optim_kernel.zero_grad()
                y_pred = self.model(x)
                y_ref = ref(x)
                loss = self.L1loss(y_pred, y_truth) + 10*self.divloss(y_pred, y_ref)
                print(self.divloss(y_pred, y_ref))
                sum_loss += loss.item()
                loss.backward()
                self.optim_kernel.step()
                
                # postfix for progress bar
                postfix = OrderedDict()
                postfix['Loss(optim_shell)'] = round(sum_val_loss/(step+1), 3)
                postfix['Loss(optim_kernel)'] = round(sum_loss/(step+1), 3)
                pbar.set_postfix(postfix)
                
                if DEBUG_FLAG and step > 1:
                    break
                
        return round(sum_val_loss/n_steps, 3), round(sum_loss/n_steps, 3)
    
    def validate(self, epoch):
        '''
        Searching | Validation process
        '''
        sum_loss = 0
        list_patient_dirs = [os.path.join(self.config['data']['source_val'], 'pt_'+str(i)) for i in range(201,241)]
        list_Dose_score = []
        with torch.no_grad():
            self.model.eval()
            for patient_dir in list_patient_dirs:
                dict_images = read_data(patient_dir)
                list_images = pre_processing(dict_images)

                x = list_images[0]
                [x] = val_transform([x])

                y_truth = list_images[1]
                possible_dose_mask = list_images[2]

                y_loss = list_images[1:]

                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                x = x.unsqueeze(0)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)
                possible_dose_mask = torch.as_tensor(possible_dose_mask, device=self.device, dtype=torch.float)
                y_loss = torch.as_tensor(y_loss, device=self.device, dtype=torch.float)
                y_loss = y_loss.unsqueeze(1)

                y_pred = self.model(x)
                
                loss = self.L1loss(y_pred, y_loss)
                sum_loss += loss.item()


                x = np.array(x.cpu().data[0, :, :, :, :])
                y_pred = np.array(y_pred.cpu().data[0, :, :, :, :])
                y_truth = np.array(y_truth.cpu().data)
                possible_dose_mask = np.array(possible_dose_mask.cpu().data)


                # Post processing and evaluation
                y_pred[np.logical_or(possible_dose_mask < 1, y_pred < 0)] = 0

                Dose_score = 70. * get_3D_Dose_dif(y_pred, y_truth,
                                                possible_dose_mask)

                list_Dose_score.append(Dose_score)
        
        print('===============> mean Dose score: '
                                    + str(np.mean(list_Dose_score)))
        print('===============> sum loss:', round(sum_loss/40, 3))

        return round(sum_loss/40, 3)

    
if __name__ == '__main__':
    searching = Searching(jupyter = False)
    gene = searching.search()