import pdb
import os
import torch
import torch.nn as nn
from loss import L1Loss
from helper import calc_param_size
from searched import SearchedNet
from torch.optim import Adam
from adabound import AdaBound
from torch.optim.lr_scheduler import CosineAnnealingLR
# from tqdm import tqdm
from tqdm import tqdm
from collections import defaultdict
import pickle
from genotype import Genotype
import shutil
from search import Base
from dataloader import get_loader, read_data, val_transform, pre_processing
import numpy as np
from evaluate import *

from torchsummary import summary

from torchviz import make_dot

DEBUG_FLAG = False

    
class Training(Base):
    '''
    Training the searched network
    jupyter: if True, run in Jupyter Notebook, otherwise in shell.
    for_final_training: if False, for k-fold-cross-val, otherwise final training will use the whole training dataset.
    new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
    '''
    def __init__(self, jupyter=True, for_final_training=False, new_lr=False):
        super().__init__(jupyter=jupyter, for_search=False, for_final_training=for_final_training)
        self._init_model()
        self.check_resume(new_lr=new_lr)
        self.train_epoch, self.val_epoch = get_loader()
    
    def _init_model(self):
        geno_file = self.config['search']['geno_file']
        with open(geno_file, 'rb') as f:
            gene = eval(pickle.load(f)[0])
        self.model = SearchedNet(in_channels=9, 
                              init_n_kernels=self.config['train']['init_n_kernels'], 
                              out_channels=1, 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'],
                              channel_change=self.config['search']['channel_change'],
                              gene=gene).to(self.device)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))
        self.loss = L1Loss().to(self.device)

        self.optim = Adam(self.model.parameters())
        self.scheduler = CosineAnnealingLR(self.optim, T_max=self.config['train']['epochs'], eta_min=1e-7)
        

    def check_resume(self, new_lr=False):
        self.last_save = self.config['train']['last_save']
        self.best_shot = self.config['train']['best_shot']
        if os.path.exists(self.last_save):
            state_dicts = torch.load(self.last_save, map_location=self.device)
            self.epoch = state_dicts['epoch'] + 1
            self.history = state_dicts['history']
            self.model.load_state_dict(state_dicts['model_param'])
            if not new_lr:
                self.optim.load_state_dict(state_dicts['optim'])
                self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.best_val_loss = state_dicts['best_loss']
        else:
            self.epoch = 0
            self.history = defaultdict(list)
            self.best_val_loss = 1.0

    def main_run(self):
        n_epochs = self.config['train']['epochs']
        
        for epoch in range(n_epochs):
            is_best = False
            loss = self.train()
            val_loss = self.validate(epoch)
            self.scheduler.step(val_loss)
            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            if val_loss < self.best_val_loss:
                is_best = True
                self.best_val_loss = val_loss
            
            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
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
        print('Training Finished.')
        return 
        
    
    def train(self):
        '''
        Training | Training process
        '''
        self.model.train()
        #n_steps = self.train_generator.steps_per_epoch
        n_steps = 50
        sum_loss = 0

        summary(self.model, (9, 128, 128, 128))

        y = self.model(torch.rand(1, 9, 128, 128, 128).cuda())
        g = make_dot(y)
        g.render('espnet_model', view=False)

        with tqdm(self.train_epoch, total = n_steps,
                  desc = 'Training | Epoch {} | Training'.format(self.epoch)) as pbar:
            for step, list_loader_output in enumerate(pbar):
                x = list_loader_output[0]
                y_truth = list_loader_output[1:]
                y_truth = torch.Tensor(np.array([item.numpy() for item in y_truth]))

                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)

                self.optim.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(),
#                                          self.config['search']['grad_clip'])
                self.optim.step()
                
                pbar.set_postfix(Loss=round(sum_loss/(step+1), 3))
                
                if DEBUG_FLAG and step >= 1:
                    break
                
        return round(sum_loss/n_steps, 3)
    
    
    def validate(self, epoch):
        '''
        Training | Validation process
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
                #y_loss = torch.Tensor(np.array([item.numpy() for item in y_loss]))
                #print(y_loss[0].shape)

                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                x = x.unsqueeze(0)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.float)
                possible_dose_mask = torch.as_tensor(possible_dose_mask, device=self.device, dtype=torch.float)
                y_loss = torch.as_tensor(y_loss, device=self.device, dtype=torch.float)
                y_loss = y_loss.unsqueeze(1)

                y_pred = self.model(x)
                
                loss = self.loss(y_pred, y_loss)
                sum_loss += loss.item()

                
                if epoch % 1 == 0:
                    x = np.array(x.cpu().data[0, :, :, :, :])
                    y_pred = np.array(y_pred.cpu().data[0, :, :, :, :])
                    y_truth = np.array(y_truth.cpu().data)
                    possible_dose_mask = np.array(possible_dose_mask.cpu().data)


                    # Post processing and evaluation
                    y_pred[np.logical_or(possible_dose_mask < 1, y_pred < 0)] = 0

                    # print(y_pred, y_truth)
                    Dose_score = 70. * get_3D_Dose_dif(y_pred, y_truth,
                                                    possible_dose_mask)

                    # patient_name = patient_dir.split('/')[-1]
                    # print(patient_name, Dose_score)
                    list_Dose_score.append(Dose_score)

                

        if epoch % 1 == 0:
            print('===============> mean Dose score: '
                                        + str(np.mean(list_Dose_score)))
        print('===============> sum loss:', round(sum_loss/40, 3))

        return round(sum_loss/40, 3)

    
if __name__ == '__main__':
    training = Training(jupyter = False)
    training.main_run()


