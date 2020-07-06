from os.path import join, exists, dirname, abspath, realpath
import subprocess, h5py, numpy as np, time, sys, h5py, pickle, argparse, json, shutil, os
import pandas as pd
from numpy.random import choice
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
from torch.autograd import Variable
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from os import system,chdir,getcwd,makedirs, listdir
from tempfile import mkdtemp
from sklearn.metrics import accuracy_score,roc_auc_score
from pprint import pprint
from time import time
from torch.utils.data import DataLoader
from torchsummary import summary
print(dirname(abspath("__file__")))
sys.path.append(dirname(abspath("__file__")))
import pytorch_lightning as pl
import bilm

from datasets.utils import *
from datasets.mhcpepata import  MHCPepDataset
from models.pepnet import Net


def cnt_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_setup():
    return {
               "run": "my_class",
               "repeat": 160,
               "trial_resources": {"cpu": 0, "gpu": 1},
               "config": {
                   "pep_layers": [5],
                   "pep_embed_size": 40,
                   "pep_len": 40,
                   "pep_block": "basic",
                   "pep_conv_fn": 256,
                   "mhc_embed_size": 40,
                   "mhc_len": 34,
                   "adam_lr": 1e-03,
                   "adam_beta1": 0.9,
                   "adam_beta2": 0.99,
                   "dense_size": 64,
                   "train_epoch_scale": 1,
                   "class_num": 1,
                   "batch_size": 128,
                   "mass_embed_size": 2560,
                   "mode": 'train'
               },
            }

class MHCPeptideClassifier(pl.LightningModule):

    def __init__(self, config):
        super(MHCPeptideClassifier, self).__init__()
        self.config = config
        self.net = Net(self.config)
        self.bin_loss = torch.nn.BCELoss()
        # summary(self.net.PEPnet, (self.config['pep_embed_size']+self.config['mhc_len']*self.config['mhc_embed_size'], self.config['pep_len']))
        print('num of params:', cnt_param(self.net))

    def prepare_data(self):
        print('Loading training data...')
        print('Train prefix : ', self.config['trainset_prefix'])
        self.trainset = MHCPepDataset(self.config['trainset_prefix'])
        print('Loading validation data...')
        print('Validation prefix : ', self.config['validset_prefix'])
        self.validset = MHCPepDataset(self.config['validset_prefix'])
 
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)

    # def test_dataloader(self):
    #     print('Loading test data..')
    #     print('Test prefix : ', self.config['testset_prefix'])
    #     return DataLoader(self.testset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config['adam_lr'], betas=(self.config['adam_beta1'], self.config['adam_beta2']))
        return self.optimizer

    def forward(self, mhc_inputs, pep_inputs, lenpep_inputs, elmos):
        return self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos)
      
    def training_step(self, batch, batch_idx):
        mhc_inputs, pep_inputs, lenpep_inputs, elmos, labels, relation, masslabels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        m, v, mass_pred = self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos)
        
        # relation_mapper = {'=':0, '<':1, '>':2}
        mass_pick = (masslabels != -1)
        real_relation = (m < labels) + 1  # 2 if mean < label, 1 otherwise
        affinity_pick = (labels!=-1) & (real_relation != relation)  # in relation, 2 if real (normalized) affinity  < label, 1 otherwise

        m, v, labels = m[affinity_pick], v[affinity_pick], labels[affinity_pick]
        if len(m)>0:
            normal_distr = Normal(m, v)
            aff_loss = -torch.mean(normal_distr.log_prob(labels))
        else:
            aff_loss = 0

        mass_pred, masslabels = mass_pred[mass_pick], masslabels[mass_pick]
        mass_loss = self.bin_loss(mass_pred, masslabels)

        loss = aff_loss + mass_loss
    
        # label_np = labels.cpu().detach().numpy()
        # label_bin = (label_np > 0.426).astype(float)
        # o_mean_np = m.cpu().detach().numpy()
        # try:
        #     t_spearmanr = spearmanr(label_np, o_mean_np)[0]
        # except ValueError as err:
        #     print ('fail to calculate train spearmanr:', err)

        # try:
        #     t_auc = roc_auc_score(label_bin, o_mean_np)
        # except ValueError as err:
        #     print ('fail to calculate train auc:', err)

        # masslabel_np = masslabels.cpu().detach().numpy()
        # masspred = mass_pred.cpu().detach().numpy()
        # try:
        #     t_mass_auc = roc_auc_score(masslabel_np, masspred)
        # except ValueError as err:
        #   print ('fail to calculate train mass auc:', err)

        tensorboard_logs = {'train_loss': loss}
        # 'train_auc': torch.Tensor(t_auc),
        # 'train_mass_auc': torch.Tensor(t_mass_auc),
        # 'train_spearmanr': torch.Tensor(t_spearmanr),
        return {
                'loss': loss,\
                'progress_bar': {'train_loss': loss},\
                'log': tensorboard_logs
              }

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        # train_auc_mean = torch.stack([x['train_auc'] for x in outputs]).mean()
        # train_mass_auc_mean = torch.stack([x['train_mass_auc'] for x in outputs]).mean()
        # train_spearmanr_mean += output['train_spearmanr']

        # log training accuracy at the end of an epoch
        # 'train_auc': train_auc_mean.item(), 
        # 'train_mass_auc': train_mass_auc_mean.item(),
        # 'train_spearmanr': train_spearmanr_mean.item()
        results = {
            'log': {'avg_train_loss': train_loss_mean.item()},
            'progress_bar': {'avg_train_loss': train_loss_mean.item()}
        }
        return results

    def validation_step(self, data, batch_idx):
        mhc_inputs, pep_inputs, lenpep_inputs, elmos, labels, relation, masslabels = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

        m, v, mass_pred = self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos) # both mbsize x label_dim

        mass_pick = masslabels != -1
        real_relation = (m < labels) + 1  # 2 if mean < label, 1 otherwise
        affinity_pick = (labels!=-1) & (real_relation != relation)  # in relation, 2 if real (normalized) affinity  < label, 1 otherwise

        m, v, labels = m[affinity_pick], v[affinity_pick], labels[affinity_pick]
        if len(m)>0:
            normal_distr = Normal(m, v)
            aff_loss = -torch.mean(normal_distr.log_prob(labels))
        else:
            aff_loss = 0

        mass_pred, masslabels = mass_pred[mass_pick], masslabels[mass_pick]
        mass_loss = self.bin_loss(mass_pred, masslabels)

        loss = aff_loss + mass_loss

        # label_np = labels.cpu().detach().numpy()
        # label_bin = (label_np > 0.426).astype(float)
        # o_mean_np = m.cpu().detach().numpy()
        # try:
        #   v_spearmanr = spearmanr(label_np, o_mean_np)[0]
        # except ValueError as err:
        #     print ('fail to calculate train spearmanr:', err)

        # try:
        #   v_auc = roc_auc_score(label_bin, o_mean_np)
        # except ValueError as err:
        #   print ('fail to calculate val auc:', err)

        # masslabel_np = masslabels.cpu().detach().numpy()
        # masspred = mass_pred.cpu().detach().numpy()
        # try:
        #   v_mass_auc = roc_auc_score(masslabel_np, masspred)
        # except ValueError as err:
        #   print ('fail to calculate val mass auc:', err)
        
        tensorboard_logs = {'val_loss': loss}
        # 'val_auc': torch.Tensor(v_auc),
        # 'val_mass_auc': torch.Tensor(v_mass_auc),
        # 'val_spearmanr': torch.Tensor(v_spearmanr),
        return {
                'val_loss': loss,
                'progress_bar': {'val_loss': loss},
                'log': tensorboard_logs
        }

    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # val_auc_mean = torch.stack([x['val_auc'] for x in outputs]).mean()
        # val_mass_auc_mean = torch.stack([x['val_mass_auc'] for x in outputs]).mean()
        # 'val_mass_auc': val_mass_auc_mean.item(),
        # 'val_auc': val_auc_mean.item()
        return {'val_loss': avg_loss,
                'progress_bar': {'avg_val_loss': avg_loss.item()}}
        
    # def predictClass(self, dataset):
    #     self.net.eval()
    #     for i, data in enumerate(dataset, 0):
    #         mhc_inputs, pep_inputs, lenpep_inputs, elmos = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device), data[3].to(self.device)
    #         m, v, mass_pred = self.net(mhc_inputs, pep_inputs, lenpep_inputs, elmos) # both mbsize x label_dim
    #         mean_y = m.cpu().detach().numpy()
    #         var_y = v.cpu().detach().numpy()
    #         mass_pred_y = mass_pred.cpu().detach().numpy()
    #         t_out = np.hstack((mean_y, var_y, mass_pred_y))
    #         out = t_out if i == 0 else np.vstack((out, t_out))
    #     return out

    # def saveModel(self, checkpoint_dir):
    #     path = join(checkpoint_dir, 'checkpoint.pt')
    #     torch.save(self.net, path)
    #     optim_path = join(checkpoint_dir, 'optim.pt')
    #     torch.save({'optimizer': self.optimizer.state_dict()}, optim_path)
    #     return path

    # def restoreModel(self, checkpoint_path):
    #     self.net = torch.load(checkpoint_path)
    #     self.optimizer = self.configure_optimizers()
    #     self.optimizer.load_state_dict(torch.load(join(dirname(checkpoint_path), 'optim.pt'))['optimizer'])

if __name__ == "__main__":
    pwd = dirname(realpath("__file__"))

    trainDir = join(pwd, "train")

    valDir = join(pwd, "val")
   
    allDataDir = join(pwd, "alldata")
    baseAllData = join(allDataDir, "CV")

<<<<<<< HEAD
    print("Splitting the data into train and validation sets..")
    CVSplit(baseAllData, trainDir, valDir)
    print("Splitting Done..")
    system(' '.join(['cp', join(dirname(abspath("__file__")), 'data/data.py'), join(bilm.__path__[0])]))
    print("Preprocessing Training data..")
    system(' '.join(['python {}/datasets/preprocess.py -o {}/trial -f {}/train/trainraw -a {}'.format(pwd, pwd, pwd, "train")]))
    print("Preprocessing validation data..")
=======
    CVsplit(baseAllData, trainDir, valDir)
    system(' '.join(['cp', join(dirname(abspath("__file__")), 'data.py'), join(bilm.__path__[0])]))
    system(' '.join(['python {}/datasets/preprocess.py -o {}/trial -f {}/train/trainraw -a {}'.format(pwd, pwd, pwd, "train")]))
>>>>>>> 1fe0117824f96cc0c92c123403990dcfc12ce561
    system(' '.join(['python {}/datasets/preprocess.py -o {}/trial -f {}/val/valraw -a {}'.format(pwd, pwd, pwd, "val")]))

    model_arch = 'mhccat2pep_pepres_relation_massspec_elmo_novar_v3_normal_noeps_bs1024_init1'
    outdir = join(join(pwd, "data"), model_arch)
    if not exists(outdir):
        makedirs(outdir)

    best_model_dir = join(outdir, 'best_model')
    evalout = join(outdir,  'best_model_eval.txt')
    best_trial_dir = realpath(join(outdir, 'best_trial_random'))
    historyfile = realpath(join(outdir, 'train.log'))
    print('Using model.py under {}'.format(realpath(os.environ['PYTHONPATH'])))

    with open(join(best_trial_dir, 'params.json')) as f:
            best_config = json.loads(f.readline())
    print('Best config:', best_config)

    for key, item in get_setup()['config'].items():
        if key not in best_config.keys():
            best_config[key] = item
            
    best_config["trainset_prefix"] = '/'.join([pwd, "trial", "train.h5.batch"])
    best_config["validset_prefix"] = '/'.join([pwd, "trial", "val.h5.batch"])

    model = MHCPeptideClassifier(config=best_config)
    outdir = join(pwd, 'trial')
    trainer = pl.Trainer(default_root_dir = outdir) 
    trainer.fit(model)
