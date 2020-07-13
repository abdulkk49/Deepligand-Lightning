from os.path import join, exists, dirname, abspath
import subprocess, h5py, numpy as np
import torch
from os.path import join, exists, dirname, abspath
import subprocess, h5py, numpy as np
import torch
class MHCPepDataset(torch.utils.data.Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        cnt = 1
        self.num_sample = 0
        while exists(prefix+str(cnt)):
            with h5py.File(prefix+str(cnt),'r') as dataall:
                self.num_sample += dataall['mhc'].shape[0]
            cnt += 1

    def __getitem__(self, index):
        bs = 1024
        cnt = index // bs
        i = index % bs
        with h5py.File(self.prefix+str(cnt + 1),'r') as dataall:
            # print("batch num is " + str(cnt+1) )
            label  = dataall['label'][i,:].astype(np.float32)
            masslabel = dataall['masslabel'][i,:].astype(np.float32)
            mhc = dataall['mhc'][i,:,:].astype(np.float32)
            pep = dataall['pep'][i,:,:].astype(np.float32)
            peplen = dataall['peplen'][i,:].astype(np.float32)
            relation = dataall['relation'][i,:].astype(np.uint8)
            elmo = dataall['elmo'][i,:,:].astype(np.float32)
  
        return mhc, pep, peplen, elmo, label, relation, masslabel


    def __len__(self):
        return self.num_sample