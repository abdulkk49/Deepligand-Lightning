from os.path import join, exists, dirname, abspath
import subprocess, h5py, numpy as np

class MHCPepDataset(torch.utils.data.Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        cnt = 1
        self.label = []
        self.masslabel = []
        self.mhc = []
        self.pep = []
        self.peplen = []
        self.relation = []
        self.elmo = []
        self.num_sample = 0
        while exists(prefix+str(cnt)):
            print('batch', cnt)

            with h5py.File(prefix+str(cnt),'r') as dataall:
                self.label.append(dataall['label'][()].astype(np.float32))
                self.masslabel.append(dataall['masslabel'][()].astype(np.float32))
                self.mhc.append(dataall['mhc'][()].astype(np.float32))
                self.pep.append(dataall['pep'][()].astype(np.float32))
                self.peplen.append(dataall['peplen'][()].astype(np.float32))
                self.relation.append(dataall['relation'][()].astype(np.uint8))
                self.elmo.append(dataall['elmo'][()].astype(np.float32))
            
            cnt += 1
            self.num_sample += len(self.label[-1])

        self.data_len = len(self.mhc)

    def __getitem__(self, index):
        bs = 50000
        cnt = index // bs
        i = index % bs
        return self.mhc[cnt][i], self.pep[cnt][i], self.peplen[cnt][i],self.elmo[cnt][i], self.label[cnt][i], self.relation[cnt][i], self.masslabel[cnt][i]

    def __len__(self):
        return self.num_sample