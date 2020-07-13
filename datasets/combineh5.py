import h5py, sys, argparse
from os.path import join, exists, dirname, abspath, realpath
sys.path.append(dirname(abspath("__file__")))
from os.path import join, exists, dirname, abspath
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-t", "--dt", required=True)
    return parser.parse_args()


args = parse_args()
dt = args.dt
    
pwd = dirname(realpath("__file__"))
print(pwd)
direc = join(pwd, "trial")
with h5py.File(join(direc, dt + "all.h5"), 'w') as f:
    count = 1
    row1 = 0
    prefix = join(direc, dt + ".h5.batch")
    print
    while exists(prefix+str(count)):
        h5fr = h5py.File(prefix+str(count),'r')
        label = h5fr['label'][()].astype(np.float32)
        masslabel= h5fr['masslabel'][()].astype(np.float32)
        mhc = h5fr['mhc'][()].astype(np.float32)
        pep = h5fr['pep'][()].astype(np.float32)
        peplen = h5fr['peplen'][()].astype(np.float32)
        relation = h5fr['relation'][()].astype(np.uint8)
        elmo = h5fr['elmo'][()].astype(np.float32)
        dslen = mhc.shape[0]
        if row1 == 0: 
            f.create_dataset('mhc',shape=(dslen, mhc.shape[1], mhc.shape[2]), maxshape=(None, mhc.shape[1], mhc.shape[2]))
            f.create_dataset('pep',shape=(dslen, pep.shape[1], pep.shape[2]), maxshape=(None, pep.shape[1], pep.shape[2]))
            f.create_dataset('label',shape=(dslen, label.shape[1]) , maxshape=(None, label.shape[1]))
            f.create_dataset('masslabel', shape=(dslen, masslabel.shape[1]) , maxshape=(None, masslabel.shape[1]))
            f.create_dataset('peplen', shape=(dslen, peplen.shape[1]) ,  maxshape=(None, peplen.shape[1]))
            f.create_dataset('relation',  shape=(dslen, relation.shape[1]) , maxshape=(None, relation.shape[1]))
            f.create_dataset('elmo', shape=(dslen, elmo.shape[1], elmo.shape[2]), maxshape=(None, elmo.shape[1], elmo.shape[2]))
        if row1+dslen <= len(f['mhc']) :
            f['mhc'][row1:row1+dslen,:,:] = mhc[:]
            f['pep'][row1:row1+dslen,:,:] = pep[:]
            f['label'][row1:row1+dslen,:] = label[:]
            f['masslabel'][row1:row1+dslen,:] = masslabel[:]
            f['peplen'][row1:row1+dslen,:] = peplen[:]
            f['relation'][row1:row1+dslen,:] = relation[:]
            f['elmo'][row1:row1+dslen,:,:] = elmo[:]
        else :
            f['mhc'].resize((row1+dslen, mhc.shape[1], mhc.shape[2]))
            f['pep'].resize((row1+dslen, pep.shape[1], pep.shape[2]))
            f['label'].resize((row1+dslen, label.shape[1]))
            f['masslabel'].resize((row1+dslen, masslabel.shape[1]))
            f['peplen'].resize((row1+dslen, peplen.shape[1]))
            f['relation'].resize((row1+dslen, relation.shape[1]))
            f['elmo'].resize((row1+dslen, elmo.shape[1], elmo.shape[2]))

            f['mhc'][row1:row1+dslen,:,:] = mhc[:]
            f['pep'][row1:row1+dslen,:,:] = pep[:]
            f['label'][row1:row1+dslen,:] = label[:]
            f['masslabel'][row1:row1+dslen,:] = masslabel[:]
            f['peplen'][row1:row1+dslen,:] = peplen[:]
            f['relation'][row1:row1+dslen,:] = relation[:]
            f['elmo'][row1:row1+dslen,:,:] = elmo[:]
        row1 += dslen
        count += 1