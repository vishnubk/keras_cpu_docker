import numpy as np
import glob
data = glob.glob('/beegfs/vishnu/native_test_set_numpy/*.npy')
labels = ('/beegfs/vishnu/native_test_set_numpy/pfd_correct.txt')
gbncc_data = [np.load(f) for f in data]

print(np.shape(f) for f in gbncc_data)

