import sys, os, glob
sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/home/psr')
import numpy as np
from ubc_AI.training import pfddata
import math
import time
t0 = time.time()
#pfd_files_pulsars = glob.glob('/beegfs/vishnu/scripts/neural_network/test/pulsars/*.pfd')
pfd_files_nonpulsars = sorted(glob.glob('/beegfs/vishnu/scripts/neural_network/test/nonpulsars/*.pfd'))

fraction = 4
current_segment = 2
max_value = int(math.ceil(len(pfd_files_nonpulsars)/fraction))
print(max_value)
double_max_value = 2 * max_value
triple_max_value = 3 * max_value
print(double_max_value)
# Initialise data objects from getdata class
#data_object_pulsars = [pfddata(f) for f in pfd_files_pulsars] 
data_object_nonpulsars = [pfddata(f) for f in pfd_files_nonpulsars[double_max_value:triple_max_value]] 
print('loaded data into memory')
# Extract 4 features based on Zhu et.al 2014

#1 time vs phase plot
#time_phase_plots_pulsars = [f.getdata(intervals=48) for f in data_object_pulsars]
time_phase_plots_nonpulsars = [f.getdata(intervals=48) for f in data_object_nonpulsars]
print('time phase done')
#2 freq vs phase plot
#freq_phase_plots_pulsars = [f.getdata(subbands=48) for f in data_object_pulsars]
freq_phase_plots_nonpulsars = [f.getdata(subbands=48) for f in data_object_nonpulsars]

print('freq phase done')
#3 Pulse Profile
#pulse_profile_pulsars = [f.getdata(phasebins=64) for f in data_object_pulsars]
pulse_profile_nonpulsars = [f.getdata(phasebins=64) for f in data_object_nonpulsars]

print('pulse profile done')
#4 DM Curve

#dm_curve_pulsars = [f.getdata(DMbins=60) for f in data_object_pulsars]
dm_curve_nonpulsars = [f.getdata(DMbins=60) for f in data_object_nonpulsars]

print('dm curve done')
#Save all features as numpy array files

#np.save('time_phase_gbncc_test_data_pulsars.npy', time_phase_plots_pulsars) 
np.save('time_phase_gbncc_test_data_nonpulsars_part3.npy', time_phase_plots_nonpulsars) 

#np.save('freq_phase_gbncc_test_data_pulsars.npy', freq_phase_plots_pulsars) 
np.save('freq_phase_gbncc_test_data_nonpulsars_part3.npy', freq_phase_plots_nonpulsars) 

#np.save('pulse_profile_gbncc_test_data_pulsars.npy', pulse_profile_pulsars) 
np.save('pulse_profile_gbncc_test_data_nonpulsars_part3.npy', pulse_profile_nonpulsars) 

#np.save('dm_curve_gbncc_test_data_pulsars.npy', dm_curve_pulsars) 
np.save('dm_curve_gbncc_test_data_nonpulsars_part3.npy', dm_curve_nonpulsars) 
t1 = time.time()
print('Total time taken for the code to execute is %s seconds' %str(t1-t0))
