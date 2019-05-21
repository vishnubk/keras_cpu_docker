import sys, os, glob
sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/home/psr')
import numpy as np
from ubc_AI.training import pfddata

#pfd_files_pulsars = glob.glob('/beegfs/vishnu/scripts/neural_network/train/pulsars/*.pfd')
pfd_files_nonpulsars = glob.glob('/beegfs/vishnu/scripts/neural_network/train/nonpulsars/*.pfd')

# Initialise data objects from getdata class
#data_object_pulsars = [pfddata(f) for f in pfd_files_pulsars] 
data_object_nonpulsars = [pfddata(f) for f in pfd_files_nonpulsars] 

# Extract 4 features based on Zhu et.al 2014

#1 time vs phase plot
#time_phase_plots_pulsars = [f.getdata(intervals=48) for f in data_object_pulsars]
time_phase_plots_nonpulsars = [f.getdata(intervals=48) for f in data_object_nonpulsars]

#2 freq vs phase plot
#freq_phase_plots_pulsars = [f.getdata(subbands=48) for f in data_object_pulsars]
freq_phase_plots_nonpulsars = [f.getdata(subbands=48) for f in data_object_nonpulsars]

#3 Pulse Profile
#pulse_profile_pulsars = [f.getdata(phasebins=64) for f in data_object_pulsars]
pulse_profile_nonpulsars = [f.getdata(phasebins=64) for f in data_object_nonpulsars]

#4 DM Curve

#dm_curve_pulsars = [f.getdata(DMbins=60) for f in data_object_pulsars]
dm_curve_nonpulsars = [f.getdata(DMbins=60) for f in data_object_nonpulsars]

#Save all features as numpy array files

#np.save('time_phase_data_pulsars.npy', time_phase_plots_pulsars) 
np.save('time_phase_data_nonpulsars.npy', time_phase_plots_nonpulsars) 

#np.save('freq_phase_data_pulsars.npy', freq_phase_plots_pulsars) 
np.save('freq_phase_data_nonpulsars.npy', freq_phase_plots_nonpulsars) 

#np.save('pulse_profile_data_pulsars.npy', pulse_profile_pulsars) 
np.save('pulse_profile_data_nonpulsars.npy', pulse_profile_nonpulsars) 

#np.save('dm_curve_data_pulsars.npy', dm_curve_pulsars) 
np.save('dm_curve_data_nonpulsars.npy', dm_curve_nonpulsars) 

