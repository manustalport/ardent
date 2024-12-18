import numpy as np 
import os
from matplotlib import pyplot as plt
import ardent

######## MODULE 1: data-driven detection limits ########
sys_name = 'K2-312'
rvFile = 'K2-312_MultiDgp_Residuals_test3.dat'
Mstar = 1.18 # [M_Sun] 
rangeP = [2., 200.] # [days]
rangeK = [0.1, 1.3] # [rms] 
### --- Following are optional parameters
Nsamples = int(1000)
Nphases = int(5) 
fapLevel = 0.01 # 1% FAP threshold 
nbins = 17 

ardent.DataDL(sys_name, rvFile, Mstar, rangeP, rangeK, Nsamples, Nphases, fapLevel, nbins)


######## MODULE 2: dynamical detection limits ########
NlocalCPU = int(3) # The number of CPUs allocated to compute the dynamical detection limits, to fasten the computation.
# If NlocalCPU is set to 0, the program will be compatible with external cluster.
# To use the code on a cluster, import all the needed files in the execution folder. Create a virtual environement to install rebound and reboundx.
# Use the argument --array of sbatch to call the programme for each value of P_inject
param_file = 'system_parameters.dat'
DataDrivenLimitsFile = 'Data-driven_95MassLimits.dat'
Nplanets = int(2) # The number of known planets in the system

if NlocalCPU == 0:
    import sys
    shift = int(sys.argv[1])
    ardent.DynDL(shift, Nplanets, param_file, DataDrivenLimitsFile)
    
elif NlocalCPU > 0:
    from joblib import Parallel, delayed
    P_inject = np.genfromtxt(DataDrivenLimitsFile, usecols=(0), skip_header=(2))
    N = len(P_inject)
    _=Parallel(n_jobs=NlocalCPU)(delayed(ardent.DynDL)(shift, Nplanets, param_file, DataDrivenLimitsFile) for shift in range(N))


######## Final plot ########
sys_name = 'K2-312'
data_driven = 'Data-driven_95MassLimits.dat' 
stability_driven = 'Final_DynamicalDetectLim.dat' 

P = np.genfromtxt(stability_driven, usecols=(0), skip_header=int(2)) 
M_stb = np.genfromtxt(stability_driven, usecols=(1), skip_header=int(2)) 
M_data = np.genfromtxt(data_driven, usecols=(1), skip_header=int(2)) 

indexes = np.argsort(P)
P = np.array(P)[indexes] 
M_stb = np.array(M_stb)[indexes]

fig = plt.figure(figsize=(5,4)) 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(P, M_data, color='xkcd:mahogany', lw=3.0, zorder=2, label=r'\large{w/o stability}') 
plt.plot(P, M_stb, ls=':', color='xkcd:fire engine red', lw=3.0, zorder=10, label=r'\large{w stability}')
# plt.plot(P, M_data, color='gray', lw=3.0, zorder=2, label=r'\large{w/o stability}') 
# plt.plot(P, M_stb, ls=':', color='red', lw=3.0, zorder=10, label=r'\large{w stability}')
plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
plt.xscale('log') 
plt.xlabel(r'\Large{Period [d]}')
plt.ylabel(r'\Large{Mass [M$_{\oplus}$]}')
plt.tick_params(labelsize=12)
plt.legend(loc='upper left')
plt.tight_layout() 
plt.savefig('FinalDetectionLimits_'+sys_name+'.png', format='png', dpi = 300)


