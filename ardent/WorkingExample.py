import numpy as np
import os
import ardent as ard

# Retrieve the residual RV timeseries and create the ardent object 
data = np.genfromtxt('K2-312_MultiDgp_Residuals_test3.dat')
jdb, rv, rv_err = data[:,0:3].T

vec = ard.ARD_tableXY(jdb, rv, rv_err)        #init the ARDENT object time-series
vec.ARD_AddStar(mass=1.18,starname='K2-312')  #init the star -- no space character allowed in star name
vec.ARD_Set_output_dir('FinalTest')

vec.ARD_ImportPlanets('system_parameters.dat')

vec.ARD_PlotPlanets(new=True)

vec.ARD_DetectionLimitRV(rangeP=[2., 200.], rangeK=[0.1, 1.2], Nsamples=1500) 

vec.ARD_Plot_DataDL(axis_y_var='K')

vec.ARD_DetectionLimitStab(NlocalCPU = 4,DataDLfile='TestPlots/K2-312_Data-drivenDL_0.p',integration_time=1000., NAFFthr=0., GR=True, relaunch=True)

