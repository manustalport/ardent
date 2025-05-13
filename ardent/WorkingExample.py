import numpy as np
import os
import ardent

# Retrieve the residual RV timeseries and create the ardent object 
data = np.genfromtxt('K2-312_MultiDgp_Residuals_test3.dat')
jdb, rv, rv_err = data[:,0:3].T

vec = ardent.ARDENT_tableXY(jdb, rv, rv_err)        #init the ARDENT object time-series
vec.ARDENT_AddStar(mass=1.18,starname='K2-312')  #init the star -- no space character allowed in star name
vec.ARDENT_Set_output_dir('TestRun')

vec.ARDENT_ImportPlanets('system_parameters.dat')

vec.ARDENT_PlotPlanets(new=True)

vec.ARDENT_DetectionLimitRV(rangeP=[2., 200.], rangeK=[0.1, 1.2], Nsamples=1500) 

vec.ARDENT_Plot_DataDL(axis_y_var='K')

vec.ARDENT_DetectionLimitStab(NlocalCPU=4, DataDLfile='TestRun/K2-312_Data-drivenDL_0.p', integration_time=1000., NAFFthr=0., GR=True, relaunch=True)

vec.ARDENT_Plot_StabDL(DataDLfile='TestRun/DataDL_M_perc95_0.dat', DynDLfile='TestRun/K2-312_DynamicalDL_0.dat')

