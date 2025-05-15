# ARDENT
The Algorithm for the Refinement of DEtection limits via N-body stability Threshold (ARDENT) is a python package for the computation of detection limits from radial velocity data. In addition to the classic data-driven detection limits, ARDENT includes the orbital stability constraint (what we call dynamical detection limits). The final output is a more constraining detection limits curve, taking into account both the detectability of potential planets and their dynamical plausibility. 

The code can be used as two separate modules, for the data-driven detection limits and the dynamical detection limits. ARDENT can be used to refine the detection limits, test the dynamical plausibility of a planet candidate, and check if there is dynamical room for additional planets in a certain period range. 

### Dependencies
ARDENT is built on a series of python packages. 
+ numpy
+ matplotlib
+ rebound
+ reboundx
+ joblib
+ tqdm
+ PyAstronomy

### Installation 
The simplest way to install ardent on your machine is to use pip. Given that some dependencies must have specific versions, I recommend using a virtual environment to avoid conflicts with your own system. 

### Example use 
Follow the steps of the ardent/WorkingExample.py file. 

The first step in to using ardent is to initialise the ardent class. It takes as arguments the RV timeseries within which you want to compute the detection limits. 

```ruby
import numpy as np
import os
import ardent

data = np.genfromtxt('K2-312_MultiDgp_Residuals_test3.dat')
jdb, rv, rv_err = data[:,0:3].T
vec = ardent.ARDENT_tableXY(jdb, rv, rv_err) 
```

Secondly, you want to specify information about the host star (name and mass) and the known planets (orbital elements). This information is used to produce the plots, but mostly to compute the dynamical detection limits. 

```ruby
vec.ARDENT_AddStar(mass=1.18,starname='K2-312') # No space character allowed in star name
vec.ARDENT_Set_output_dir('TestRun') # Let's put the products of our simulations in a folder named TestRun
vec.ARD_AddPlanets(p=873.1, k=196.1, e=0.85, omega=42.0, asc_node = 0.0, mean_long=5, inc=90.0)
vec.ARD_AddPlanets(p=0.7195, k=3.638, e=0.00, omega=0.0, asc_node = 0.0, mean_long=147, inc=90.0)
```

In this case, the system is known to contain 2 planets. Alternatively, you can include the planets from an input parameter file (here, named system_parameters.dat). The procedure to do so is as simple: 

```ruby
vec.ARDENT_ImportPlanets('system_parameters.dat')
```

You can plot the planetary system. 

```ruby
vec.ARDENT_PlotPlanets(new=True)
```

Now it is time to focus on the detection limits! Ardent works on a 2-step process. You must call first a function to compute the classic data-driven detection limits before calling the function computing the dynamical detection limits. Let's suppose you want to compute the detection limits in a period range between 2 and 200 days. 

```ruby
vec.ARDENT_DetectionLimitRV(rangeP=[2., 200.]) 
```

This function produces injection-recovery tests, and plots the 50 and 95% data-driven detection limits in the mass-period space. For a full list of arguments to this function, check the docstring. If you want to look at the data-driven detection limits in the RV semi-amplitude versus period space, this is also possible: 

```ruby
vec.ARDENT_Plot_DataDL(axis_y_var='K')
```

Now that you computed the data-driven detection limits, you can start the computation of the dynamical detection limits. This is done very easily in ardent within a single line of code (but carefully read the following). 

```ruby
vec.ARDENT_DetectionLimitStab(NlocalCPU=4, DataDLfile='TestRun/K2-312_Data-drivenDL_0.p', integration_time=1000., NAFFthr=0., GR=True, relaunch=True)
```

The input parameters that you specify in this function are crucial for the reliability of your results! This function will initiate numerical evolution simulations and evaluate the orbital stability from those simulations. Therefore you need to pay a strong importance to the convergence of your simulations. Below, I briefly explain each argument of this function and how to best set it up. 

+ ```NlocalCPU``` Ardent is usable both on a local machine and on a HPC cluster. Used locally, set ```NlocalCPU``` to the number of CPUs you want to allocate to the computation. Because ardent computes the orbital stability on a series of systems, the more CPUs you dedicate, the faster will be the processing. Alternatively, if you use ardent on a HPC cluster, set ```NlocalCPU`` to 0 to specify the code that you launch it from a cluster. You can then launch the code with with the slurm argument --array to specify the number of cluster cores you want to use.
+ ```DataDLfile``` Path to the output data-driven detection limits file. Ardent computes the dynamical detection limits based on the data-driven ones.
+ ```nbins``` The number of period bins inside which you compute the mass limits.
+ ```integration_time``` The total integration time (in [years]) of the numerical simulations. This is a critical parameter to obtain reliable detection limits. The numerical computation of orbital stability in ardent is based on the estimation of chaos via the NAFF fast chaos indicator. You need to integrate each planetary system long enough for the algorithm to properly converge. However, you do not want to integrate for too long, in which case the computation time will get very long. As a rule of thumb, a convenient integration time is around 100 thousand orbits of the outermost planet in the system. NOTE TO MYSELF: INTRODUCING USERS TO CHAOS INDICATORS, NAFF, STABILITY CALIBRATION FOLLOWING STALPORT+2022, CONVERGENCE TIME, RISKS TO COMPLICATE A LOT THE USE OF ARDENT AND TO MAKE IT NOT USER-FRIENDLY DESPITE ALL THE EFFORTS. SHOULD I USE A MORE BASIC STABILITY ESTIMATION, LOOSING IN EFFICIENCY BUT GAINING A LOT IN EASE OF USE? SPEAK WITH JEAN-BAPTISTE FOR HIS OPINION ON THIS. 


### Contributors 
+ Manu Stalport, University of Liège (contact point)
+ Michaël Cretignier, University of Oxford

### Citations
