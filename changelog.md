# Changelog

### Version 1.2.1
* Solved issues with plots of detection limits in the orbital plane (ARDENT_FinalPlot function): resolution preserved in zoomed-in plots, no truncation of mass limits, X and Y axes have identical ranges, adjusted the interpolation method, parameter 'nbins' added to the function to enable modifying the number of bins of the 'RV only' plot. 
* Introduction of a version number for dynamical runs. This enables to re-run dynamical detection limits with different simulation parameters and keep all former results in memory.
* Updated tutorial Jupyter notebook. 

### Version 1.2.0
* New feature: inject planets with random orbital eccentricity, following the beta distribution of Kipping et al. (2013) (keyword: `ecc_inject='beta'`).
* New feature: inject planets with random orbital inclination, following either a uniform distribution in cos(i) (keyword: `inc_inject='random1'`), or a Gaussian (90, sigma=5deg) truncated between 75 and 90 deg (keyword: `inc_inject='random2'`).  

### Version 1.1.0
* New feature: dense period sampling is possible around the known planets for the computation of dynamical detection limits (keyword: `fine_grid`).
* Add option of inset plot in plotting function `ARDENT_Plot_StabDL` (keyword: `inset_plot`). 
* New feature: ARDENT is now compatible with externally-sourced data-driven detection limits to add the dynamical constraints (keyword: `ExternalDataDL`)
