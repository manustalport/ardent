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

ARDENT is installable on Linux and MacOS distributions.

### Example use 
Follow the steps of the ardent/hands-on_tutorial.ipynb file, that will guide you through the various ARDENT functions. 

### Contributors 
+ Manu Stalport, University of Liège (contact point)
+ Michaël Cretignier, University of Oxford

### Citations
If you use ARDENT for your research, please cite Stalport, et al. 2025, submitted to A&A. 
