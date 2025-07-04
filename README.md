![ARDENT Logo](logo/ARDENTlogo.png)

# ARDENT
The Algorithm for the Refinement of DEtection limits via N-body stability Threshold (ARDENT) is a python package for the computation of exoplanet detection limits from radial velocity data. In addition to the classic data-driven detection limits, ARDENT includes the orbital stability constraint (what we call dynamical detection limits). The final output is a more constraining detection limits curve, taking into account both the detectability of potential planets and their dynamical plausibility. 

The code can be used as two separate modules, for the data-driven detection limits and the dynamical detection limits. ARDENT can be used to refine the detection limits, test the dynamical plausibility of a planet candidate, and check if there is dynamical room for additional planets in a certain period range. 

### Dependencies
ARDENT is built on a series of python packages, contained in the requirements.txt file. 
+ numpy
+ matplotlib
+ pandas
+ rebound
+ reboundx
+ joblib
+ tqdm
+ PyAstronomy

### Installation 
To use ARDENT, you can clone the repository to your computer: `git clone https://github.com/manustalport/ardent`. 
Once on your computer, add ARDENT to your `$PYTHONPATH`. To proceed, in a terminal, `cd` into the ARDENT folder and then: 
- `echo "export PYTHONPATH=$PWD/ardent:\$PYTHONPATH" >> ~/.bash_profile`
- `source ~/.bash_profile`

It is recommended using a virtual environment to run ARDENT, as some dependencies run on specific versions. This will avoid conflicts with your own system. 
In a terminal, run `python -m venv ardent_venv` to create a virtual environment named ardent_venv. 
You need to activate this environment with `source ardent_venv/bin/activate`. 

Finally, install all the dependencies at once in your new virtual environment: `pip install -r requirements.txt`

The code was tested with Python 3.11, rebound 4.4.3 and reboundx 4.3.0. 

ARDENT is installable on Linux and MacOS distributions.

### Example use 
Follow the steps of the ardent/hands-on_tutorial.ipynb file, that will guide you through the various ARDENT functions. 

### Contributors 
+ Manu Stalport, University of Liège (contact point)
+ Michaël Cretignier, University of Oxford

### Citations
If you use ARDENT for your research, please cite: 

Stalport, et al. 2025, submitted to A&A: ARDENT -- A Python package for fast dynamical detection limits with radial velocities 

