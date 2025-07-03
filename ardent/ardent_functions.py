import os
import pickle
from datetime import datetime
from random import uniform

import numpy as np
import pandas as pd
import rebound
import reboundx
from PyAstronomy.pyTiming import pyPeriod
from reboundx import constants
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.special import erf
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

# ---------- Define constants
Gconst = 6.6743*10**(-11) # The universal gravitation constant, in units of m^3/(kg*s^2)  ;  Value from CODATA 2018
mE_S = 3.986004e14 / 1.3271244e20 # Earth-to-Solar mass ratio
mJ_S = 1.2668653e17 / 1.3271244e20 # Jupiter-to-Solar mass ratio
mE_J = 3.986004e14 / 1.2668653e17 # Earth-to-Jupiter mass ratio


def erf_function(x, b, c):
    return 0.5 * erf(b * (x - c)) + 0.5


def fit_erf(x, y):
    initial_guess = [1, np.mean(x)]
    popt, pcov = curve_fit(erf_function, x, y, p0=initial_guess)
    return popt


def ang_rad(angle):
    """Conversion DEGREES to RADIANS in [-pi,pi]"""
    angle = angle * np.pi / 180.
    angle = (angle + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    return angle
    
    
def AmpStar(ms, P, K, e=0., i=90.):
    """
    Function computing the planet mass and semi-major axis of its orbit given the RV semi-amplitude, period of RV signal and stellar mass.
    
    Arguments
    ---------
    ms (float): stellar mass [M_Sun]
    P (float): period of RV signal [days]
    K (float): RV semi-amplitude [m/s]
    e (float, optional): the orbital eccentricity (default=0)
    i (float, optional): the orbital inclination [deg] (default=90)
    
    Ouptut
    ------
    mp (float): planet mass [M_Earth]
    a (float): semi-major axis of the planetary orbit [AU]
    """
    if i == 0.:
        print('[WARNING] Orbital inclination of at least one body set to 0 degrees')
    
    i = ang_rad(i) # Conversion in  radians, in [-pi,pi[
    mp = (np.sqrt(1-e**2)/np.sin(i)) * (K/28.435) * ms**(2./3.) * (P/365.25)**(1./3.) # [M_Jup] -- valid for any e and i
    mp = mp / mE_J # [M_Earth]
    
    a = (P/365.25)**(2./3.) * ((ms+mp*mE_S)/(1.+mE_S))**(1./3.)

    return mp, a


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODULE 1: data-driven detection limits %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def DataDL(output_file, rvFile, Mstar, rangeP, rangeK, inc_inject, Nsamples=int(2000), Nphases=int(10), fapLevel=0.01):
    """
    Function that executes planet injection-recovery tests in RV timeseries.
    
    Arguments
    ---------
    output_file (string): name of the file that will contain the output of the injection-recovery tests.
    rvFile (string or dict): file (or dictionary) name of the RV residual timeseries, i.e. RV with the Keplerians of known planets removed.
    Mstar (float): stellar mass [M_Sun]
    rangeP, rangeK (list of floats): minimum and maximum orbital periods and RV semi-amplitudes with which to inject a planet (rangeP=[Pmin,Pmax] units of [days] and rangeK=[Kmin,Kmax] units of [RV rms])
    inc_inject: orbital inclination with which to inject the planets [degrees]
    Nsamples (int, optional): number of draws in the (P, M) 2D space
    Nphases (int, optional): number of different orbital phases with which to inject a given planet in the 2D space (P, M). The phase is then spread evenly in [-pi, pi[. The total number of injection-recovery tests corresponds to Nsamples*Nphases.
    fapLevel (float, optional): the FAP threshold under which we consider a signal as significant in the GLS periodogram.
    """
    if type(rvFile)==str:
        t = np.genfromtxt(rvFile, usecols=(0))
        rv = np.genfromtxt(rvFile, usecols=(1))
        rv_err = np.genfromtxt(rvFile, usecols=(2))
    else:
        t = rvFile['jdb']
        rv = rvFile['rv']
        rv_err = rvFile['rv_err']

    output_dir = os.path.dirname(output_file)+'/'

    N = len(rv)
    rms = np.sqrt(np.sum(rv**2)/N)
    Pmin = rangeP[0]
    Pmax = rangeP[1]
    Kmin = rangeK[0]*rms
    Kmax = rangeK[1]*rms
    
    detect_rate = np.zeros(Nsamples)
    phase = np.linspace(-np.pi, np.pi, num=Nphases, endpoint=False)
    
    P = np.array([10**(uniform(np.log10(Pmin), np.log10(Pmax))) for i in range(Nsamples)])
    K = np.array([uniform(Kmin, Kmax) for i in range(Nsamples)])
    M, a = AmpStar(Mstar, P, K, 0., inc_inject)

    for i in tqdm(range(Nsamples)):
        detect = int(0)
        for j in range(Nphases):
            rv_simu = rv - K[i] * np.sin((t*2*np.pi/P[i])+phase[j])

            if 0.6*Pmin > 1.1:
                period_start = 0.6*Pmin # Periodogram search starts at 40% below of Pmin
            else:
                period_start = 1.1
            period_end = 1.4 * Pmax # Periodogram search ends at 40% above Pmax
            
            clp = pyPeriod.Gls((t, rv_simu, rv_err), Pbeg=period_start, Pend=period_end, ls=True, norm="ZK")
            power = clp.power
            plevels = clp.powerLevel(fapLevel)
            imax = np.argmax(power)
            Pmaxpeak = 1/clp.freq[imax]

            if power[imax] > plevels and abs(P[i]-Pmaxpeak) < (P[i]*5./100): # power of max peak above 1% FAP and criterion of 5% on P
                detect += 1

        detect_rate[i] = detect / Nphases
    
    output = {'P':P,'K':K,'M':M,'detect_rate':detect_rate,'Nsamples':Nsamples,'Nphases':Nphases,'Mstar':Mstar,'FAP':fapLevel*100,'rangeP':rangeP,'rangeK':rangeK,'inc_inject':inc_inject}
    pickle.dump(output,open(output_file,'wb'))

    file = open(output_dir+'Injection-recovery_tests.dat', 'w')
    file.write('Period[days] Mass[M_Earth] DetectionRate[%]' + '\n')
    file.write('------------ ------------- ----------------' + '\n')
    for i in range(len(P)):
        file.write(str(P[i]) + ' ' + str(M[i]) + ' ' + str(detect_rate[i]) + '\n')
    file.close()


def Stat_DataDL(output_file, percentage=95, nbins=15, axis_y_var='M'):
    """
    Computes detection limit level curves. Default is 95% detection limits.
    
    Arguments
    ---------
    output_file (string): output file name of the planet injection-recovery tests
    percentage (float, optional): percentage of detection required (default=95%)
    nbins (int, optional): number of Period bins in which to compute the detection limits (default=15)
    axis_y_var (string, optional): the y-axis variable with which to compute the detection limits. Options are mass 'M' or RV semi-amplitude 'K' (default='M')
    
    Output
    ------
    binP: array of period values corresponding to the center of each period bin
    y_threshold: array of 'percentage'% detection limits, one element per period bin
    """
    output = pd.read_pickle(output_file)
    P = np.array(output['P'])
    yvar = np.array(output[axis_y_var])
    detect_rate = np.array(output['detect_rate'])
    rangeP = np.array(output['rangeP'])
    Pmin = rangeP[0] ; Pmax = rangeP[1]
    ymin = np.min(yvar) ; ymax = np.max(yvar)

    bins = 10**np.linspace(np.log10(Pmin), np.log10(Pmax), nbins+1)
    binP = []
    y_threshold = []
    grid = np.linspace(0,ymax,100)
    for p1,p2 in zip(bins[0:-1],bins[1:]):
        selection = (P>=p1)&(P<=p2)
        binP.append((p1+p2)/2)
        try:
            params = fit_erf(yvar[selection],detect_rate[selection])
        except:
            params = [1,np.max(yvar[selection])]
        model = erf_function(grid,params[0],params[1])
        y_threshold.append(grid[np.argmin(abs(model-percentage/100))])
        
    if axis_y_var == 'M':
        output_dir = os.path.dirname(output_file)+'/'
        print(' [INFO] Creation of an output file M-vs-P for %d%% data-driven DL'%percentage)
        split_filename = output_file.split('_')[-1]
        version = int(split_filename.split('.')[0])
        DataDLfile = 'DataDL_' + axis_y_var + '_perc%d_%d.dat' %(percentage, version)
        file = open(output_dir+DataDLfile, 'w')
        file.write('Period ' + axis_y_var + '\n')
        file.write('------ -' + '\n')
        for i in range(len(binP)):
            file.write(str(binP[i]) + ' ' + str(y_threshold[i]) + '\n')
        file.close()

    return np.array(binP), np.array(y_threshold)

  
################### HILL RADIUS
def HillRad(a, Mp, Mstar):
    """
    Hill radius [AU], provided the semi-major axis (a) of the planet orbit [AU], the star mass (Mstar) and planet mass (Mp, same units as Mstar).
    """
    r_H = a * (Mp/(3*Mstar))**(1./3.)
    
    return r_H


################### ANALYTIC STABILITY CRITERION
def AnalyticStability(Mstar, a, e, inc, M):
    """
    Analytic estimation of orbital stability following the AMD framework (Laskar & Petit 2017). This function combines the Hill AMD stability criterion of Petit & Laskar (2018) with the AMD criterion in presence of first-order mean-motion resonances (MMR) of Petit & Laskar (2017).
    
    Arguments
    ---------
    Mstar (float): the stellar mass [M_Sun]
    a, e, inc, M (1D arrays): the semi-major axis [AU], orbital eccentricity, orbital inclination (rad in [-pi,pi]), and mass [M_Sun] of all the planets in the system.
    
    Output
    ------
    analytic_stab (int): indicator of orbital stability. 1: AMD-stable ; 0: AMD-unstable, indicating that further numerical investigations are needed.
    """
    ### Convert G in units of AU, Solar mass and year
    G = Gconst * 1.9884*10**30 # Converting kg in solar mass
    G = G / (149597870700**3) # Converting meters in AU
    G = G * 31557600**2 # Converting seconds in years
        
    NB_pairs = len(a) - 1
    alpha = np.zeros(NB_pairs)
    gamma = np.zeros(NB_pairs)
    epsilon = np.zeros(NB_pairs)
    for i in range(NB_pairs):
        alpha[i] = a[i] / a[i+1]
        gamma[i] = M[i] / M[i+1]
        epsilon[i] = (M[i]+M[i+1]) / Mstar
    
    # ---------- Computation of AMD for each planet pair
    Lambda_ext = M * np.sqrt(G*Mstar*a)  # The Lambda coordinate of the Poincare canonical heliocentric reference frame, applied to the outer body of the considered pair.
    C = np.zeros(NB_pairs) # The total angular momentum of each pair.
    AMD = np.zeros(NB_pairs) # The AMD of each pair
    
    for i in range(NB_pairs):
#        C[i] = Lambda_ext[i]*(1-np.sqrt(1-e[i]**2)) + Lambda_ext[i+1]*(1-np.sqrt(1-e[i+1]**2)) # Hyp: i=90 deg
        C[i] = Lambda_ext[i]*(1-np.sqrt(1-e[i]**2)*np.cos(inc[i])) + Lambda_ext[i+1]*(1-np.sqrt(1-e[i+1]**2)*np.cos(inc[i+1])) # General on inclination
        AMD[i] = C[i] / Lambda_ext[i+1]
    
    # ---------- Selection and computation of the critical AMD
    analytic_stab = 1 # binary number saying if the system is analytically stable (1) or unstable (0)
    
    for i in range(NB_pairs):
        alpha_cir = 1 - (1.46*epsilon[i]**(2./7.))
        alpha_R = 1 - 1.5*epsilon[i]**(1./4.) - 0.316*epsilon[i]**(0.5)
        
        if alpha[i] < alpha_cir and alpha[i] < alpha_R: # Hill
            Cc_H = gamma[i] * alpha[i]**0.5 + 1 - (1+gamma[i])**1.5 * np.sqrt(alpha[i]/(gamma[i]+alpha[i]) * (1+3**(4./3.)*epsilon[i]**(2./3.)*gamma[i]/((1+gamma[i])**2)))
            beta = AMD[i] / Cc_H
            if beta >= 1.:
                analytic_stab = 0
            
        elif alpha[i] < alpha_cir and alpha[i] >= alpha_R: # MMR
            r = 0.80199
            g = 3**4 * (1-alpha[i])**5 / (2**9 * r * epsilon[i]) - 32*r*epsilon[i] / (9*(1-alpha[i])**2)
            Cc_MMR = (gamma[i] / (1+gamma[i])) * g**2 / 2.
            beta = AMD[i] / Cc_MMR
            if beta >= 1.:
                analytic_stab = 0
            
        elif alpha[i] >= alpha_cir:
            analytic_stab = 0
    
    return analytic_stab


################### ORBIT CROSSING
def OrbitCrossing(a, e):
    """
    Function checking if the osculating orbits of any consecutive planet pair cross in a planetary system. Returns True if the orbits of any consecutive planet pair cross. Returns False if none of the planet pairs has crossing orbits.
    
    Arguments
    ---------
    a, e (1D array): the semi-major axis (any unit) and orbital eccentricity of all the planets in the system.
    
    Output
    ------
    orb_cross (int): 1 if the orbits of at least one planet pair cross. 0 otherwise.
    """
    orb_cross = 0
    NB_pairs = len(a) - 1
    for i in range(NB_pairs):
        apo_inner = a[i] * (1+e[i])
        peri_outer = a[i+1] * (1-e[i+1])
        if apo_inner >= peri_outer:
            orb_cross = 1
    
    return orb_cross
    
    
################### MMR CHECK
########## In complement of the orbit crossing criterion, to not analytically exclude crossing orbits that are in MMR
def MMR_check(P):
    """
    Function checking if any consecutive planet pair is close to a low-order mean-motion resonance (MMR), following the criterion described in Dai et al. (2024). This function checks for the following MMR:
    Order 1: 2:1, 3:2, 4:3, 5:4, 6:5
    Order 2: 3:1, 5:3, 7:5
    Order 3: 4:1, 5:2
    
    Arguments
    ---------
    P (1D array): period (any units)
    
    Output
    ------
    in_MMR (int): 1 if at least one planet pair is close to one of the MMR listed above. 0 otherwise.
    """
    in_MMR = int(0) # 1 if any planet pair is close enough to one of the low-order MMR listed below ; 0 otherwise
    bounds_1 = [-0.015, 0.03]
    bounds_23= [-0.015, 0.015]
    
    for i in range(len(P)-1):
        Pratio = P[i+1]/P[i]
        Delta_1 = [Pratio / (p/(p-1)) - 1. for p in range(2,7)] # 1st order MMR: 2:1, 3:2, 4:3, 5:4, 6:5
        Delta_2 = [Pratio / (p/(p-2)) - 1. for p in range(3,8,2)] # 2nd order MMR: 3:1, 5:3, 7:5
        Delta_3 = [Pratio / (p/(p-3)) - 1. for p in range(4,6)] # 3rd order MMR: 4:1, 5:2
        
        for elem in Delta_1:
            if bounds_1[0] <= elem <= bounds_1[1]:
                in_MMR = 1
        for elem in Delta_2:
            if bounds_23[0] <= elem <= bounds_23[1]:
                in_MMR = 1
        for elem in Delta_3:
            if bounds_23[0] <= elem <= bounds_23[1]:
                in_MMR = 1
    
    return in_MMR
    
    
################### 1:1 MMR CHECK
def in_11MMR(P):
    """
    Function checking if any consecutive planet pair is close to a 1:1 MMR.
    
    Arguments
    ---------
    P (1D array): period (any units)
    
    Output
    ------
    MMR11 (int): 1 if at least one planet pair is close to the 1:1 MMR. 0 otherwise.
    """
    MMR11 = int(0) # 1 if any planet pair is close enough to one of the low-order MMR listed below ; 0 otherwise
    bounds = [-0.09, 0.09]
    
    Delta_0 = [P[i+1]/P[i] - 1 for i in range(len(P)-1)]
        
    for elem in Delta_0:
        if bounds[0] <= elem <= bounds[1]:
            MMR11 = 1
    
    return MMR11


################### DYNAMICAL EVOLUTION AND STABILITY ESTIMATION
def Stability(KepParam, phase_param, Mstar, T, dt, min_dist, max_dist, max_drift_a=0.2, GR=False):
    """
    Numerical estimation of orbital stability via short-term integration.
    
    Arguments
    ---------
    KepParam: input Keplerian parameters. This is a 2D vector of the form: KepParam = [a, phase, ecc, w, inc, Omega, Mass] where
        a : semi-major axis [AU]
        phase : orbital phase. Can be either mean longitude [rad], pericenter time, or mean anomaly [rad]
        ecc : orbital eccentricity
        w : argument of perisatron [rad]
        inc : orbital inclination [[rad]
        Omega : longitude of ascending node [rad]
        Mass : planetary mass [M_Sun]
    phase_param (string): Parameter used to trace the planetary orbital phase. Can be either of "mean_long", "pericenter_time", "mean_anomaly".
    Mstar (float): stellar mass [M_Sun]
    T (float): Total integration time [years]
    dt (float): integration timestep [years]
    min_dist and max_dist (floats): The minimum and maximum allowed distances, respectively. The former serves as close encounter criterion, the latter as escape criterion.
    max_drift_a (float, optional): semi-major axis drift threshold beyond which the system is claimed unstable. This stability criterion is a relative drift measurement: drift = (a(t2)-a(t1)) / a(0). (Default=0.2, i.e. 20%)
    GR (bool, optional): include the general relativity module if True (default=False)
    """
    
    a = KepParam[0]
    phase = KepParam[1]
    e = KepParam[2]
    w = KepParam[3]
    inc = KepParam[4]
    O = KepParam[5]
    M = KepParam[6]
    Nbodies = len(a)
    
    sim = rebound.Simulation()
    sim.add(m=Mstar)
    
    if phase_param == 'mean_long': # Mean Longitude is provided
        sim.add(m=M[0], a=a[0], l=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=M[q], a=a[q], l=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q])
            q += 1
            
    elif phase_param == 'pericenter_time': # Time of pericenter passage is provided
        sim.add(m=M[0], a=a[0], T=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=M[q], a=a[q], T=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q])
            q += 1
            
    else: # Mean Anomaly is provided
        sim.add(m=M[0], a=a[0], M=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=M[q], a=a[q], M=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q])
            q += 1
      
    sim.move_to_com()
    sim.dt = dt * 2*np.pi # Conversion to rebound units: 1yr = 2pi
    sim.integrator = "whfast"
    sim.ri_whfast.corrector = 17
    sim.ri_whfast.safe_mode = 0
    sim.exit_max_distance = max_dist
    sim.exit_min_distance = min_dist
    
    if GR == True:
        ##*************** REBOUNDX PART *****************
        rebx = reboundx.Extras(sim)
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = constants.C
        ##************************************************

    Noutputs = int(2000)
    a_simu = np.zeros((Nbodies, Noutputs))

    try:
        for j in range(Noutputs):
            t_output = (j+1) * T / Noutputs
            sim.integrate(t_output*2*np.pi)

            for q in range(Nbodies):
                a_simu[q][j] = sim.particles[q+1].a # The semi-major axis is retained at each output
                
        ######### stability computation -- drift in semi-major axis
        var_a = np.zeros(Nbodies)
        for i in range(Nbodies):
            var_a[i] = np.abs((np.median(a_simu[i][int(Noutputs/2):]) - np.median(a_simu[i][:int(Noutputs/2)])) / a[i])
            
        drift_a = max(var_a)

        if drift_a < max_drift_a:
            stab = 1.
        else:
            stab = 0.
                
    except rebound.Escape:
        stab = 0.

    except rebound.Encounter:
        stab = 0.

    return stab


#############################
################### MAIN CODE
def DynDL(shift, output_file1, output_file2, keplerian_table, D95, inc_inject, Mstar, T=None, dt=None, min_dist=3.0, max_dist=5.0, Nphases=4, max_drift_a=0.2, GR=False, MassPrecision=0.5):
    """
    Computation of the dynamical detection limits
    
    Arguments
    ---------
    shift (int): period bin index (needed for parallel computations)
    output_file1 (string): name of the full output file, containing all the (P, M) tested combinations
    output_file2 (string): name of the dynamical detection limits file
    keplerian_table (dict): table of planetary Keplerian parameters
    D95 (2d array): arrays of periods and masses defining the 95% data-driven detection limits
    inc_inject (float): orbital inclination of the injected planet [deg]
    Mstar (float): stellar mass [M_Sun]
    T (float, optional): total integration time [years] (default=Pouter*500000)
    dt (float, optional): integration timestep [years] (default=Pinner/50)
    min_dist (float, optional): Criterion on close-encounter [Hill_radius] (default=3)
    max_dist (float, optional): Criterion on escape [AU] (default=5)
    Nphases (int, optional): Number of orbital phases per injected (P, K) at which to compute the orbital stability (default=4)
    max_drift_a (float, optional): stability threshold based on maximum relative drift in semi-major axis of the planets (default=0.2, i.e. 20%)
    GR (bool): General relativity correction included if GR=True, False otherwise. (default=False)
    MassPrecision (float): Precision threshold on the dynamical mass limit estimation [M_Earth]. (default=0.5 M_Earth)
    """
        
    AllOutput = output_file1
    DynDL_file = output_file2

    # --- If this is the first call to this function, create the output files
    if shift == int(0):
        file = open(AllOutput, 'a')
        file.write('Period Mass Stability_rate' + '\n')
        file.write('------ ---- --------------' + '\n')
        file.close()

        file0 = open(DynDL_file, 'a')
        file0.write('Period Mass Stability_rate' + '\n')
        file0.write('------ ---- --------------' + '\n')
        file0.close()
    
    # ---------- Get the parameters
    table_keplerian = keplerian_table.copy()
    Nplanets = len(table_keplerian)

    # ---------- Extract the period and mass of the injected planet, according to the data-driven detection limits. Then convert P to a.
    P_inject = np.array(D95['period']) # [days]
    M_lim100 = np.array(D95['mass']) # [M_Earth]
    a_lim100 = (P_inject/365.25)**(2./3.) * ((Mstar + M_lim100*mE_S)/(1+mE_S))**(1./3.) # [AU]

    injected_body = np.array([P_inject[shift],a_lim100[shift],0,0,0,0,0,inc_inject,0,np.nan,M_lim100[shift]]) # ['period','semimajor','mean_long','mean_anomaly','pericenter_time','ecc','periastron','inc','asc_node','semi-amp','mass']
    table_keplerian.loc[len(table_keplerian)] = injected_body
    index0 = np.where(np.sort(table_keplerian['semimajor'].values)==table_keplerian['semimajor'].values[-1])[0][0]
    table_keplerian = table_keplerian.sort_values(by='semimajor') # ---------- Sort the parameters by increasing a
    
    if dt is None:
        dt = np.min(table_keplerian['period'])/(365.25*50) # By default, P_inner/50 in [yr]
    if T is None:
        T = 1e5*np.max(table_keplerian['period'])/365.25 # By default, 100k * P_outer [yr]

    P = np.array(table_keplerian['period'])/365.25              # [yr]
    K = np.array(table_keplerian['semi-amp'])
    e = np.array(table_keplerian['ecc'])
    w = ang_rad(np.array(table_keplerian['periastron']))
    O = ang_rad(np.array(table_keplerian['asc_node']))
    inc = ang_rad(np.array(table_keplerian['inc']))
    M = np.array(table_keplerian['mass']) *mE_S                # [MSun]
    a = np.array(table_keplerian['semimajor'])                 # [AU]
    
    mean_long = np.array(table_keplerian['mean_long'])
    mean_anomaly = np.array(table_keplerian['mean_anomaly'])
    peri_time = np.array(table_keplerian['pericenter_time'])
    if np.sum(np.isnan(mean_long)) == 0:
        phase = ang_rad(mean_long)
        phase_param = 'mean_long'
    elif np.sum(np.isnan(peri_time)) == 0:
        phase = ang_rad(peri_time)
        phase_param = 'pericenter_time'
    else:
        phase = ang_rad(mean_anomaly)
        phase_param = 'mean_anomaly'
    
    params = [a,phase,e,w,inc,O,M] # Keep the order of the elements! a,phase,e,w,inc,O,M
    phases_inject = np.linspace(-np.pi, np.pi, Nphases, endpoint=False) # With endpoint=False, generate Nphases points with constant interval in [start, stop[ (the last value is excluded)
    
    min_dist = min_dist * HillRad(a[0], M[0], Mstar)
    max_dist = max_dist * a[-1]

    P_inject = P_inject/365.25

    # ---------- Start the iterative process to find the minimum mass at which stability rate = 0%
    print('\n [INFO] Processing stability estimation at period %f[d] (bin %.0f / %.0f)  <---------'%(P_inject[shift]*365.25,shift+1,len(D95['period'])))
    
    MMR11 = in_11MMR(P)
    if MMR11 == 0: # ARDENT does not consider the 1:1 MMR case
        stab_rate_analytic = AnalyticStability(Mstar, a, e, inc, M)
        if stab_rate_analytic == 0: # i.e., if and only if the system (including the injected planet) is AMD-unstable, we have to further test if the system is truly short-term unstable
            crossing = OrbitCrossing(a, e)
            if crossing == 1 and MMR_check(P) == 0: # If orbits are crossing with no planet pair close to MMR, for sure the system is unstable. No need of numerical simulations.
                stab_rate = 0
            else:
                stab_rate_numeric = 0
                for j in range(Nphases):
                    phase[index0] = phases_inject[j]
                    stab_rate_numeric += Stability(params, phase_param, Mstar, T, dt, min_dist, max_dist, max_drift_a, GR) # stab_rate_numeric is a binary number: 0 = unstable, 1 = stable
        
                stab_rate = round((stab_rate_numeric / Nphases) * 100., 2) # Rate of stable systems, in %
                
        else: # in this case, the system is AMD-stable
            stab_rate = 100
    else: # In case any planet pair is close to 1:1MMR, we define the system as unstable
        stab_rate = 0
        
    file = open(AllOutput, 'a')
    file.write(str(P_inject[shift]*365.25) + ' ' + str(M[index0]/mE_S) + ' ' + str(stab_rate) + '\n')
    file.close()

    M_lim100[shift] = M_lim100[shift]*mE_S

    iteration=0
    if (stab_rate <= 0.1): # If the rate of stability is smaller than 0.1%
        Thresh = MassPrecision * mE_S # mass precision criterion, expressed in [M_Sun]
        dM = 1000000.
        q = int(0)
        M_max = M_lim100[shift]
        M_min = 0.001*mE_S
        while dM > Thresh:
            iteration+=1
            if q == 0 and M_lim100[shift] > 0.001*mE_S:
                M[index0] = 0.001*mE_S
                a[index0] = P_inject[shift]**(2./3.) * ((Mstar+M[index0])/(1.+mE_S))**(1./3.)

                if MMR11 == 0:
                    stab_rate_analytic = AnalyticStability(Mstar, a, e, inc, M)
                    if stab_rate_analytic == 0:
                        crossing = OrbitCrossing(a, e)
                        if crossing==1 and MMR_check(P) == 0:
                            stab_rate = 0.
                        else:
                            stab_rate_numeric = 0.
                            for j in range(Nphases):
                                phase[index0] = phases_inject[j]
                                stab_rate_numeric += Stability(params, phase_param, Mstar, T, dt, min_dist, max_dist, max_drift_a, GR)

                            stab_rate = round((stab_rate_numeric / Nphases) * 100., 2) # Rate of stable systems, in %
                    else:
                        stab_rate = 100
                else:
                    stab_rate = 0
                    
                file = open(AllOutput, 'a')
                file.write(str(P_inject[shift]*365.25) + ' ' + str(M[index0]/mE_S) + ' ' + str(stab_rate) + '\n')
                file.close()

                if stab_rate > 0.1:
                    M_max = M_lim100[shift]
                    M_min = 0.001*mE_S
                    dM = M_max - M_min
                else:
                    dM = 0. # Get out of the loop
                    Mlim = M[index0]

            elif q == 0 and M_lim100[shift] == 0.001*mE_S:
                dM = 0. # Get out of the loop
                Mlim = M[index0]

            else:
                M[index0] = (M_max+M_min) / 2.
                a[index0] = P_inject[shift]**(2./3.) * ((Mstar+M[index0])/(1.+mE_S))**(1./3.)
                stab_rate_analytic = AnalyticStability(Mstar, a, e, inc, M)
                if stab_rate_analytic == 0:
                    crossing = OrbitCrossing(a, e)
                    if crossing==1 and MMR_check(P) == 0:
                        stab_rate = 0.
                    else:
                        stab_rate_numeric = 0.
                        for j in range(Nphases):
                            phase[index0] = phases_inject[j]
                            stab_rate_numeric += Stability(params, phase_param, Mstar, T, dt, min_dist, max_dist, max_drift_a, GR)
                        stab_rate = round((stab_rate_numeric / Nphases) * 100., 2)
                else:
                    stab_rate = 100
                    
                file = open(AllOutput, 'a')
                file.write(str(P_inject[shift]*365.25) + ' ' + str(M[index0]/mE_S) + ' ' + str(stab_rate) + '\n')
                file.close()

                if stab_rate > 0.1:
                    M_min = M[index0]

                elif stab_rate <= 0.1:
                    M_max = M[index0]

                dM = M_max - M_min
                Mlim = M_max

            q += 1

        file = open(DynDL_file, 'a')
        file.write(str(P_inject[shift]*365.25) + ' ' + str(Mlim/mE_S) + ' ' + str(stab_rate) + '\n')
        file.close()

    else:
        file = open(DynDL_file, 'a')
        file.write(str(P_inject[shift]*365.25) + ' ' + str(M[index0]/mE_S) + ' ' + str(stab_rate) + '\n')
        file.close()
        
    print(' [INFO] Completed stability estimation at period %f[d] (bin %.0f / %.0f) <---------'%(P_inject[shift]*365.25,shift+1,len(D95['period'])))


#############################
################### Long integration
def LongTermStab(output_file, keplerian_table, Mstar, T=None, dt=None, min_dist=3.0, max_dist=5.0, Noutputs=1000, GR=False):
    """
    Long-term numerical integration of a specific system solution.

    Arguments
    ---------
    output_file (string): name of the file that will contain the outputs of the numerical integration
    keplerian_table (array of floats): orbital parameters of the planets in the system
    Mstar (float): stellar mass [M_Sun]
    T (float, optional): total integration time [years] (default=Pouter*1e6)
    dt (float, optional): integration timestep [years] (default=Pinner/100)
    min_dist (float, optional): minimal approach (close encounter) threshold [Hill radius] (default=3)
    max_dist (float, optional): maximal distance (escape) threshold [AU] (default=5)
    Noutputs (int, optional): number of output timesteps saved in a file (default=1000)
    GR (bool, optional): include general relativity module (default=False)
    """

    table_keplerian = keplerian_table.copy()

    if dt is None:
        dt = np.min(table_keplerian['period'])/(365.25*100) # By default, Pb/100 in [yr] (smaller dt than in DynDL, because here WHFast integrator will be used)
    if T is None:
        T = 1e6*np.max(table_keplerian['period'])/365.25 # By default, 1M * P_outer [yr]
        
    e = np.array(table_keplerian['ecc'])
    w = ang_rad(np.array(table_keplerian['periastron']))
    inc = ang_rad(np.array(table_keplerian['inc']))
    O = ang_rad(np.array(table_keplerian['asc_node']))
    m = np.array(table_keplerian['mass']) *mE_S                # [MSun]
    a = np.array(table_keplerian['semimajor'])                 # [AU]
    
    mean_long = np.array(table_keplerian['mean_long'])
    mean_anomaly = np.array(table_keplerian['mean_anomaly'])
    peri_time = np.array(table_keplerian['pericenter_time'])
    if np.sum(np.isnan(mean_long)) == 0:
        phase = ang_rad(mean_long)
        phase_param = 'mean_long'
    elif np.sum(np.isnan(peri_time)) == 0:
        phase = ang_rad(peri_time)
        phase_param = 'pericenter_time'
    else:
        phase = ang_rad(mean_anomaly)
        phase_param = 'mean_anomaly'
    
    print(' [INFO] --Long term stability-- Start of the numerical integration ')
    sim = rebound.Simulation()
    sim.add(m=Mstar)
    
    if phase_param == 'mean_long': # Mean Longitude is provided
        sim.add(m=m[0], a=a[0], l=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=m[q], a=a[q], l=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q])
            q += 1
            
    elif phase_param == 'pericenter_time': # Time of pericenter passage is provided
        sim.add(m=m[0], a=a[0], T=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=m[q], a=a[q], T=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q])
            q += 1
            
    else: # Mean Anomaly is provided
        sim.add(m=m[0], a=a[0], M=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=m[q], a=a[q], M=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q])
            q += 1

    sim.move_to_com()
    sim.dt = dt * 2*np.pi # Conversion to rebound units: 1yr = 2pi
    sim.integrator = "whfast"
    sim.ri_whfast.corrector = 17
    sim.ri_whfast.safe_mode = 0
    sim.exit_max_distance = max_dist * a[-1]
    sim.exit_min_distance = min_dist * HillRad(a[0], m[0], Mstar)

    if GR == True:
        ##*************** REBOUNDX PART *****************
        rebx = reboundx.Extras(sim)
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = constants.C
        ##************************************************

    try:
        file = open(output_file, 'w')
        file.write('time[yr] a[AU] mean_long[rad] ecc w[rad] inc[rad] asc_node[rad]' + '\n')
        file.write('-------- ----- -------------- --- ------ -------- -------------' + '\n')
        for j in tqdm(range(Noutputs)):
            t_output = (j+1) * T / Noutputs
            sim.integrate(t_output*2*np.pi)

            file.write(str(t_output))
            for q in range(len(inc)):
                file.write(' ' + str(sim.particles[q+1].a) + ' ' + str(sim.particles[q+1].l) + ' ' + str(sim.particles[q+1].e) + ' ')
                file.write(str(sim.particles[q+1].omega) + ' ' + str(sim.particles[q+1].inc) + ' ' + str(sim.particles[q+1].Omega))
            file.write('\n')
                
        file.close()
        print(' [INFO] --Long term stability-- Processing finished! The system survived the integration. ')
        
    except rebound.Escape:
        print(' [INFO] --Long term stability-- Simulation stopped! Escape of one particle. ')
        
    except rebound.Encounter:
        print(' [INFO] --Long term stability-- Simulation stopped! Close encounter. ')

