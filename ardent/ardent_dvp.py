import numpy as np
import os
from matplotlib import pyplot as plt
from random import uniform
from PyAstronomy.pyTiming import pyPeriod
from tqdm import tqdm
import rebound
import reboundx
from reboundx import constants

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODULE 1: data-driven detection limits %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
def DataDL(sys_name, rvFile, Mstar, rangeP, rangeK, Nsamples=int(2000), Nphases=int(10), fapLevel=0.01, nbins=int(20), plot=True):
    """
    Computation of data-driven detection limits. This function outputs a file containing the 95% mass detection limits for periods between Pmin and Pmax.
    
    Arguments
    ---------
    sys_name (string): the name of the system under study.
    rvFile (string): file name of the RV residual timeseries, i.e. RV with the Keplerians of known planets removed.
    Mstar (float): stellar mass [M_Sun]
    rangeP, rangeK (list of floats): minimum and maximum orbital periods and RV semi-amplitudes with which to inject a planet (rangeP=[Pmin,Pmax] units of [days] and rangeK=[Kmin,Kmax] units of [RV rms])
    Nsamples (int, optional): the number of injected planets in the 2D space (P, M) at a given orbital phase
    Nphases (int, optional): the number of different orbital phases with which to inject a given planet in the 2D space (P, M). The phase is then spread evenly in [0, 2pi[. The total number of injection-recovery tests corresponds to Nsamples*Nphases.
    fapLevel (float, optional): the FAP threshold under which we consider a signal as significant in the GLS periodogram.
    nbins (int, optional): the number of period bins with which to output the 95% mass detection limits.
    plot (bool, optional): plot the result. default=True.
    """
    cwd = os.getcwd() # Current Working Directory
    path_output = cwd + '/' + sys_name + '_DetectionLimits'
    if os.path.isdir(path_output) == False:
        os.system('mkdir ' + path_output)
    
    t = np.genfromtxt(rvFile, usecols=(0))
    rv = np.genfromtxt(rvFile, usecols=(1))
    rv_err = np.genfromtxt(rvFile, usecols=(2))
    N = len(rv)
    rms_rv = np.sqrt(np.sum(rv**2)/N)
    
    Pmin = rangeP[0]
    Pmax = rangeP[1]
    Kmin = rangeK[0]*rms_rv
    Kmax = rangeK[1]*rms_rv
    
    P = np.zeros(Nsamples)
    M = np.zeros(Nsamples)
    detect_rate = np.zeros(Nsamples)
    phase = np.linspace(-np.pi, np.pi, num=Nphases, endpoint=False)
    
    for i in tqdm(range(Nsamples)):
        logP = uniform(np.log10(Pmin), np.log10(Pmax))
        P[i] = 10**logP
        K = uniform(Kmin, Kmax)
        M[i] = (K/28.435) * Mstar**(2./3.) * (P[i]/365.25)**(1./3.) # [M_Jup]
        M[i] = M[i] * 1.2668653e17 / 3.986004e14 # [M_Earth]

        detect = int(0)
        for j in range(Nphases):
            rv_simu = rv - K * np.sin((t*2*np.pi/P[i])+phase[j])

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
    
    
    bins = np.linspace(np.log10(Pmin), np.log10(Pmax), nbins+1)

    sub_P = []
    sub_M = []
    for i in range(len(P)):
        if detect_rate[i] < 0.9999:
            sub_P = np.append(sub_P, P[i])
            sub_M = np.append(sub_M, M[i])

    digitized = np.digitize(np.log10(sub_P), bins)
    subP_means = [sub_P[digitized == i].mean() for i in range(1, len(bins))]

    M_bins = [sub_M[digitized == i] for i in range(1, len(bins))]
    q = 95 / 100.
    M95 = [np.quantile(M_bins[i],q) for i in range(nbins)] # The 95% mass limit of data-driven detection for each period bin

#     os.chdir('/Users/manustalport/Documents/CHEOPS_ProdexLiege/DynamicalAnalyses/DynamicalDetectionLimits/DataDrivenAlgo_Custom')
#    if os.path.isdir(sys_name) == False:
#        os.mkdir(sys_name)
#    os.chdir(sys_name)
    os.chdir(path_output)
    file = open('Data-driven_95MassLimits.dat', 'w')
    file.write('Period[days] Mass[M_Earth]' + '\n')
    file.write('------------ -------------' + '\n')
    for i in range(len(M95)):
        file.write(str(subP_means[i]) + ' ' + str(M95[i]) + '\n')
    file.close()
    
    file = open('Injection-recovery_tests.dat', 'w')
    file.write('Period[days] Mass[M_Earth] DetectionRate[%]' + '\n')
    file.write('------------ ------------- ----------------' + '\n')
    for i in range(len(P)):
        file.write(str(P[i]) + ' ' + str(M[i]) + ' ' + str(detect_rate[i]) + '\n')
    file.close()
    
    if plot == True:
        detect_rate = detect_rate * 100.
        if Nphases < 8:
            cmap = plt.get_cmap('gnuplot', Nphases)
        else:
            cmap = plt.get_cmap('gnuplot', 8)
        
        fig = plt.figure(figsize=(6,4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.scatter(P, M, c=detect_rate, s=10.0, alpha=0.4, edgecolors='black', linewidths=0.2, cmap=cmap)
        plt.plot(subP_means, M95, color='black')
        plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
        plt.xscale('log')
        plt.ylim(0, 1.05*max(M))
        cb = plt.colorbar(label=r'\large{Detection rate [$\%$]}', pad=0., ticks=[0.,25., 50., 75., 100.])
        plt.xlabel(r'\large{Period [d]}')
        plt.ylabel(r'\large{Mass [M$_{\oplus}$]}')
        plt.tick_params(labelsize=11)
        plt.tight_layout()
        plt.savefig('Data-driven_DL_'+sys_name+'.png', format='png', dpi = 300)
    
    os.chdir(cwd)
    



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODULE 2: dynamical detection limits %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
################### PART NAFF
def naf(
  t,
  f,
  nfreqs=5,
  circ=2 * np.pi,
  tstep=None,
  secant_xtol=1e-15,
  secant_relftol=1,
  secant_Nmax=10,
  basis_maxprod=0.5,
):
    """NAFF algorithm (see Laskar, 2003)
    t, f are arrays of time (equally spaced) and signal values.
    nfreqs is the maximum desired number of frequencies in the decomposition.
    Depending on the signal the algorithm may return less terms than nfreqs.
    circ is the circumference of a unit circle (default is 2.pi but can be 360, 1, etc.)
    tstep = is the time step. If not provided it is set to t[1]-t[0].
    """
    N = len(t)
    N = N - N % 6
    step = t[1] - t[0] if tstep is None else tstep
    T = N / 2 * step
    x = t[:N]
    g = f[:N].astype(complex)
    nf = nfreqs
    freqs = np.empty(nf)
    amps = np.empty(nf, dtype=complex)
    vecs = np.empty((nf, N), dtype=complex)
    basis = np.zeros((nf, nf), dtype=complex)
    no_more_freq = False
    for n in range(nf):
        freqs[n], vecs[n, :], amps[n] = _naf_findmax(
          x,
          g * _naf_xi((x - x[0]) / T - 1),
          step,
          N,
          secant_xtol,
          secant_relftol,
          secant_Nmax,
        )
        # Check if frequency has already been detected
        for k in range(n):
            basis[n, k] = _naf_prod(vecs[n, :] * _naf_xi((x - x[0]) / T - 1), vecs[k, :])
            if abs(basis[n, k]) > basis_maxprod:
                no_more_freq = True
        # stop the algorithm if it is the case
        if no_more_freq:
            nf = n
            freqs = freqs[:n]
            amps = amps[:n]
            basis = basis[:n, :n]
            break
        # completing orthonormal basis and projecting
        # module of the orthogonal vector
        mod = np.sqrt(1 - sum(abs(basis[n, k]) ** 2 for k in range(n)))
        # complete basis matrix and nth vector
        basis[n, n] = mod
        vecs[n, :] = vecs[n, :] / mod
        amps[n] = amps[n] / mod
        for k in range(n):
            vecs[n, :] -= basis[n, k] * vecs[k, :] / mod
        g -= amps[n] * vecs[n, :]
    # perform the basis change
    amps = np.linalg.solve(basis.T, amps)
    return (freqs * circ / (2 * np.pi), amps)


def nafprint_header(fundnames, fundfreqs, outfile=None):
    """Write the header of a naf output file
    If outfile is not provided, the standard output is used
    """
    print('Fundamental frequencies', file=outfile)
    for k in range(len(fundfreqs)):
        print(fundnames[k] + ':', fundfreqs[k], file=outfile)


def _naf_xi(t):
    """Weight function for the NAFF algorithm (see Laskar, 2003)."""
    return 1 + np.cos(np.pi * t)


def _naf_findmax(t, f, step, N, xtol, relftol, Nmax):
    """Find the maximum amplitude of the Fourier transform of f"""

    # Derivative of the amplitude
    def dAmp(nu):
        einu = np.exp(1j * nu * t)
        return 2 * np.real(1j * _naf_prod(f, einu) * np.conj(_naf_prod(t * f, einu)))

    # ifft of f to find a starting value for the secant method
    Amp = np.abs(np.fft.ifft(f))
    # Frequencies given by the ifft
    nus = -2 * np.pi / step * np.fft.fftfreq(N)
    dnu = 2 * np.pi / step / N
    # Frequency at the ifft maximum
    nu0 = nus[np.argmax(Amp)]
    # Derivative of amplitude
    dA0 = dAmp(nu0)
    # Find the zero of the derivative (maximum of amplitude)
    if dA0 > 0:
        nu = _naf_secant(dAmp, nu0 + dnu, nu0, xtol, relftol, Nmax)
    else:
        nu = _naf_secant(dAmp, nu0 - dnu, nu0, xtol, relftol, Nmax)
    einut = np.exp(1j * nu * t)
    return (nu, einut, _naf_prod(f, einut))


def _naf_prod(f, g):
    """Integral of f * conj(g)"""
    return sum(f * np.conj(g)) / len(f)


def _naf_secant(f, a0, b0, xtol, relftol, Nmax):
    """Secant method to find the zero of f"""
    a = a0
    b = b0
    fa = f(a0)
    fb = f(b0)
    N = 0
    while fb != 0 and abs(a - b) > xtol and abs(fa / fb - 1) > relftol and Nmax > N:
        c = b + (a - b) * fb / (fb - fa)
        a = b
        fa = fb
        b = c
        fb = f(b)
        N += 1
    return b

  
################### HILL RADIUS
def HillRad(a, Mp, Mstar):
    """Hill radius [AU]:
    r_H = a * (m1/(3*m0))**(1./3.)
    , where the indexes 0, 1 refer to the star, and planet"""

    mE_S = 3.986004e14 / 1.3271244e20
    r_H = a * (Mp/(3*Mstar))**(1./3.)
    
    return r_H

################### DYNAMICAL EVOLUTION AND STABILITY ESTIMATION
def Stability(KepParam, ML, Mstar, Nplanets, T, dt, min_dist, max_dist, NAFF, Noutputs=int(20000), NAFF_Thresh=0., GR=1):
    """Function for the dynamical evolution followed with stability estimation of the orbits.
    KepParam: input Keplerian parameters. This is a 2D vector of the form: KepParam = [a, lam, ecc, w, inc, Omega, Mass] where
        a is semi-major axis [AU]
        lam is mean longitude [rad]
        ecc is orbital eccentricity
        w is argument of perisatron [rad]
        inc is orbital inclination [[rad]
        Omega is longitude of ascending node [rad]
        Mass is planetary mass [M_Sun]
    Mstar: stellar mass [M_Sun]
    Nplanets: number of KNOWN planets (not counting the injected body)
    T: Total integration time
    dt: rebound integration timestep
    min_dist and max_dist: The minimum and maximum allowed distances, respectively. The former serves as close encounter criterion, the latter as escape criterion.
    NAFF (bool): True if NAFF computation is desired after the numerical integration. False if no NAFF computation needed.
    Noutputs: the number of output times desired to compute the orbital stability. Default is 20000.
    NAFF_Thresh: The NAFF threshold above which the system is considered unstable. Default is 0. """
    Nbodies = int(Nplanets + 1) # The total number of bodies excluding the star, i.e. Nplanets + 1 injected planet
    
    a = KepParam[0]
    phase = KepParam[1]
    e = KepParam[2]
    w = KepParam[3]
    inc = KepParam[4]
    O = KepParam[5]
    M = KepParam[6]
    
    if ML == True: # Mean Longitude is provided
        sim = rebound.Simulation()
        sim.add(m=Mstar)
        sim.add(m=M[0], a=a[0], l=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=M[q], a=a[q], l=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q]) # Infos sur les definitions des parametres orbitaux et leurs "unites": voir 'REBOUND: infos pratiques' dans Notes.
            q += 1
            
    elif ML == False: # Mean Anomaly is provided
        sim = rebound.Simulation()
        sim.add(m=Mstar)
        sim.add(m=M[0], a=a[0], M=phase[0], omega=w[0], e=e[0], Omega=O[0], inc=inc[0])
        q = int(1)
        while q < len(a):
            sim.add(primary=sim.particles[0], m=M[q], a=a[q], M=phase[q], omega=w[q], e=e[q], Omega=O[q], inc=inc[q]) # Infos sur les definitions des parametres orbitaux et leurs "unites": voir 'REBOUND: infos pratiques' dans Notes.
            q += 1
      
    sim.move_to_com()
    sim.dt = dt
    sim.integrator = "ias15" # REMARQUE: integrateur non symplectique utilise par soucis de generalite avec l'etude dynamique: on veut pouvoir etendre l'etude aux systemes binaires.

    sim.exit_max_distance = max_dist
    sim.exit_min_distance = min_dist
    
    if GR == 1:
        ##*************** REBOUNDX PART *****************
        rebx = reboundx.Extras(sim)
        gr = rebx.load_force("gr")
        rebx.add_force(gr)
        gr.params["c"] = constants.C
        ##************************************************
#        eilambda = np.zeros((Nbodies, Noutputs))
#        time = np.zeros(Noutputs)
#    elif GR == 0:
#        eilambda = np.zeros((Nbodies, Noutputs))
#        time = np.zeros(Noutputs)

    if NAFF == True:
        eilambda = np.zeros((Nbodies, Noutputs))
        time = np.zeros(Noutputs)
        dt_output = round(T / (Noutputs * dt / (2*np.pi))) # The number of timesteps between consecutive outputs

        try:
            for j in range(Noutputs):
                t_output = dt_output*(j+1)*dt
                sim.integrate(t_output) # sans le 2eme argument, on a implicitement que exact_finish_time=1
                time[j] = t_output/(2*np.pi)

                for q in range(Nbodies):
                    eilambda[q][j] = sim.particles[q+1].l
                    
            ######### stability computation -- NAFF algorithm
            var_freq = np.zeros(Nbodies)
            t_NAFF = np.arange(1,int(Noutputs/2 + 1)) # Vecteur d'entiers allant de 1 a Noutputs/2 inclus
            Delta_t = time[0]

            for i in range(Nbodies):
                freq1, amp = naf(t_NAFF, eilambda[i][:int(Noutputs/2)],1)
                freq1 = freq1 / Delta_t
                freq2, amp = naf(t_NAFF, eilambda[i][int(Noutputs/2):],1)
                freq2 = freq2 / Delta_t
                #################################################################################
                var_freq_1 = abs(freq2-freq1 - 1./(2*Delta_t)).item()
                var_freq_2 = abs(freq2-freq1).item()
                var_freq_3 = abs(freq2-freq1 + 1./(2*Delta_t)).item()
                delta_n = min(var_freq_1,var_freq_2,var_freq_3)

                mE_S = 3.986004e14 / 1.3271244e20
                n = a[i]**(-1.5) * 2*np.pi * ((Mstar+M[i])/(1+mE_S))**0.5
                var_freq[i] = np.log10(abs(delta_n/n))

            NAFF_max = max(var_freq)

            if NAFF_max < NAFF_Thresh:
                stab = 1.
            else:
                stab = 0.
                    

        except rebound.Escape:
            stab = 0.

        except rebound.Encounter:
            stab = 0.

        return stab
        
        
    if NAFF == False:
        a_AU = np.zeros((Nbodies, Noutputs))
        ecc = np.zeros((Nbodies, Noutputs))
        w_rad = np.zeros((Nbodies, Noutputs))
        inc_rad = np.zeros((Nbodies, Noutputs))
        O_rad = np.zeros((Nbodies, Noutputs))
        try:
            time = np.zeros(Noutputs)
            dt_output = round(T / (Noutputs * dt / (2*np.pi))) # The number of timesteps between consecutive outputs
            
            file = open('LongTermEvo.dat', 'w')
            file.write('a[AU] ecc w[rad] inc[rad] O[rad]' + '\n')
            file.write('----- --- ------ -------- ------' + '\n')
            for j in range(Noutputs):
                t_output = dt_output*(j+1)*dt
                sim.integrate(t_output) # sans le 2eme argument, on a implicitement que exact_finish_time=1
                time[j] = t_output/(2*np.pi)

                for q in range(Nbodies):
#                    a_AU[q][j] = sim.particles[q+1].a
#                    ecc[q][j] = sim.particles[q+1].e
#                    w_rad[q][j] = sim.particles[q+1].omega
#                    inc_rad[q][j] = sim.particles[q+1].inc
#                    O_rad[q][j] = sim.particles[q+1].Omega
                    file.write(str(sim.particles[q+1].a) + ' ' + str(sim.particles[q+1].e) + ' ' + str(sim.particles[q+1].omega) + ' ')
                    file.write(str(sim.particles[q+1].inc) + ' ' + str(sim.particles[q+1].Omega) + '\n')
            
            file.close()
            return stab = 1
                    
        except rebound.Escape:
            stab = 0

        except rebound.Encounter:
            stab = 0

        return stab


################### ANALYTIC STABILITY CRITERION
def AnalyticStability(Mstar, a, e, M):
    """
    Analytic estimation of orbital stability following the AMD framework (Laskar & Petit 2017). This function combines the Hill AMD stability criterion of Petit & Laskar (2018) with the AMD criterion in presence of first-order mean-motion resonances (MMR) of Petit & Laskar (2017).
    
    Arguments
    ---------
    Mstar (float): the stellar mass [M_Sun]
    a, e, M (1D arrays): the semi-major axis [AU], orbital eccentricity, and mass [M_Sun] of all the planets in the modelled system.
    
    Output
    ------
    analytic_stab (int): indicator of orbital stability. 1: AMD-stable ; 0: AMD-unstable, indicating that further numerical investigations are needed.
    """
    NB_pairs = len(a) - 1
    # ---------- Define G
    G = 6.6743*10**(-11) # The universal gravitation constant, in units of m^3/(kg*s^2)  ;  Value from CODATA 2018
    ### Convert G in units of AU, Solar mass and year
    G = G * 1.9884*10**30 # Converting kg in solar mass
    G = G / (149597870700**3) # Converting meters in AU
    G = G * 31557600**2 # Converting seconds in years
        
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
        C[i] = Lambda_ext[i]*(1-np.sqrt(1-e[i]**2)) + Lambda_ext[i+1]*(1-math.sqrt(1-e[i+1]**2))
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
    orb_cross (bool): True if the orbits of at least one planet pair cross. False otherwise.
    """
    orb_cross = False
    NB_pairs = len(a) - 1
    for i in range(NB_pairs):
        apo_inner = a[i] * (1+e[i])
        peri_outer = a[i+1] * (1-e[i+1])
        if apo_inner >= peri_outer:
            orb_cross = True
    
    return orb_cross


#############################
################### MAIN CODE
#def DynDL(shift, Nplanets, param_file, DataDrivenLimitsFile, output_file, DetectLim0File):
def DynDL(sys_name, shift, Nplanets, param_file, DataDrivenLimitsFile):
    """
    Computation of the dynamical detection limits
    
    Arguments
    ---------
    shift (int): index indicating at which value of the period one computes the dynamical detection limits (for parallel computations)
    Nplanets (int): the number of known planets in the system (not counting for the injected one)
    param_file (string): name of the parameters file
    DataDrivenLimitsFile (string): name of the data-driven detection limits file
    output_file (string): name of the extensive output file
    DetectLim0File (string): name of the dynamical detection limits output file
    """
    cwd = os.getcwd() # Current Working Directory
    path_output = cwd + '/' + sys_name + '_DetectionLimits'
    if os.path.isdir(path_output) == False:
        os.system('mkdir ' + path_output)
        
    # ---------- Define constants
    mE_S = 3.986004e14 / 1.3271244e20 # Earth-to-Solar mass ratio
    mJ_S = 1.2668653e17 / 1.3271244e20 # Jupiter-to-Solar mass ratio
    
    output_file = "AllStabilityRates.dat"
    DetectLim0File = "Final_DynamicalDetectLim.dat"
    # --- If this is the first call to this function, create the output files
    if shift == int(0):
        os.chdir(path_output)
        file = open(output_file, 'a')
        file.write('Period Mass Stability_rate' + '\n')
        file.write('------ ---- --------------' + '\n')
        file.close()

        file0 = open(DetectLim0File, 'a')
        file0.write('Period Mass Stability_rate' + '\n')
        file0.write('------ ---- --------------' + '\n')
        file0.close()
        os.chdir(cwd)
    
    # ---------- Get the parameters
    param_names = np.genfromtxt(param_file, usecols=(0), dtype=None, encoding=None)
    param_values = np.genfromtxt(param_file, usecols=(1))

    P = np.zeros(Nplanets+1)
    phase = np.zeros(Nplanets+1)
    e = np.zeros(Nplanets+1)
    w = np.zeros(Nplanets+1)
    inc = np.zeros(Nplanets+1)
    O = np.zeros(Nplanets+1)
    K = np.zeros(Nplanets+1)
    ### Derived planet parameters
    a = np.zeros(Nplanets+1) # semi-major axis [AU]
    M = np.zeros(Nplanets+1) # planet mass [MSun]
    
    Mstar = -9999
    T = -9999
    dt = -9999
    # Same for the optional parameters
    Nphases = -9999
    min_dist = -9999
    max_dist = -9999
    Noutputs = -9999
    NAFF_Thresh = -9999
    GR = -9999

    for j in range(len(param_names)):
        if param_names[j] == 'Mstar':
            Mstar = param_values[j]
        if param_names[j] == 'Nphases':
            Nphases = int(param_values[j])
#            Nphases_given = True
        if param_names[j] == 'e_inject':
            e[-1] = param_values[j]
        if param_names[j] == 'w_inject':
            w[-1] = param_values[j]
        if param_names[j] == 'i_inject':
            inc[-1] = param_values[j]
        if param_names[j] == 'O_inject':
            O[-1] = param_values[j]
        if param_names[j] == 'T':
            T = param_values[j]
        if param_names[j] == 'dt':
            dt = param_values[j] * 2*np.pi # Conversion to REBOUND units: 1yr = 2pi
        if param_names[j] == 'min_dist':
            min_dist = param_values[j]
#            mindist_given = True
        if param_names[j] == 'max_dist':
            max_dist = param_values[j]
#            maxdist_given = True
        if param_names[j] == 'Noutputs':
            Noutputs = int(param_values[j])
#            Nout_given = True
        if param_names[j] == 'NAFF_Thresh':
            NAFF_Thresh = param_values[j]
#            NAFFthresh_given = True
        if param_names[j] == 'GR':
            GR = int(param_values[j])
#            GR_given = True
         
    # --- Raise exception if some necessary parameter was not specified in the parameters file
    if Mstar == -9999:
        raise Exception(f"Mstar not found in the parameters file")
#    if T == -9999:
#        raise Exception(f"T (total integration time) not found in the parameters file")
#    if dt == -9999:
#        raise Exception(f"dt (integration timestep) not found in the parameters file")
    
    # --- Define some defaults values if not specified in the parameters file
    if Nphases == -9999:
        Nphases = int(3)
    if min_dist == -9999:
        min_dist = 1.
    if max_dist == -9999:
        max_dist = 5.
    if Noutputs == -9999:
        Noutputs = int(20000)
    if NAFF_Thresh == -9999:
        NAFF_Thresh = 0.0
    if GR == -9999:
        GR = int(0)
        
    # --- Get the known planet's parameters
    for i in range(Nplanets):
        for j in range(len(param_names)):
            if param_names[j] == 'P_%d' %(i+1):
                P[i] = param_values[j] / 365.25 #[years]
            if param_names[j] == 'ML_%d' %(i+1):
                phase[i] = param_values[j]
                ML = True
            if param_names[j] == 'MA_%d' %(i+1):
                phase[i] = param_values[j]
                ML = False # Input is the Mean Anomaly MA instead of the Mean Longitude ML
            if param_names[j] == 'e_%d' %(i+1):
                e[i] = param_values[j]
            if param_names[j] == 'w_%d' %(i+1):
                w[i] = param_values[j]
            if param_names[j] == 'i_%d' %(i+1):
                inc[i] = param_values[j]
            if param_names[j] == 'O_%d' %(i+1):
                O[i] = param_values[j]
            if param_names[j] == 'K_%d' %(i+1):
                K[i] = param_values[j]

        M[i] = (K[i] / 28.435) * P[i]**(1./3.) * Mstar**(2./3.)   # in Jupiter masses
        M[i] = M[i] * mJ_S

        a[i] = P[i]**(2./3.) * ((Mstar+M[i])/(1.+mE_S))**(1./3.)
    
    # ---------- Extract the period and mass of the injected planet, according to the data-driven detection limits. Then convert P to a.
    P_inject = np.genfromtxt(DataDrivenLimitsFile, usecols=(0), skip_header=(2))
    M_lim100 = np.genfromtxt(DataDrivenLimitsFile, usecols=(1), skip_header=(2)) # The 100% detection threshold mass for each period
    
    P_inject = P_inject / 365.25 # [yr]
    M_lim100 = M_lim100 * mE_S # [M_Sun]

    M[-1] = M_lim100[shift]
    P[-1] = P_inject[shift]
    a[-1] = P_inject[shift]**(2./3.) * ((Mstar+M[-1])/(1.+mE_S))**(1./3.)
    
    
    # ---------- Sort the parameters by increasing a
    indexes = np.argsort(a)
    a = np.array(a)[indexes]
    P = np.array(P)[indexes]
    phase = np.array(phase)[indexes]
    e = np.array(e)[indexes]
    w = np.array(w)[indexes]
    inc = np.array(inc)[indexes]
    O = np.array(O)[indexes]
    M = np.array(M)[indexes]
    
    # ---------- If not specified, propose automatically an optimal integration time T and time-step dt
    if T == -9999:
        T = P[-1] * 100000
    if dt == -9999:
        dt = P[0] / 50
    
    # ---------- Convert all the angles to radians
    phase = phase * np.pi / 180.
    phase = (phase + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    w = w * np.pi / 180.
    w = (w + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    inc = inc * np.pi / 180.
    inc = (inc + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    O = O * np.pi / 180.
    O = (O + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]

    # ---------- Put the Keplerians into a 2D array
    params = [a,phase,e,w,inc,O,M] # Keep the order of the elements! a,lam,e,w,inc,O,M
    phases_inject = np.linspace(-np.pi, np.pi, Nphases, endpoint=False) # With endpoint=False, I generate Nphases points with constant interval in [start, stop[ (the last value is excluded)
    
    min_dist = min_dist * HillRad(a[0], M[0], Mstar)
    max_dist = max_dist * a[-1]
    NAFF = True

    # ---------- Start the iterative process to find the minimum mass at which stability rate = 0%
    print('Processing stability estimation at period bin ' + str(shift))
    os.chdir(path_output)
    stab_rate = AnalyticStability(Mstar, a, e, M)
    if stab_rate == 0: # i.e., if and only if the system (including the injected planet) is AMD-unstable, we perform the numerical simulations to precise the system stability
        crossing = OrbitCrossing(a, e)
        if crossing == True: # If orbits are crossing, for sure the system is unstable. No need of numerical simulations.
            stab_rate = 0.
        else:
            stab = 0.
            for j in range(Nphases):
                phase[indexes[-1]] = phases_inject[j]
                stab += Stability(params, ML, Mstar, Nplanets, T, dt, min_dist, max_dist, NAFF, Noutputs, NAFF_Thresh, GR) # stab is a binary number: 0 = unstable, 1 = stable
            stab_rate = round((stab / Nphases) * 100.) # Rate of stable systems, in %
    else:
        stab_rate = 100
        
    file = open(output_file, 'a')
    file.write(str(P_inject[shift]*365.25) + ' ' + str(M[indexes[-1]]/mE_S) + ' ' + str(stab_rate) + '\n')
    file.close()

    if stab_rate <= 0.1: # If the rate of stability is smaller than 0.1%
        Thresh = 0.5 * mE_S # mass precision criterion is 0.5 M_Earth (expressed in [M_Sun])
        dM = 1000000.
        q = int(0)
        while dM > Thresh:
            if q == 0 and M_lim100[shift] > 0.001*mE_S:
                M[indexes[-1]] = 0.001*mE_S
                a[indexes[-1]] = P_inject[shift]**(2./3.) * ((Mstar+M[indexes[-1]])/(1.+mE_S))**(1./3.)
                
                crossing = OrbitCrossing(a, e)
                if crossing == True:
                    stab_rate = 0.
                else:
                    stab = 0.
                    for j in range(Nphases):
                        phase[indexes[-1]] = phases_inject[j]
                        stab += Stability(params, ML, Mstar, Nplanets, T, dt, min_dist, max_dist, NAFF, Noutputs, NAFF_Thresh, GR) # stab is a binary number: 0 = unstable, 1 = stable
                    stab_rate = round((stab / Nphases) * 100.)
                    
                file = open(output_file, 'a')
                file.write(str(P_inject[shift]*365.25) + ' ' + str(M[indexes[-1]]/mE_S) + ' ' + str(stab_rate) + '\n')
                file.close()

                if stab_rate > 0.1:
                    M_max = M_lim100[shift]
                    M_min = 0.001*mE_S
                    dM = M_max - M_min

                else:
                    dM = 0. # Get out of the loop
                    Mlim = M[indexes[-1]]

            elif q == 0 and M_lim100[shift] == 0.001*mE_S:
                dM = 0. # Get out of the loop
                Mlim = M[indexes[-1]]

            else:
                M[indexes[-1]] = (M_max+M_min) / 2.
                a[indexes[-1]] = P_inject[shift]**(2./3.) * ((Mstar+M[indexes[-1]])/(1.+mE_S))**(1./3.)
                
                crossing = OrbitCrossing(a, e)
                if crossing == True:
                    stab_rate = 0.
                else:
                    stab = 0.
                    for j in range(Nphases):
                        phase[indexes[-1]] = phases_inject[j]
                        stab += Stability(params, ML, Mstar, Nplanets, T, dt, min_dist, max_dist, NAFF, Noutputs, NAFF_Thresh, GR)
                    stab_rate = round((stab / Nphases) * 100.)
                    
                file = open(output_file, 'a')
                file.write(str(P_inject[shift]*365.25) + ' ' + str(M[indexes[-1]]/mE_S) + ' ' + str(stab_rate) + '\n')
                file.close()

                if stab_rate > 0.1:
                    M_min = M[indexes[-1]]

                elif stab_rate <= 0.1:
                    M_max = M[indexes[-1]]

                dM = M_max - M_min
                Mlim = M_max

            q += 1

        file = open(DetectLim0File, 'a')
        file.write(str(P_inject[shift]*365.25) + ' ' + str(Mlim/mE_S) + ' ' + str(stab_rate) + '\n')
        file.close()

    else:
        file = open(DetectLim0File, 'a')
        file.write(str(P_inject[shift]*365.25) + ' ' + str(M[indexes[-1]]/mE_S) + ' ' + str(stab_rate) + '\n')
        file.close()
        
        

################### PLOT THE FINAL DETECTION LIMITS
def plot_DL(sys_name, data_driven='Data-driven_95MassLimits.dat', stability_driven='Final_DynamicalDetectLim.dat'):
    """
    Plot the detection limits, both data-driven limits and dynamical detection limits.
    
    Arguments
    ---------
    sys_name (string): Name of the system
    data_driven (string, optional): Filename of the data-driven detection limits
    stability_driven (string, optional): Filename of the dynamical detection limits
    """
    cwd = os.getcwd() # Current Working Directory
    path_output = cwd + '/' + sys_name + '_DetectionLimits'
    if os.path.isdir(path_output) == False:
        os.system('mkdir ' + path_output)
    os.chdir(path_output)
    
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
    plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
    plt.xscale('log')
    plt.xlabel(r'\Large{Period [d]}')
    plt.ylabel(r'\Large{Mass [M$_{\oplus}$]}')
    plt.tick_params(labelsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('FinalDetectionLimits_'+sys_name+'.png', format='png', dpi = 300)


################### TEST THE STABILITY OF A PLANET CANDIDATE
def LongTermStab(Mstar, Keplerians, T, dt, min_dist, max_dist, Noutputs=int(1000), ML=True, GR=1):
    """
    Investigate the stability of a candidate planet with given Period, Mass, Orbital phase.
    
    Arguments
    ---------
    Mstar (float): Stellar mass [M_Sun]
    Keplerians (2D array): Keplerian elements of the known planets and the planet candidate
    T (int): Total integration time [yr]
    dt (float): Integration timestep [yr]
    min_dist, max_dist (float): the minimum approach and maximum distance criteria for the integration (in units of [Hill radius] and [a_outer], respectively)
    
    """
    mE_S = 3.986004e14 / 1.3271244e20 # Earth-to-Solar mass ratio
        
    P = Keplerians[0]
    phase = Keplerians[1]
    e = Keplerians[2]
    w = Keplerians[3]
    inc = Keplerians[4]
    O = Keplerians[5]
    M = Keplerians[6]
    
    P = P / 365.25
    a = P**(2./3.) * ((Mstar+M)/(1.+mE_S))**(1./3.)
    M = M * mE_S
    # ---------- Convert all the angles to radians
    phase = phase * np.pi / 180.
    phase = (phase + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    w = w * np.pi / 180.
    w = (w + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    inc = inc * np.pi / 180.
    inc = (inc + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    O = O * np.pi / 180.
    O = (O + np.pi) % (2*np.pi) - np.pi # Conversion into [-pi, pi]
    
    # ---------- Sort the parameters by increasing a
    indexes = np.argsort(a)
    a = np.array(a)[indexes]
    phase = np.array(phase)[indexes]
    e = np.array(e)[indexes]
    w = np.array(w)[indexes]
    inc = np.array(inc)[indexes]
    O = np.array(O)[indexes]
    M = np.array(M)[indexes]
    
    # ---------- Put the Keplerians into a 2D array
    params = [a,phase,e,w,inc,O,M] # Keep the order of the elements! a,lam,e,w,inc,O,M
    
    min_dist = min_dist * HillRad(a[0], M[0], Mstar)
    max_dist = max_dist * a[-1]
    NAFF = False
    
    test_stab = Stability(params, ML, Mstar, Nplanets, T, dt, min_dist, max_dist, NAFF, Noutputs, NAFF_Thresh, GR)
    
    return test_stab
    
