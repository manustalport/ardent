import getopt
import os
import pickle
import sys

import ardent_functions as ardf
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from PyAstronomy.pyTiming import pyPeriod
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d

#NlocalCPU = int(3) # The number of CPUs allocated to compute the dynamical detection limits, to fasten the computation.
#    # If NlocalCPU is set to 0, the program will be compatible with external cluster.
#    # To use the code on a cluster, import all the needed files in the execution folder. Create a virtual environement to install rebound and reboundx.
#    # Use the argument --array of sbatch to call the programme for each value of P_inject
    
# ---------- Define constants
mE_S = 3.986004e14 / 1.3271244e20 # Earth-to-Solar mass ratio
mJ_S = 1.2668653e17 / 1.3271244e20 # Jupiter-to-Solar mass ratio
mE_J = 3.986004e14 / 1.2668653e17 # Earth-to-Jupiter mass ratio

class ARDENT_tableXY(object):
    def __init__(self, x, y, yerr, ):
        self.x = np.array(x)       # jdb times
        self.y = np.array(y)       # RV time-series in m/s
        self.yerr = np.array(yerr) # RV uncertainties in m/s
        self.baseline = np.round(np.max(self.x) - np.min(self.x),0)
        self.planets = []
        self.ARDENT_AddStar()
        self.ARDENT_Set_output_dir()


    def ARDENT_Set_output_dir(self,output_dir=None):
        if output_dir is None:
            output_dir = os.getcwd()+'/output/'
            self.output_dir = output_dir
        else:
            self.output_dir = output_dir+'/'
            
        if os.path.isdir(output_dir) == False:
            os.system('mkdir ' + output_dir)
            
        self.tag = self.output_dir+self.starname+'_'
        
        
    def ARDENT_AddStar(self,mass=1.00,starname='Sun'):
        self.starname = starname
        self.mstar = mass


    def ARDENT_Plot(self, xUnits='BJD-2457000', yUnits='m/s', rangePeriodogram=[2.,200.], fapLevel=0.01, new=True):
        if new:
            fig = plt.figure(figsize=(11,3.5))
        plt.rc('font', size=12)
        
        ##### RV residuals timeseries plot
        plt.subplot(1,2,1)
        RVtimespan = (max(self.x) - min(self.x))
        tmin = min(self.x) - RVtimespan / 20
        tmax = max(self.x) + RVtimespan / 20
        plt.errorbar(self.x,self.y,yerr=self.yerr,color='k',capsize=0,marker='o',ls='',alpha=0.3)
        plt.xlabel('Time [' + xUnits + ']', size='large')
        plt.ylabel('RV [' + yUnits + ']', size='large')
        plt.xlim(tmin,tmax)
        plt.title(self.starname + ' RV timeseries', size='large')

        ##### Periodogram plot
        plt.subplot(1,2,2)
        clp = pyPeriod.Gls((self.x, self.y), Pbeg=rangePeriodogram[0], Pend=rangePeriodogram[1])
        p_RV = clp.power
        plevel_RV = clp.powerLevel(fapLevel)
        plt.plot(1/clp.freq, p_RV, 'k', lw=1.2, rasterized=True, alpha=0.3)
        plt.axhline(plevel_RV, c='k', lw=1.0, linestyle='--', zorder=2, label='FAP ' + str(round(fapLevel*100)) + '%')
        plt.xlim(rangePeriodogram[0], rangePeriodogram[1])
        plt.xscale('log')
        plt.ylabel('Power', size='large')
        plt.xlabel('Period [days]', size='large')
        plt.legend()
        plt.title(self.starname + ' GLS periodogram', size='large')

        plt.tight_layout()
        plt.subplots_adjust(left=0.09,right=0.95,wspace=0.20)
        plt.savefig(self.tag+'RVresiduals.png', format='png', dpi = 300)


    def ARDENT_ResetPlanets(self):
        self.planets = []


    def ARDENT_ImportPlanets(self,filename):
        """
        Import the parameters of the known planets (mass and orbital elements) from an imput file.
        """
        param_names = np.genfromtxt(filename, usecols=(0), dtype=None, encoding=None)
        param_values = np.genfromtxt(filename, usecols=(1))
        self.mstar = param_values[param_names=='Mstar'][0]
        conv = 180/np.pi

        nb_planet = np.sum([p.split('_')[0]=='P' for p in param_names])
        phase_param = np.sum([p.split('_')[0]=='ML' for p in param_names])
        if phase_param > 0:
            ML = True
        else:
            ML = False
            
        for i in np.arange(1,1+nb_planet):
            p = param_values[param_names=='P_%.0f'%(i)][0]
            k = param_values[param_names=='K_%.0f'%(i)][0]
            e = param_values[param_names=='e_%.0f'%(i)][0]
            omega = param_values[param_names=='w_%.0f'%(i)][0] # [deg]
            inc = param_values[param_names=='inc_%.0f'%(i)][0] # [deg]
            asc_node = param_values[param_names=='asc_node_%.0f'%(i)][0] # [deg]
            if ML == True:
                mean_long = param_values[param_names=='ML_%.0f'%(i)][0] # [deg]
                mean_anomaly = np.nan
            else:
                mean_anomaly = param_values[param_names=='MA_%.0f'%(i)][0] # [deg]
                mean_long = np.nan

            mass,semi_axis = ardf.AmpStar(self.mstar, p, k, e=e, i=inc)

            self.planets.append([p, semi_axis, mean_long, mean_anomaly, e, omega, inc, asc_node, k, mass])
        self.ARDENT_ShowPlanets()


    def ARDENT_AddPlanets(self, p=365.25, semi_major=np.nan, mean_long=0.0, mean_anomaly=np.nan, e=0.0, omega=0.0, inc=90.0, asc_node=0.0, k=0.10, mass=np.nan):
        """
        Add a planet to ARDENT, by specifying its mass and orbital elements. Either a mean_long or mean_anomaly parameters are needed.
        
        Arguments
        ---------
            p : orbital period [days]
            semi_major : semi_major axis [AU]
            mean_long : mean longitude [deg]
            mean_anomaly : mean anomaly [deg]
            e : orbital eccentricity
            omega : argument of periastron [deg]
            inc : orbital inclination [deg]
            asc_node : longitude of the ascending node [deg]
            k : RV semi-amplitude [m/s]
            mass : planetary mass [M_Earth]
        """

        if (semi_major!=semi_major)|(mass!=mass):
            mass_comp,semi_major_comp = ardf.AmpStar(self.mstar, p, k, e=e, i=inc)
        
        if mass!=mass:
            mass = mass_comp
            print('\n [INFO] Mass calculated to be %.2f Earth mass'%(mass))
        
        if semi_major!=semi_major:
            semi_major = semi_major_comp
            print('\n [INFO] Semi-major axis calculated to be %.2f AU'%(semi_major))

        self.planets.append([p, semi_major, mean_long, mean_anomaly, e, omega, inc, asc_node, k, mass])
        self.ARDENT_ShowPlanets()


    def ARDENT_ShowPlanets(self):
        """
        Display the parameters of all the planets currently saved in ARDENT.
        """
        printed_table = pd.DataFrame(self.planets,columns=['period','semimajor','mean_long','mean_anomaly','ecc','periastron','inc','asc_node','semi-amp','mass'])
        printed_table.index = ['planet %.0f'%(i) for i in range(1,1+len(self.planets))]
        print('\n [INFO] Planet parameters: \n')
        print(printed_table)
        self.planets_table = printed_table.astype('float')


    def ARDENT_PlotPlanets(self, new=True, savefig=True, legend=True):
        """
        Plot the planetary orbits.
        """
    
        table = pd.DataFrame(self.planets,columns=['period','semimajor','mean_long','mean_anomaly','ecc','periastron','inc','asc_node','semi-amp','mass'])
        phases = np.linspace(-np.pi,np.pi,1000)
        phases2 = np.linspace(0,np.pi,1000)
        
        a = np.array(table['semimajor'])
        b = np.array(table['semimajor'])*np.sqrt(1-np.array(table['ecc'])**2)
        c = np.sqrt(a**2-b**2)

        x = np.cos(phases)*a[:,np.newaxis] +c[:,np.newaxis]
        y = np.sin(phases)*b[:,np.newaxis]

        if new:
            plt.figure(figsize=(5,5))
        plt.axis('equal')
        plt.rc('font', size=12)

        plt.scatter([0],[0],color='k',marker='x',label=r'%.2f $M_{\odot}$'%(self.mstar))
        for xi,yi,wi,mi in zip(x,y,np.array(table['periastron']),np.array(table['mass'])):
            X = np.array([xi,yi])
            R = np.array([[np.cos(np.pi/180*wi),-np.sin(np.pi/180*wi)],[np.sin(np.pi/180*wi),np.cos(np.pi/180*wi)]])
            X = np.dot(X.T,R).T
            plt.plot(X[0],X[1],lw=4,color='white')
            if mi<100:
                plt.plot(X[0],X[1],label=r'%.2f $M_{\oplus}$'%(mi))
            else:
                plt.plot(X[0],X[1],label=r'%.2f $M_{Jup}$'%(mi*mE_J))

#        plt.plot(np.sin(phases),np.cos(phases),color='k',ls='-.',lw=1)
#        plt.axhline(y=0,color='k',ls=':',lw=1,alpha=0.2)
#        plt.axvline(x=0,color='k',ls=':',lw=1,alpha=0.2)
        plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
        if legend:
            plt.legend(loc=2)
        plt.xlabel('X [AU]', size='large')
        plt.ylabel('Y [AU]', size='large')

        hz_inf = np.polyval(np.array([-0.47739653,  0.08719618,  2.92658674, -2.58897268,  1.11537673, -0.06598796]),self.mstar) # From Kopparapu(2013)
        hz_sup = np.polyval(np.array([ 1.15443229, -6.7284057 , 12.93412033, -8.26643736,  2.72788611,-0.15849661]),self.mstar) # From Kopparapu(2013)
        
        hmax = [hz_sup*np.cos(phases2),hz_sup*np.sin(phases2)]
        hmin = [hz_inf*np.cos(phases2),hz_inf*np.sin(phases2)]

        hmin[1] = interp1d(hmin[0], hmin[1], kind='linear', bounds_error=False, fill_value=0)(hmax[0])

        plt.fill_between(hmax[0],hmin[1],hmax[1],alpha=0.1,color='g', linewidth=0.0)
        plt.fill_between(hmax[0],-hmin[1],-hmax[1],alpha=0.1,color='g', linewidth=0.0)
        if savefig:
            ax = plt.gca() ; xlim = ax.get_xlim()[1]
            plt.savefig(self.tag+'Planetary_system.png', format='png', dpi = 300)
            plt.xlim(-1.25,1.25)
            plt.ylim(-1.25,1.25)
            plt.savefig(self.tag+'Planetary_system_zoom.png', format='png', dpi = 300)
            plt.xlim(-xlim,xlim)
            plt.ylim(-xlim,xlim)


#    def ARDENT_Plot_MapUpperMass(self, InjectionRecoveryFile=None, DynDLfile=None, x_au=1.25, detection_limit='RV', interp='zero'):
#
#        mstar = self.mstar
#
#        if detection_limit=='RV':
#            if InjectionRecoveryFile is None:
#                statistic = ardf.Stat_DataDL(self.output_file_DL, percentage=95, nbins=10, axis_y_var='M')
#            elif InjectionRecoveryFile is not None:
#                statistic = ardf.Stat_DataDL(InjectionRecoveryFile, percentage=95, nbins=10, axis_y_var='M')
#        else:
#            if DynDLfile is None:
#                P = np.genfromtxt(self.output_file_STDL2, usecols=(0), skip_header=int(2))
#                M_stb = np.genfromtxt(self.output_file_STDL2, usecols=(1), skip_header=int(2))
#                statistic = [P,M_stb]
#            elif DynDLfile is not None:
#                P = np.genfromtxt(DynDLfile, usecols=(0), skip_header=int(2))
#                M_stb = np.genfromtxt(DynDLfile, usecols=(1), skip_header=int(2))
#                statistic = [P,M_stb]
#
#        a = (statistic[0]/365.25)**(2./3.) * ((mstar+statistic[1]*mE_S)/(1.+mE_S))**(1./3.)
#
#        grid = np.linspace(-x_au,x_au,1000)
#        Gx,Gy = np.meshgrid(grid,grid)
#        R = np.ravel(np.sqrt(Gx**2+Gy**2))
#        M = interp1d(a, statistic[1], kind=interp, bounds_error=False, fill_value=np.nan)(R)
#        M = np.reshape(M,(1000,1000))
#
#        plt.rc('font', size=12)
#        plt.pcolormesh(Gx,Gy,M,vmin=0,vmax=16,cmap='gnuplot')
#        ax = plt.colorbar(pad=0)
#        ax.ax.set_ylabel('95$\%%$ Mass limit detection', size='large')


    def ARDENT_Plot_MapUpperMass(self, InjectionRecoveryFile=None, DynDLfile=None, x_au=1.25, detection_limit='RV', interp='zero'):
        """
        Plot the grid of mass detection limits in the orbital plane.
        """

        mstar = self.mstar

        if InjectionRecoveryFile is None:
            statistic = ardf.Stat_DataDL(self.output_file_DL, percentage=95)
            Mmax = max(statistic[1])
        elif InjectionRecoveryFile is not None:
            statistic = ardf.Stat_DataDL(InjectionRecoveryFile, percentage=95)
            Mmax = max(statistic[1])

#        if detection_limit=='RV':
#            if InjectionRecoveryFile is None:
#                statistic = ardf.Stat_DataDL(self.output_file_DL, percentage=95)
#            elif InjectionRecoveryFile is not None:
#                statistic = ardf.Stat_DataDL(InjectionRecoveryFile, percentage=95)
        if detection_limit!='RV':
            if DynDLfile is None:
                P = np.genfromtxt(self.output_file_STDL2, usecols=(0), skip_header=int(2))
                M_stb = np.genfromtxt(self.output_file_STDL2, usecols=(1), skip_header=int(2))
                statistic = [P,M_stb]
            elif DynDLfile is not None:
                P = np.genfromtxt(DynDLfile, usecols=(0), skip_header=int(2))
                M_stb = np.genfromtxt(DynDLfile, usecols=(1), skip_header=int(2))
                statistic = [P,M_stb]
                
        a = (statistic[0]/365.25)**(2./3.) * ((mstar+statistic[1]*mE_S)/(1.+mE_S))**(1./3.)

        grid = np.linspace(-x_au,x_au,1000)
        Gx,Gy = np.meshgrid(grid,grid)
        R = np.ravel(np.sqrt(Gx**2+Gy**2))
        M = interp1d(a, statistic[1], kind=interp, bounds_error=False, fill_value=np.nan)(R)
        M = np.reshape(M,(1000,1000))

        plt.rc('font', size=12)
        plt.pcolormesh(Gx,Gy,M,vmin=0,vmax=round(Mmax),cmap='gnuplot')
        ax = plt.colorbar(pad=0)
        ax.ax.set_ylabel('95$\%%$ Mass limit detection', size='large')


    def ARDENT_FinalPlot(self, InjectionRecoveryFile=None, DynDLfile=None):#, interp='zero'):
        """
        Plot the mass detection limits in the orbital plane, together with the planetary orbits of the known planets. This function produces and compares plots of data-driven and dynamical detection limits.
        
        Arguments (optional)
        ---------
        InjectionRecoveryFile (string): output file name of the injection-recovery tests
        DynDLfile (string): name of the file containing the dynamical detection limits
        """
        
        fig = plt.figure(figsize=(14,7))
        #plt.title(self.starname)
        plt.rc('font', size=12)
        
        plt.subplot(1,2,1)
        self.ARDENT_PlotPlanets(new=False,savefig=False) ; ax = plt.gca() ; xlim = ax.get_xlim()[1]
        self.ARDENT_Plot_MapUpperMass(InjectionRecoveryFile, DynDLfile, x_au=xlim,detection_limit='RV',interp='zero')
        plt.title('RV detection limits', size='large')

        plt.subplot(1,2,2)
        self.ARDENT_PlotPlanets(new=False,savefig=False, legend=False)
        self.ARDENT_Plot_MapUpperMass(InjectionRecoveryFile, DynDLfile, x_au=xlim,detection_limit='RV+Stab',interp='zero')
        plt.title('RV + stability detection limits', size='large')

        plt.subplots_adjust(left=0.09,right=0.95,wspace=0.20)
        plt.savefig(self.tag+'Summary_analysis.png', format='png', dpi = 300)

        fig = plt.figure(figsize=(14,7))

        zoom = 1.00
        plt.subplot(1,2,1)
        self.ARDENT_PlotPlanets(new=False,savefig=False,legend=False)
        self.ARDENT_Plot_MapUpperMass(InjectionRecoveryFile, DynDLfile, x_au=xlim,detection_limit='RV',interp='zero')
        plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)
        plt.title('RV detection limits', size='large')

        plt.subplot(1,2,2)
        self.ARDENT_PlotPlanets(new=False,savefig=False, legend=False)
        self.ARDENT_Plot_MapUpperMass(InjectionRecoveryFile, DynDLfile, x_au=xlim,detection_limit='RV+Stab',interp='zero')
        plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)
        plt.title('RV + stability detection limits', size='large')

        plt.subplots_adjust(left=0.09,right=0.95,wspace=0.20)
        plt.savefig(self.tag+'Summary_analysis_zoom.png', format='png', dpi = 300)


    def ARDENT_DetectionLimitRV_auto(self, rangeP=None, fap_level=0.01):
        """
        Automatic, system-independent computation of data-driven detection limits for RV data.
        
        Arguments (optional)
        ---------
        rangeP (list of floats): The range of orbital periods [days] within which to compute the detection limits. Default is 55% of the RV timeseries baseline.
        fap_level (float): The maximum False Alarm Probability (FAP) required to detect a RV signal.
        """
        
        if rangeP is None:
            rangeP = [2,int(np.round(self.baseline*0.55,-1))]

        print(' [INFO] Grid of period between: ',rangeP)
        print('\n [INFO] First iteration with low resolution')
        
        rms = np.std(self.y)
        print(' [INFO] RMS of the RV vector = %.2f m/s'%(rms))

        #first iteration with sparse sampling
        self.ARDENT_DetectionLimitRV(rangeP=rangeP, fap_level=fap_level, Nsamples=300, Nphases=6)
        file1 = self.output_file_DL

        statistic = ardf.Stat_DataDL(file1, percentage=95, nbins=4, axis_y_var='K')

        K95 = np.median(statistic[1])
        print(' [INFO] K95 detected around %.2f m/s'%(K95))
        Kmin = np.round((K95-0.30*rms)/rms,2)
        Kmax = np.round((K95+0.20*rms)/rms,2)
        if Kmin<0.10:
            Kmin = 0.10
        rangeK = [Kmin,Kmax]
        print(' [INFO] Second iteration with high resolution')
        print(' [INFO] Grid of semi-amplitude between: ',rangeK, ' [rms]')
    
        #second iteration with dense sampling
        self.ARDENT_DetectionLimitRV(rangeP=rangeP, rangeK=rangeK, fap_level=fap_level, Nsamples=700, Nphases=6)
        file2 = self.output_file_DL

        print(' [INFO] Merging of the low and high resolutions files...')
        f1 = pd.read_pickle(file1)
        f2 = pd.read_pickle(file2)
        f = {
            'P':np.hstack([f1['P'],f2['P']]),
            'K':np.hstack([f1['K'],f2['K']]),
            'M':np.hstack([f1['M'],f2['M']]),
            'detect_rate':np.hstack([f1['detect_rate'],f2['detect_rate']]),
            'Nsamples':np.hstack([f1['Nsamples'],f2['Nsamples']]),
            'Nphases':f2['Nphases'],
            'Mstar':f2['Mstar'],
            'FAP':f2['FAP'],
            'rangeP':f2['rangeP'],
            'rangeK':np.hstack([f1['rangeK'],f2['rangeK']]),
            'inc_inject':f2['inc_inject']
             }
        output_file = file2.split('InjectRecovTests')[0]+'InjectRecovTestsMerged'+file2.split('InjectRecovTests')[-1]
        pickle.dump(f,open(output_file,'wb'))
        self.output_file_DL = output_file

        plt.figure(figsize=(5,10))
        plt.subplot(2,1,1)
        self.ARDENT_Plot_DataDL(nbins=10, percentage=[95,50], axis_y_var='M', new=False)
        plt.subplot(2,1,2)
        self.ARDENT_Plot_DataDL(nbins=10, percentage=[95,50], axis_y_var='K', new=False, legend=False)
        plt.subplots_adjust(left=0.13,right=0.95,hspace=0.25,top=0.97,bottom=0.07)
        plt.tight_layout()
        plt.savefig(output_file.replace('.p','.png'), format='png', dpi = 300)


    def ARDENT_DetectionLimitRV(self, rangeP=[2., 600.], rangeK=[0.1, 1.2], inc_inject=90., fap_level=0.01, Nsamples=2000, Nphases=4):
        """
        RV detection limits computation (data-driven).
        
        Arguments (optional)
        ---------
        rangeP (list of floats): Range of orbital periods [days] within which to compute the detection limits
        rangeK (list of floats): Range of RV semi-amplitudes [m/s]
        inc_inject (float): Orbital inclination [deg] of the injected body
        fap_level (float): Maximum False Alarm Probability (FAP) for a signal to be detected
        Nsamples (int): Number of injected planets in the 2D space (P, K)
        Nphases (int): Number of orbital phases with which to inject a planet at (P, K). Based on this number, the orbital phase of each injection-recovery test is spread evenly in [-pi,pi[. The total number of injection-recovery tests is given by Nsamples*Nphases.
        """
        
        rvFile = {'jdb':self.x,'rv':self.y,'rv_err':self.yerr}
        Mstar = self.mstar
        output_dir = self.output_dir
        
        print(self.tag)

        version = int(0)
        output_file = self.tag+'InjectRecovTests_%d.p'%version
        while os.path.exists(output_file):
            version += 1
            output_file = self.tag+'InjectRecovTests_%d.p'%version

        self.output_file_DL = output_file
        ardf.DataDL(output_file, rvFile, Mstar, rangeP, rangeK, inc_inject, Nsamples, Nphases, fap_level)
        
        self.ARDENT_Plot_DataDL(output_file, percentage=[95,50], nbins=6)


    def ARDENT_Plot_DataDL(self, output_file=None, percentage=[50,95], nbins=6, axis_y_var='M', new=True, legend=True):
        """
        Plot the detection limits obtained from the ARDENT_DetectionLimitRV() method
        """

        if output_file is None:
            output_file = self.output_file_DL

        planets = self.planets_table.copy()

        output_dir = os.path.dirname(output_file)+'/'
        output = pd.read_pickle(output_file)
        P = output['P']
        M = output[axis_y_var]
        detect_rate = output['detect_rate']
        Nphases = output['Nphases']
        Mstar = output['Mstar']

        if axis_y_var!='M':
            ylabel = 'K [m/s]'
            keyword = 'semi-amp'
        else:
            ylabel = 'Mass [M$_{\oplus}$]'
            keyword = 'mass'

        detect_rate = detect_rate * 100.
        if Nphases < 8:
            cmap = plt.get_cmap('gnuplot', Nphases)
        else:
            cmap = plt.get_cmap('gnuplot', 8)
        
        if new:
            fig = plt.figure(figsize=(6,4))
        plt.title(self.starname)
        if planets is not None:
            planets2 = planets.copy()
            planets2 = planets2.loc[(planets2['period']>np.min(P))&(planets2['period']<np.max(P))]
            variable = np.array(planets2[keyword])
            variable[variable>1.05*np.max(M)] = np.max(M)
            plt.scatter(planets2['period'],variable,color='k',marker='^',s=40,zorder=9)
            plt.scatter(planets2['period'],planets2[keyword],color='k',marker='*',s=100,zorder=10)

        plt.scatter(P, M, c=detect_rate, s=10.0, alpha=0.4, edgecolors='black', linewidths=0.2, cmap=cmap, vmin=0, vmax=100)
        
        cmap2 = plt.get_cmap('gnuplot', 100)
        norm = mcolors.Normalize(vmin=0, vmax=100)
        sm = ScalarMappable(cmap=cmap2, norm=norm)
        for n,p in enumerate(percentage):
            subP_means, M95 = ardf.Stat_DataDL(output_file, percentage=p, nbins=nbins, axis_y_var=axis_y_var)
            plt.plot(subP_means, M95, color=sm.to_rgba(p),label=r'%.0f $\%%$'%(p),marker='o',markeredgecolor='k')

        if legend:
            plt.legend(loc={'M':2,'K':4}[axis_y_var])
        plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
        plt.xscale('log')
        plt.ylim(0, 1.05*max(M))
        cb = plt.colorbar(pad=0., ticks=[0.,25., 50., 75., 100.])
        cb.set_label(label=r'Detection rate [$\%$]', size='large')
        plt.rc('font', size=12)
        plt.xlabel('Period [d]', size='large')
        plt.ylabel(ylabel, size='large')
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        
        fig_title = output_file.replace('.p', '_'+axis_y_var+'vsP.png')
        plt.savefig(fig_title, format='png', dpi = 300)
    

    def ARDENT_DetectionLimitStab(self, NlocalCPU=1, InjectionRecoveryFile=None, param_file=None, nbins=15, integration_time=None, dt=None, Nphases=4, min_dist=3, max_dist=5, max_drift_a=0.2, GR=False, fine_grid=True, relaunch=False):
        """
        Function computing the dynamical detection limits (i.e. detection limits that include the constraint of orbital stability), starting from the data-driven detection limits.
        
        Arguments (optional)
        ---------
        NlocalCPU (int): Number of CPUs to dedicate to the computation of dynamical detection limits. Use NlocalCPU=0 to run the code on an external cluster. (default=1)
        InjectionRecoveryFile (string): Filename of the data-driven injection-recovery tests
        param_file (string): Name of the input file containing numerical integration parameters (in replacement of specifying them as arguments of this function)
        nbins (int): The number of period values with which to compute the data-driven and dynamical detection limits (default=15)
        integration_time (float): Total integration time used to compute the orbital stability [yr] (default=Pouter*500000)
        dt (foat): Integration timestep [yr] (default=Pinner/50)
        Nphases (int): Number of orbital phases per injected (P, K) at which to compute the orbital stability (default=4)
        min_dist (float): Criterion on close-encounter [Hill_radius] (default=3)
        max_dist (float): Criterion on escape [AU] (default=5)
        max_drift_a (float): Maximum relative drift allowed in semi-major axis of the planets for the system to be classified stable (default=0.2, i.e. 20%)
        GR (bool): General relativity correction included if GR=True, False otherwise. (default=False)
        fine_grid (bool): Compute a denser grid of detection limits around the periastron and apastron of each planet in the system. (default=True)
        relaunch (bool): Set to True to overwrite any old run with identical settings, False otherwise. (default=False)
        """
    
        mstar = self.mstar
        table_keplerian = self.planets_table.copy()
        
        if InjectionRecoveryFile is None:
            subP_means, M95 = ardf.Stat_DataDL(self.output_file_DL, nbins=nbins, percentage=95)
            output = pd.read_pickle(self.output_file_DL)
            inc_inject = output['inc_inject']
            
            version = int(0)
            self.output_file_STDL1 = self.tag+"AllStabilityRates_%d.dat"%version
            output_file = self.output_file_STDL1
            self.output_file_STDL2 = self.tag+"DynamicalDL_%d.dat"%version
            while os.path.exists(self.output_file_STDL1):
                version += 1
                self.output_file_STDL1 = self.tag+"AllStabilityRates_%d.dat"%version
                output_file = self.output_file_STDL1
                self.output_file_STDL2 = self.tag+"DynamicalDL_%d.dat"%version
        else:
            subP_means, M95 = ardf.Stat_DataDL(InjectionRecoveryFile, nbins=nbins, percentage=95)
            output = pd.read_pickle(InjectionRecoveryFile)
            inc_inject = output['inc_inject']
            
            split_filename = InjectionRecoveryFile.split('_')[-1]
            version = int(split_filename.split('.')[0])
            self.output_file_STDL1 = self.tag+"AllStabilityRates_%d.dat"%version
            output_file = self.output_file_STDL1
            self.output_file_STDL2 = self.tag+"DynamicalDL_%d.dat"%version
            
        D95 = pd.DataFrame({'period':subP_means,'mass':M95})
        
        if fine_grid == True: # Thinner grid to explore around existing planets
            grid_p = np.array(D95['period'])
            line = np.array(table_keplerian.index)
            N_finegrids = int(0)
            for l in line:
                p, m, e, a = table_keplerian.loc[l,['period','mass','ecc','semimajor']]
                if p<np.max(D95['period']) and p>np.min(D95['period']):
                    N_finegrids += 1

                    Pout_min = 365.25 * (a * (1+e))**(3./2.) * ((mstar+m*mE_S)/(1.+mE_S))**(-1./2.) - (p/6)
                    Pout_max = 365.25 * (a * (1+e))**(3./2.) * ((mstar+m*mE_S)/(1.+mE_S))**(-1./2.) + (p/6)
                    Pin_max = 365.25 * (a * (1-e))**(3./2.) * ((mstar+m*mE_S)/(1.+mE_S))**(-1./2.) + (p/6)
                    Pin_min = 365.25 * (a * (1-e))**(3./2.) * ((mstar+m*mE_S)/(1.+mE_S))**(-1./2.) - (p/6)
                    if Pin_max < Pout_min:
                        Pin = 10**np.linspace(np.log10(Pin_min),np.log10(Pin_max),6)
                        Pout = 10**np.linspace(np.log10(Pout_min),np.log10(Pout_max),6)
                        grid_p = np.hstack([grid_p,Pin]) ; grid_p = np.hstack([grid_p,Pout])

                    else:
                        Periods = 10**np.linspace(np.log10(Pin_min), np.log10(Pout_max), 6)
                        grid_p = np.hstack([grid_p,Periods])

            if N_finegrids > 0:
                grid_p = np.unique(np.round(np.sort(grid_p),4))
                D95_interp = interp1d(np.array(D95['period']), np.array(D95['mass']), kind='linear', bounds_error=False, fill_value=0)(grid_p)
            
                D95 = pd.DataFrame({'period':grid_p,'mass':D95_interp})
        N = len(D95['period'])
        self.D95 = D95
        
        if param_file is not None:
            param_names = np.genfromtxt(param_file, usecols=(0), dtype=None, encoding=None)
            param_values = np.genfromtxt(param_file, usecols=(1))
            for index in range(len(param_names)):
                if param_names[index] == 'T':
                    integration_time = param_values[index]
                if param_names[index] == 'dt':
                    dt = param_values[index]
                if param_names[index] == 'Nphases':
                    Nphases = int(param_values[index])
                if param_names[index] == 'min_dist':
                    min_dist = param_values[index]
                if param_names[index] == 'max_dist':
                    max_dist = param_values[index]
                if param_names[index] == 'Noutputs':
                    Noutputs = int(param_values[index])
                if param_names[index] == 'max_drift_a':
                    max_drift_a = param_values[index]
                if param_names[index] == 'GR':
                    GR = int(param_values[index])
                    
        if NlocalCPU > 0: #local CPU
            if os.path.exists(self.output_file_STDL1) and os.path.exists(self.output_file_STDL2):
                if relaunch:
                    print(' [INFO] An old processing has been found. Overwriting the output files (relaunch=True). ')
                    
                    dustbin = Parallel(n_jobs=NlocalCPU)(delayed(ardf.DynDL)(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, max_drift_a=max_drift_a, GR=GR) for shift in range(N))
                    
                else:
                    print(' [INFO] An old processing has been found, and relaunch=False. First delete or rename the output files below prior to launch a new processing, or set relaunch to True: \n %s \n %s \n '%(self.output_file_STDL1, self.output_file_STDL2))
                    
            else:
                dustbin = Parallel(n_jobs=NlocalCPU)(delayed(ardf.DynDL)(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, max_drift_a=max_drift_a, GR=GR) for shift in range(N))

        elif NlocalCPU == 0: #cluster
            ##### On the cluster, the code always overwrites potential old processings with the same name.
            shift = int(sys.argv[1])
#                    n_jobs = int(sys.argv[2])
            ardf.DynDL(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, max_drift_a=max_drift_a, GR=GR)


#        if os.path.exists(self.output_file_STDL1) and os.path.exists(self.output_file_STDL2):
#            if relaunch:
#                print(' [INFO] An old processing has been found. Overwriting the output files (relaunch=True). ')
#                if NlocalCPU == 0: #cluster
#                    shift = int(sys.argv[1])
##                    n_jobs = int(sys.argv[2])
#                    ardf.DynDL(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, Noutputs=Noutputs, NAFF_Thresh=NAFFthr, GR=GR)
#
#                elif NlocalCPU > 0: #local CPU
#                    dustbin = Parallel(n_jobs=NlocalCPU)(delayed(ardf.DynDL)(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, Noutputs=Noutputs, NAFF_Thresh=NAFFthr, GR=GR) for shift in range(N))
#
#            else:
#                print(' [INFO] An old processing has been found, and relaunch=False. First delete or rename the output files below prior to launch a new processing, or set relaunch to True: \n\n %s \n %s \n '%(self.output_file_STDL1, self.output_file_STDL2))
#
#        else:
#            if NlocalCPU == 0: #cluster
#                shift = int(sys.argv[1])
##                    n_jobs = int(sys.argv[2])
#                ardf.DynDL(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, Noutputs=Noutputs, NAFF_Thresh=NAFFthr, GR=GR)
#
#            elif NlocalCPU > 0: #local CPU
#                dustbin = Parallel(n_jobs=NlocalCPU)(delayed(ardf.DynDL)(shift, self.output_file_STDL1, self.output_file_STDL2, table_keplerian, D95, inc_inject, self.mstar, T=integration_time, dt=dt, min_dist=min_dist, max_dist=max_dist, Nphases=Nphases, Noutputs=Noutputs, NAFF_Thresh=NAFFthr, GR=GR) for shift in range(N))


    def ARDENT_Plot_StabDL(self, DataDLfile=None, DynDLfile=None):
        """
        Plot the RV detection limits, both data-driven and dynamical detection limits.
        
        Arguments (optional)
        ---------
        DataDLfile (string): filename of the data-driven detection limits file
        DynDLfile (string): filename of the dynamical detection limits file
        """
        if DynDLfile is None and DataDLfile is None:
            P = np.genfromtxt(self.output_file_STDL2, usecols=(0), skip_header=int(2))
            M_stb = np.genfromtxt(self.output_file_STDL2, usecols=(1), skip_header=int(2))
            P_dataDL = self.D95['period']
            M_dataDL = self.D95['mass']
        elif DynDLfile is not None and DataDLfile is not None:
            P = np.genfromtxt(DynDLfile, usecols=(0), skip_header=int(2))
            M_stb = np.genfromtxt(DynDLfile, usecols=(1), skip_header=int(2))
            P_dataDL = np.genfromtxt(DataDLfile, usecols=(0), skip_header=int(2))
            M_dataDL = np.genfromtxt(DataDLfile, usecols=(1), skip_header=int(2))
        else:
            print(' [ERROR] Both DataDL and DynDL files must be given, or none.')

        indexes = np.argsort(P)
        P = np.array(P)[indexes]
        M_stb = np.array(M_stb)[indexes]
        
        fig = plt.figure(figsize=(5,4))

#        plt.plot(P_dataDL, M_dataDL, color='xkcd:mahogany', alpha=0.6, lw=2, marker='o', mfc='white', ms=6, mew=1.5, zorder=2, label='RV') #color='xkcd:mahogany'darkgray goldenrod
#        plt.plot(P, M_stb, ls=':', color='xkcd:fire engine red', lw=2, marker='o', ms=4, zorder=10, label='RV + stability') #color='xkcd:fire engine red'firebrick
        plt.plot(P_dataDL, M_dataDL, ls='-', color='xkcd:mahogany', alpha=0.7, lw=1.5, marker='o', ms=8.5, zorder=2, label='RV') #color='xkcd:mahogany'darkgray goldenrod
        plt.plot(P, M_stb, ls='-', color='xkcd:fire engine red', lw=1, marker='o', ms=6, mfc='yellow', mew=1, zorder=10, label='RV + stability') #color='xkcd:fire engine red'firebrick
        plt.fill_between(x= P, y1= M_stb, facecolor= "xkcd:fire engine red", lw=0., alpha=0.1)
        plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
        plt.xscale('log')

        planets = pd.DataFrame(self.planets,columns=['period','semimajor','mean_long','mean_anomaly','ecc','periastron','inc','asc_node','semi-amp','mass'])
        planets = planets.loc[(planets['period']>np.min(P))&(planets['period']<np.max(P))]
#        planets = planets.loc[(planets['period']>2.0)&(planets['period']<600.0)]
        variable = np.array(planets['mass'])
        variable[variable>1.05*np.max(M_dataDL)] = np.max(M_dataDL)
        plt.scatter(planets['period'],variable,color='k',marker='^',s=30,zorder=20)
        plt.scatter(planets['period'],planets['mass'],color='k',marker='*',s=80,zorder=22)

        plt.rc('font', size=12)
        plt.xlabel('Period [d]', size='x-large')
        plt.ylabel(r'Mass [M$_{\oplus}$]', size='x-large')
        plt.tick_params(labelsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.ylim(0,np.max(M_dataDL)+1)
        plt.savefig(self.tag+'FinalDetectionLimits.png', format='png', dpi = 300)
        
        
    def ARDENT_TestStability(self, P_inject, m_inject, ML_inject=0., MA_inject=None, e_inject=0., w_inject=0., inc_inject=90., ascnode_inject=0., param_file=None, integration_time=None, dt=None, min_dist=3, max_dist=5, Noutputs=1000, GR=False, relaunch=False):
        """
        Test the orbital stability of a unique solution. Typically, this function is used to test the stability of a planet candidate with specific orbital parameters.
        
        Arguments
        ---------
        P_inject (float): orbital period [days] of the additional, candidate planet
        m_inject (float): mass [M_Earth] of the additional, candidate planet
        ML_inject, MA_inject, e_inject, w_inject, inc_inject, ascnode_inject (float, optional): mean longitude (default=0), mean anomaly (None), orbital eccentricity (0), argument of periastron (0), orbital inclination (90), and ascending node (0) of the candidate planet
        param_file (string, optional): name of the input initial conditions file (default=None). If this file is used, there is no need to specify the same parameters as arguments below.
        integration_time (float, optional): total integration time [years] (default=1e6 x Pouter)
        dt (float, optional): integration timestep [years] (default=Pinner / 100)
        min_dist (float, optional): minimal approach (close encounter) threshold [Hill radius] (default=3)
        max_dist (float, optional): maximal distance (escape) threshold [AU] (default=5)
        Noutputs (int, optional): number of output timesteps saved in a file (default=1000)
        GR (bool, optional): include the general relativity module if True (default=False)
        relaunch (bool, optional): Set to True to overwrite any old run with identical settings, False otherwise.
        """
    
        mstar = self.mstar
        table_keplerian = self.planets_table.copy()
        
        a_inject = (P_inject/365.25)**(2./3.) * ((mstar + m_inject*mE_S)/(1+mE_S))**(1./3.) # [AU]
        injected_planet = np.array([P_inject,a_inject,ML_inject,MA_inject,e_inject,w_inject,inc_inject,ascnode_inject,np.nan,m_inject])
        table_keplerian.loc[len(table_keplerian)] = injected_planet
        table_keplerian = table_keplerian.sort_values(by='semimajor')

        output_file = self.tag + 'TestStability_P' + str(np.round(P_inject,1)) + '_m' + str(np.round(m_inject,1)) + '.dat'
        self.output_evolution = output_file

        if os.path.exists(output_file):
            if relaunch:
                print(' [INFO] An old processing has been found. Overwriting the output file (relaunch=True). ')
                ardf.LongTermStab(output_file, table_keplerian, mstar, integration_time, dt, min_dist, max_dist, Noutputs, GR)
                        
            else:
                print(' [INFO] An old processing has been found, and relaunch=False. First rename or delete the output file below prior to launch a new simulation: \n\n ' + self.output_evolution)
                
        else:
            ardf.LongTermStab(output_file, table_keplerian, mstar, integration_time, dt, min_dist, max_dist, Noutputs, GR)
        

    def ARDENT_PlotOrbitalElements(self, P_inject, m_inject, output_file=None, Noutputs=1000):
        """
        Plot the orbital elements versus time of a planetary system including the additional planet candidate to test (semi-major axis, orbital eccentricity and argument of periastron)
        
        Arguments
        ---------
        P_inject (float): orbital period [days] of the additional, candidate planet
        m_inject (float): mass [M_Earth] of the additional, candidate planet
        output_file (string, optional): Name of the output file generated by the function 'ARDENT_TestStability'
        Noutputs (int, optional): number of output timesteps saved in a file (default=1000)
        """
    
        table_keplerian = self.planets_table.copy()
        Nbodies = len(table_keplerian) + 1 # The number of known planets + the injected one
        P = np.append(np.array(table_keplerian['period']), P_inject)
        P = np.sort(P)
        
        if output_file is None:
            time =  np.genfromtxt(self.output_evolution, skip_header=2, usecols=(0))
            Noutputs = len(time)
            a = np.zeros((Noutputs, Nbodies))
            ecc = np.zeros((Noutputs, Nbodies))
            w = np.zeros((Noutputs, Nbodies))
            for i in range(Nbodies):
                a[:,i] = np.genfromtxt(self.output_evolution, skip_header=2, usecols=(i*6+1))
                ecc[:,i] = np.genfromtxt(self.output_evolution, skip_header=2, usecols=(i*6+2))
                w[:,i] = np.genfromtxt(self.output_evolution, skip_header=2, usecols=(i*6+3))
                
        else:
            time = np.genfromtxt(output_file, skip_header=2, usecols=(0))
            Noutputs = len(time)
            a = np.zeros((Noutputs, Nbodies))
            ecc = np.zeros((Noutputs, Nbodies))
            w = np.zeros((Noutputs, Nbodies))
            for i in range(Nbodies):
                a[:,i] = np.genfromtxt(output_file, skip_header=2, usecols=(i*5+1))
                ecc[:,i] = np.genfromtxt(output_file, skip_header=2, usecols=(i*5+2))
                w[:,i] = np.genfromtxt(output_file, skip_header=2, usecols=(i*5+3))

        print(' [INFO] Preparing plot of the temporal evolution of the orbital elements. ')
        tmin = 0. - (time[-1]/50.)
        tmax = time[-1] * 51 / 50
        fig = plt.figure(figsize=(14,4))
        plt.title(self.starname + ' -- P_inject: ' + str(P_inject) + ' ; m_inject: ' + str(m_inject))
        plt.rc('font', size=12)
        colors = ['cornflowerblue', 'sandybrown', 'mediumseagreen', 'indianred', 'lightgrey', 'mediumorchid', 'palegreen', 'hotpink', 'darkturquoise']
        
        plt.subplot(1,3,1)
        for i in range(Nbodies):
            plt.plot(time, a[:,i], lw=1.8, alpha=0.8, color=colors[i], label='P=%.2f d'%P[i])
            plt.xlabel('Time [yr]', size='x-large')
            plt.ylabel(r'a [AU]', size='x-large')
            plt.xlim(tmin,tmax)
            plt.legend() #loc='upper left'

        plt.subplot(1,3,2)
        for i in range(Nbodies):
            plt.plot(time, ecc[:,i], lw=1.8, alpha=0.8, color=colors[i])
            plt.xlabel('Time [yr]', size='x-large')
            plt.ylabel(r'Ecc', size='x-large')
            plt.xlim(tmin,tmax)

        plt.subplot(1,3,3)
        for i in range(Nbodies):
            plt.plot(time, w[:,i], lw=1.8, alpha=0.8, color=colors[i])
            plt.xlabel('Time [yr]', size='x-large')
            plt.ylabel(r'$\omega$ [rad]', size='x-large')
            plt.xlim(tmin,tmax)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25)
        plt.savefig(self.tag+'LongTermEvolution.png', format='png', dpi = 300)

