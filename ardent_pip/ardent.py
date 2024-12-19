import getopt
import os
import pickle
import sys

import ardent_functions as ardf
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d

NlocalCPU = int(3) # The number of CPUs allocated to compute the dynamical detection limits, to fasten the computation.
    # If NlocalCPU is set to 0, the program will be compatible with external cluster.
    # To use the code on a cluster, import all the needed files in the execution folder. Create a virtual environement to install rebound and reboundx.
    # Use the argument --array of sbatch to call the programme for each value of P_inject

class ARD_tableXY(object):
    def __init__(self, x, y, yerr, ):
        self.x = np.array(x)       # jdb times  
        self.y = np.array(y)       # RV time-series in m/s
        self.yerr = np.array(yerr) # RV uncertainties in m/s
        self.baseline = np.round(np.max(self.x) - np.min(self.x),0)
        self.planets = []
        self.ARD_AddStar()
        self.ARD_Set_output_dir()

    def ARD_Set_output_dir(self,output_dir=None):
        if output_dir is None:
            output_dir = os.getcwd()+'/output/'
        self.output_dir = output_dir
        self.tag = output_dir+self.starname+'_'

    def ARD_Plot(self,new=True):
        if new:
            fig = plt.figure(figsize=(5,4))
        plt.errorbar(self.x,self.y,yerr=self.yerr,color='k',capsize=0,marker='o',ls='')

    def ARD_ResetPlanets(self):
        self.planets = []

    def ARD_ImportPlanets(self,filename):
        param_names = np.genfromtxt(filename, usecols=(0), dtype=None, encoding=None)
        param_values = np.genfromtxt(filename, usecols=(1))
        self.mstar = param_values[param_names=='Mstar'][0]
        conv = 180/np.pi

        nb_planet = np.sum([p.split('_')[0]=='ML' for p in param_names])
        for i in np.arange(1,1+nb_planet):
            p = np.round(param_values[param_names=='P_%.0f'%(i)][0],3)
            k = np.round(param_values[param_names=='K_%.0f'%(i)][0],3)
            e = np.round(param_values[param_names=='e_%.0f'%(i)][0],2)
            omega = np.round(param_values[param_names=='w_%.0f'%(i)][0]*conv,0)
            asc_node = np.round(param_values[param_names=='O_%.0f'%(i)][0]*conv,0)
            mean_long = np.round(param_values[param_names=='ML_%.0f'%(i)][0]*conv,0)
            mean_anomaly = np.nan
            inc = 90

            mass,semi_axis = ardf.AmpStar(self.mstar, p, k, e=e, i=inc)
            mass = np.round(mass,2)
            semi_axis = np.round(semi_axis,3)

            self.planets.append([p, k, e, omega, asc_node, mean_long, mean_anomaly, inc, mass, semi_axis])
            self.ARD_ShowPlanets()


    def ARD_AddStar(self,mass=1.00,starname='Sun'):
        self.starname = starname
        self.mstar = mass

    def ARD_AddPlanets(self, p=365.25, k=0.10, e=0.0, omega=0.0, asc_node = 0.0, inc=90.0, mean_long=0.0, mean_anomaly=np.nan):
        """
        Either a mean_long or mean_anomaly parameters are needed
        Args:
            p : period
            k : semi-amplitude
            e : semi-amplitude
            omega : periastron argument in degree
            asc_node : longitude of the ascending node in degree
            inc : inclination angle in degree
            mean_long : mean longitude in degree
            mean_anomaly : mean anomaly in degree
        """

        mass,semi_axis = ardf.AmpStar(self.mstar, p, k, e=e, i=inc)
        mass = np.round(mass,2)
        semi_axis = np.round(semi_axis,3)

        self.planets.append([p, k, e, omega, asc_node, mean_long, mean_anomaly, inc, mass, semi_axis])
        self.ARD_ShowPlanets()

    def ARD_ShowPlanets(self):
        printed_table = pd.DataFrame(self.planets,columns=['period','semi-amp','ecc','periastron','asc_node','mean_long','mean_anomaly','i', 'mass','semimajor'])
        printed_table.index = ['planet %.0f'%(i) for i in range(1,1+len(self.planets))]
        print('\n [INFO] Table updated: \n')
        print(printed_table)

    def ARD_PlotPlanets(self, new=True, savefig=True, legend=True):
        table = pd.DataFrame(self.planets,columns=['period','semi-amp','ecc','periastron','asc_node','mean_long','mean_anomaly','i', 'mass','semimajor'])
        phases = np.linspace(0,2*np.pi,1000)
        phases2 = np.linspace(0,np.pi,1000)
        
        a = np.array(table['semimajor'])
        b = np.array(table['semimajor'])*np.sqrt(1-np.array(table['ecc'])**2)
        c = np.sqrt(a**2-b**2)

        x = np.cos(phases)*a[:,np.newaxis] +c[:,np.newaxis]     
        y = np.sin(phases)*b[:,np.newaxis] 

        if new:
            plt.figure(figsize=(7,7))
        plt.axis('equal')

        plt.scatter([0],[0],color='k',marker='x',label=r'%.2f $M_{\odot}$'%(self.mstar))
        for xi,yi,wi,mi in zip(x,y,np.array(table['periastron']),np.array(table['mass'])):
            X = np.array([xi,yi])
            R = np.array([[np.cos(np.pi/180*wi),-np.sin(np.pi/180*wi)],[np.sin(np.pi/180*wi),np.cos(np.pi/180*wi)]])
            X = np.dot(X.T,R).T
            plt.plot(X[0],X[1],lw=4,color='white')
            if mi<50:
                plt.plot(X[0],X[1],label=r'%.2f $M_{\oplus}$'%(mi))
            else:
                plt.plot(X[0],X[1],label=r'%.2f $M_{Jup}$'%(mi/318))

        plt.plot(np.sin(phases),np.cos(phases),color='k',ls='-.',lw=1)
        plt.axhline(y=0,color='k',ls=':',lw=1,alpha=0.2)
        plt.axvline(x=0,color='k',ls=':',lw=1,alpha=0.2)
        if legend:
            plt.legend(loc=2)
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')

        hz_inf = np.polyval(np.array([-0.47739653,  0.08719618,  2.92658674, -2.58897268,  1.11537673, -0.06598796]),self.mstar)
        hz_sup = np.polyval(np.array([ 1.15443229, -6.7284057 , 12.93412033, -8.26643736,  2.72788611,-0.15849661]),self.mstar)
        
        hmax = [hz_sup*np.cos(phases2),hz_sup*np.sin(phases2)]
        hmin = [hz_inf*np.cos(phases2),hz_inf*np.sin(phases2)]

        hmin[1] = interp1d(hmin[0], hmin[1], kind='linear', bounds_error=False, fill_value=0)(hmax[0])

        plt.fill_between(hmax[0],hmin[1],hmax[1],alpha=0.1,color='g')
        plt.fill_between(hmax[0],-hmin[1],-hmax[1],alpha=0.1,color='g')
        if savefig:
            ax = plt.gca() ; xlim = ax.get_xlim()[1]
            plt.savefig(self.tag+'Planetary_system.png', format='png', dpi = 300)
            plt.xlim(-1.25,1.25)
            plt.ylim(-1.25,1.25)
            plt.savefig(self.tag+'Planetary_system_zoom.png', format='png', dpi = 300)
            plt.xlim(-xlim,xlim)
            plt.ylim(-xlim,xlim)

    def ARD_Plot_MapUpperMass(self, x_au=1.25, detection_limit='RV'):

        if detection_limit=='RV':        
            statistic = ardf.Stat_DataDL(self.output_file_DL, percentage=95, nbins=10, axis_y_var='M')
        else:
            P = np.genfromtxt(self.output_file_STDL2, usecols=(0), skip_header=int(2))
            M_stb = np.genfromtxt(self.output_file_STDL2, usecols=(1), skip_header=int(2))
            statistic = [P,M_stb]

        a = (statistic[0]/365.25)**(2./3.) * ((self.mstar*ardf.Mass_sun+statistic[1]*ardf.Mass_earth)/(ardf.Mass_sun+ardf.Mass_earth))**(1./3.)

        grid = np.linspace(-x_au,x_au,1000)
        Gx,Gy = np.meshgrid(grid,grid)
        R = np.ravel(np.sqrt(Gx**2+Gy**2))
        M = interp1d(a, statistic[1], kind='cubic', bounds_error=False, fill_value=np.nan)(R)
        M = np.reshape(M,(1000,1000))
        plt.pcolormesh(Gx,Gy,M,vmin=0,vmax=16,cmap='gnuplot')
        ax = plt.colorbar(pad=0)
        ax.ax.set_ylabel('95-percent Mass limit detection')

    def ARD_FinalPlot(self):
        

        fig = plt.figure(figsize=(14,7))
        #plt.title(self.starname)

        plt.subplot(1,2,1)
        self.ARD_PlotPlanets(new=False,savefig=False) ; ax = plt.gca() ; xlim = ax.get_xlim()[1]
        self.ARD_Plot_MapUpperMass(x_au=xlim,detection_limit='RV')

        plt.subplot(1,2,2)
        self.ARD_PlotPlanets(new=False,savefig=False, legend=False)
        self.ARD_Plot_MapUpperMass(x_au=xlim,detection_limit='RV+Stab')

        plt.subplots_adjust(left=0.09,right=0.95,wspace=0.20)
        plt.savefig(self.tag+'Summary_analysis.png', format='png', dpi = 300)

        fig = plt.figure(figsize=(14,7))

        zoom = 1.25
        plt.subplot(1,2,1)
        self.ARD_PlotPlanets(new=False,savefig=False,legend=False)
        self.ARD_Plot_MapUpperMass(x_au=xlim,detection_limit='RV')
        plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)

        plt.subplot(1,2,2)
        self.ARD_PlotPlanets(new=False,savefig=False, legend=False)
        self.ARD_Plot_MapUpperMass(x_au=xlim,detection_limit='RV+Stab')
        plt.xlim(-zoom,zoom) ; plt.ylim(-zoom,zoom)

        plt.subplots_adjust(left=0.09,right=0.95,wspace=0.20)
        plt.savefig(self.tag+'Summary_analysis_zoom.png', format='png', dpi = 300)


    def ARD_DetectionLimitRV_auto(self, rangeP=None, fap_level=0.01):
        
        if rangeP is None:
            rangeP = [2,int(np.round(self.baseline+5,-1))]

        
        print('\n [INFO] First iteration with low resolution')
        print(' [INFO] Grid of period between :',rangeP)
        
        rms = np.std(self.y)
        print(' [INFO] RMS of the RV vector = %.2f m/s'%(rms))

        Kmax = rms*1.00
        rangeK = [0.10,np.round(Kmax,2)]
        print(' [INFO] Grid of semi-amplitude between :',rangeK)

        #first iteration with poor sampling
        self.ARD_DetectionLimitRV(rangeP=rangeP, rangeK=rangeK, fap_level=fap_level, Nsamples=200, Nphases=4)
        file1 = self.output_file_DL

        statistic = ardf.Stat_DataDL(file1, percentage=75, nbins=6)
        K = 0.0891*statistic[1]*(statistic[0]/365.25)**(-1./3.)*(self.mstar)**(-2./3.) # K semi.amplitude

        K75 = np.mean(K)
        print(' [INFO] K75 detected around %.2f'%(K75))
        Kmin = np.round(K75-0.15*rms,2)
        Kmax = np.round(K75+0.15*rms,2)
        if Kmin<0.10:
            Kmin = 0.10
        rangeK = [Kmin,Kmax]
        print(' [INFO] Second iteration with high resolution')
        print(' [INFO] Grid of semi-amplitude between :',rangeK)
    
        #second iteration with better sampling
        self.ARD_DetectionLimitRV(rangeP=rangeP, rangeK=rangeK, fap_level=fap_level, Nsamples=500, Nphases=10)
        file2 = self.output_file_DL

        print(' [INFO] Merging of the low and high resolutions files...')
        f1 = pd.read_pickle(file1)
        f2 = pd.read_pickle(file2)
        f = {
            'P':np.hstack([f1['P'],f2['P']]),
            'K':np.hstack([f1['K'],f2['K']]),
            'M':np.hstack([f1['M'],f2['M']]),
            'detect_rate':np.hstack([f1['detect_rate'],f2['detect_rate']]),
            'Nphases':f2['Nphases'],
            'Mstar':f2['Mstar'],
            'FAP':f2['FAP'],
             }
        output_file = file2.split('MassLimits')[0]+'MassLimits_merged.p'
        pickle.dump(f,open(output_file,'wb'))
        self.output_file_DL = output_file

        plt.figure(figsize=(5,10))
        plt.subplot(2,1,1)
        self.ARD_Plot_DataDL(nbins=10, percentage=[95,75,50], axis_y_var='M', new=False) #axis_y_var='K'
        plt.subplot(2,1,2)
        self.ARD_Plot_DataDL(nbins=10, percentage=[95,75,50], axis_y_var='K', new=False) #axis_y_var='K'
        plt.subplots_adjust(left=0.13,right=0.95,hspace=0.25,top=0.97,bottom=0.07)
        plt.savefig(output_file.replace('.p','.png'), format='png', dpi = 300)


    def ARD_DetectionLimitRV(self, rangeP=[2., 600.], rangeK=[0.1, 1.3], fap_level=0.01, Nsamples=500, Nphases=4, rvFile=None):
        
        if rvFile is None:
            rvFile = {'jdb':self.x,'rv':self.y,'rv_err':self.yerr}
        Mstar = self.mstar
        output_dir = self.output_dir

        output_file = self.tag+'Data-driven_MassLimits_P%.0f_%.0f_K%.1f_%.1f_F%.0f.p'%(rangeP[0],rangeP[1],rangeK[0],rangeK[1],fap_level*100)
        self.output_file_DL = output_file

        if not os.path.exists(output_file):
            ardf.DataDL(output_file, rvFile, Mstar, rangeP, rangeK, Nsamples, Nphases, fap_level)
        else:
            print(' [INFO] An old processing has already been found! If you want to rerun it again, first Delete the .p file: \n\n %s'%(output_file))
        
        self.ARD_Plot_DataDL(output_file, percentage=[95,75,50], nbins=6)


    def ARD_Plot_DataDL(self, output_file=None, percentage=[95], nbins=6, axis_y_var='K', new=True):
        """Plot the Detection Limit obtained from the .ARD_DetectionLimitRV() method"""

        if output_file is None:
            output_file = self.output_file_DL

        planets = pd.DataFrame(self.planets,columns=['period','semi-amp','ecc','periastron','asc_node','mean_long','mean_anomaly','i', 'mass','semimajor'])

        output_dir = os.path.dirname(output_file)+'/'
        output = pd.read_pickle(output_file)
        P = output['P']
        M = output[axis_y_var]
        detect_rate = output['detect_rate']
        Nphases = output['Nphases']
        Mstar = output['Mstar']

        if axis_y_var!='M':
            ylabel = r'\large{K [m/s]}'            
            keyword = 'semi-amp'
            title = ''
        else:
            ylabel = r'\large{Mass [M$_{\oplus}$]}'
            keyword = 'mass'
            title = r'$M_{*}$ = %.2f $M_{\odot}$'%(Mstar)

        detect_rate = detect_rate * 100.
        cmap = plt.get_cmap('gnuplot', 8)
        
        if new:
            fig = plt.figure(figsize=(6,4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title(title)
        if planets is not None:
            planets2 = planets.copy()
            planets2 = planets2.loc[(planets2['period']>np.min(P))&(planets2['period']<np.max(P))]
            variable = np.array(planets2[keyword])
            variable[variable>1.05*np.max(M)] = np.max(M)
            plt.scatter(planets2['period'],variable,color='k',marker='^',s=20,zorder=9)
            plt.scatter(planets2['period'],planets2[keyword],color='k',marker='*',s=50,zorder=10)

        plt.scatter(P, M, c=detect_rate, s=10.0, alpha=0.4, edgecolors='black', linewidths=0.2, cmap=cmap, vmin=0, vmax=100)
        
        norm = mcolors.Normalize(vmin=0, vmax=100)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        for n,p in enumerate(percentage):
            subP_means, M95 = ardf.Stat_DataDL(output_file, percentage=p, nbins=nbins, axis_y_var=axis_y_var)
            plt.plot(subP_means, M95, color=sm.to_rgba(p),label='%.0f \%%'%(p),marker='o',markeredgecolor='k')

        plt.legend(loc=2)
        plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
        plt.xscale('log')
        plt.ylim(0, 1.05*max(M))
        cb = plt.colorbar(label=r'\large{Detection rate [$\%$]}', pad=0., ticks=[0.,25., 50., 75., 100.])
        plt.xlabel(r'\large{Period [d]}')
        plt.ylabel(ylabel)
        plt.tick_params(labelsize=11)
        plt.savefig(output_file.replace('.p','.png'), format='png', dpi = 300)
    

    def ARD_DetectionLimitStab(self, NlocalCPU=1, integration_time=None, dt=None, Nphases=3, min_dist=3, max_dist=5, Noutputs=20000, GR=1, relaunch=False):
        
        subP_means, M95 = ardf.Stat_DataDL(self.output_file_DL, nbins=15, percentage=95)
        D95 = pd.DataFrame({'period':subP_means,'mass':M95})

        table_keplerian = pd.DataFrame(self.planets,columns=['period','semi-amp','ecc','periastron','asc_node','mean_long','mean_anomaly','i', 'mass','semimajor'])

        grid_p = np.array(D95['period'])
        for p in np.array(table_keplerian['period']): #thinner grid to explore around existing planets
            if p<np.max(D95['period']):
                ratio = np.linspace(0.75,1.0,6)-0.05
                grid_p = np.hstack([grid_p,p*ratio])
                grid_p = np.hstack([grid_p,p/ratio])
        grid_p = np.sort(grid_p)
        D95_interp = interp1d(np.array(D95['period']), np.array(D95['mass']), kind='linear', bounds_error=False, fill_value=0)(grid_p)

        D95 = pd.DataFrame({'period':grid_p,'mass':D95_interp})
        N = len(D95['period'])

        self.D95 = D95
        self.output_file_STDL1 = self.tag+"AllStabilityRates.dat"
        self.output_file_STDL2 = self.tag+"Final_DynamicalDetectLim.dat"

        if relaunch:
            os.system('rm '+self.output_file_STDL1)
            os.system('rm '+self.output_file_STDL2)

        if os.path.exists(self.output_file_STDL1):
            print(' [INFO] An old processing has already been found! If you want to rerun it again, first Delete the .p file: \n\n %s'%(self.output_file_STDL1))
        else:        
            if NlocalCPU == 0: #cluster
                shift = int(sys.argv[1])
                ardf.DynDL(shift, table_keplerian, D95, self.tag, T=integration_time, dt=dt, Nphases=Nphases, min_dist=min_dist, max_dist=max_dist, Noutputs=Noutputs, GR=GR)
                
            elif NlocalCPU > 0: #MCp
                dustbin = Parallel(n_jobs=NlocalCPU)(delayed(ardf.DynDL)(shift, table_keplerian, D95, self.tag, T=integration_time, dt=dt, Nphases=Nphases, min_dist=min_dist, max_dist=max_dist, Noutputs=Noutputs, GR=GR) for shift in range(N))


    def ARD_Plot_StabDL(self, new=True):
        """
        Plot the detection limits, both data-driven limits and dynamical detection limits.
        
        Arguments
        ---------
        sys_name (string): Name of the system
        data_driven (string, optional): Filename of the data-driven detection limits
        stability_driven (string, optional): Filename of the dynamical detection limits
        """
        

        P = np.genfromtxt(self.output_file_STDL2, usecols=(0), skip_header=int(2))
        M_stb = np.genfromtxt(self.output_file_STDL2, usecols=(1), skip_header=int(2))

        indexes = np.argsort(P)
        P = np.array(P)[indexes]
        M_stb = np.array(M_stb)[indexes]
        
        if new:
            fig = plt.figure(figsize=(5,4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(self.D95['period'], self.D95['mass'], color='k', marker='.', lw=1.5, zorder=2, label=r'\large{w/o stability}')
        plt.plot(P, M_stb, ls='-', color='red', marker='.', lw=1.5, zorder=10, label=r'\large{w stability}')
        plt.grid(which='both', ls='--', linewidth=0.1, zorder=1)
        plt.xscale('log')

        planets = pd.DataFrame(self.planets,columns=['period','semi-amp','ecc','periastron','asc_node','mean_long','mean_anomaly','i', 'mass','semimajor'])
        planets = planets.loc[(planets['period']>np.min(P))&(planets['period']<np.max(P))]
        variable = np.array(planets['mass'])
        variable[variable>1.05*np.max(self.D95['mass'])] = np.max(self.D95['mass'])
        plt.scatter(planets['period'],variable,color='k',marker='^',s=20,zorder=9)
        plt.scatter(planets['period'],planets['mass'],color='k',marker='*',s=50,zorder=10)

        plt.xlabel(r'\Large{Period [d]}')
        plt.ylabel(r'\Large{Mass [M$_{\oplus}$]}')
        plt.tick_params(labelsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.ylim(0,np.max(self.D95['mass'])+1)
        plt.savefig(self.tag+'FinalDetectionLimits.png', format='png', dpi = 300)




