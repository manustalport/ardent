import numpy as np

import ardent as ard

output_directory = '/Users/cretignier/Documents/Python/ardent/output/'
data = np.genfromtxt('/Users/cretignier/Documents/Python/ardent/K2-312_MultiDgp_Residuals_test3.dat')
jdb, rv, rv_err = data[:,0:3].T

vec = ard.ARD_tableXY(jdb, rv, rv_err)        #init the ARDENT object time-series
vec.ARD_AddStar(mass=1.18,starname='K2-312')  #init the star 
vec.ARD_Set_output_dir(output_directory)       

#optionnal for DectionLimitRV but required for DetectionLimitStab

#vec.ARD_ImportPlanets('/Users/cretignier/Documents/Python/ardent/system_parameters.dat')

#vec.ARD_AddPlanets(p=0.7195, k=3.638, e=0.00, omega=0.0, asc_node = 0.0, mean_long=147, inc=90.0) #set the planets
vec.ARD_AddPlanets(p=873.1, k=196.1, e=0.85, omega=42.0, asc_node = 0.0, mean_long=5, inc=90.0) #set the planets
vec.ARD_PlotPlanets(new=True)

vec.ARD_DetectionLimitRV_auto(fap_level=0.01)

vec.ARD_DetectionLimitStab(
    NlocalCPU = 1, 
    Nphases = 1, 
    min_dist = 3, 
    max_dist = 5, 
    Noutputs = 20000, 
    GR=1)

vec.ARD_Plot_StabDL()


#YARARA
if False:
    import numpy as np

    import ardent as ard
    
    star = 'TOI2134'
    ins = 'HARPN'
    root = '/Users/cretignier/Documents/Yarara/'+star+'/data/s1d/'+ins
    output_directory = root+'/DETECTION_LIMIT/'

    starinfo = pd.read_pickle(root+'/STAR_INFO/Stellar_info_'+star+'.p')
    keplerian = pd.read_pickle(root+'/KEPLERIAN/table_keplerians.p')
    data = pd.read_csv(root+'/KEPLERIAN/RV_residual_YV2.csv',index_col=0)

    vec = ard.ARD_tableXY(
        data.loc[data['qc'].astype('bool'),'jdb'].values, 
        data.loc[data['qc'].astype('bool'),'rv'].values, 
        data.loc[data['qc'].astype('bool'),'rv_std'].values)

    vec.ARD_AddStar(mass=starinfo['Mstar']['YARARA'],starname=star)  
    vec.ARD_Set_output_dir(output_directory)  
        
    for line in keplerian.index:
        p, k, e, omega, mean_long = keplerian.loc[line][['p','k','e','peri','long']].astype('float')
        vec.ARD_AddPlanets(p=p, k=k, e=e, omega=omega, asc_node = 0.0, mean_long=mean_long, inc=90.0) #set the planets
    vec.ARD_PlotPlanets(new=True)

    vec.ARD_DetectionLimitRV_auto(fap_level=0.01) 

    vec.ARD_DetectionLimitStab(
        NlocalCPU = 1,
        integration_time = 1000, #years
        dt = 1/365.25,           #years
        Nphases = 1, 
        min_dist = 3, 
        max_dist = 5, 
        Noutputs = 20000, 
        GR=1)

    vec.ARD_Plot_StabDL()

