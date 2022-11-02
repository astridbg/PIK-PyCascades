# Add modules directory to path
import os
import sys
import re

sys.path.append('')
sys.path.append('../modules/gen')
sys.path.append('../modules/core')

# global imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import time
import glob
from PyPDF2 import PdfFileMerger
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau

# private imports from sys.path
#from evolve import evolve # astridg commented out
from evolve_sde import evolve # astridg edit
from EWS_functions import autocorrelation, calc_autocorrelation, calc_variance

#private imports for earth system
from earth_sys.timing import timing
from earth_sys.functions_earth_system import global_functions
from earth_sys.earth import earth_system

temp_path = "/p/projects/dominoes/nicowun/conceptual_tipping/uniform_distribution/overshoot_study/temp_input/timeseries_final/"


#measure time
#start = time.time()

#############################GLOBAL SWITCHES#########################################
time_scale = True               # time scale of tipping is incorporated
plus_minus_include = False      # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
stochasticity = True            # gaussian noise is included in the tipping element evolution  
scaled_noise = True             # noise levels for each tipping element is scaled by the respective timescale    
ews_calculate = True            # early warning signals are calculated; requires stochasticity == True
#####################################################################
duration = 50000.               # actual real simulation years
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
#coupling_strength = np.linspace(0.0, 1.0, 11, endpoint=True)
coupling_strength = np.array([0.0])
#drive global mean temperature (GMT) above pre-industrial
#GMT_files = np.sort(glob.glob(temp_path+"*.txt"))
#GMT_files = [temp_path+"Tlim20_Tpeak35_tconv1000.txt"]
GMT_mins = [0.0]
GMT_maxs = [3.5]
t_durs = [40000]

GMT_array = np.concatenate((np.concatenate((
                            np.linspace(GMT_mins[0], GMT_mins[0], 5000),
                            np.linspace(GMT_mins[0], GMT_maxs[0], t_durs[0]))),
                            np.linspace(GMT_maxs[0], GMT_maxs[0], 5000)))
GMTs = [GMT_array]
########################Declaration of variables from passed values#######################
sys_var = np.array(sys.argv[2:], dtype=float)
print(sys_var)

#Tipping ranges from distribution
limits_gis, limits_thc, limits_wais, limits_amaz = sys_var[0], sys_var[1], sys_var[2], sys_var[3]

#Probability fractions
# TO GIS
pf_wais_to_gis, pf_thc_to_gis = sys_var[4], sys_var[5]
# TO THC
pf_gis_to_thc, pf_wais_to_thc = sys_var[6], sys_var[7]
# TO WAIS
pf_thc_to_wais, pf_gis_to_wais = sys_var[8], sys_var[9]
# TO AMAZ
pf_thc_to_amaz = sys_var[10]

#------------------------------------------

#directories for the Monte Carlo simulation
mc_dir = int(sys_var[-1])

###################### Initialize switches ##########################

# Time scale
"""
All tipping times are computed in comparison to the Amazon rainforest tipping time. 
"""
if time_scale == True:
    print("Compute calibration timescale")
    #function call for absolute timing and time conversion
    time_props = timing()
    gis_time, thc_time, wais_time, amaz_time = time_props.timescales()
    conv_fac_gis = time_props.conversion()
else:
    #no time scales included
    gis_time = thc_time = wais_time = amaz_time = 1.0
    conv_fac_gis = 1.0

# Include uncertain "+-" links:
if plus_minus_include == True:
    plus_minus_links = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)))
else:
    plus_minus_links = [np.array([1., 1., 1.])]

# Include stochasticity in the tipping elements
if stochasticity == True:
    noise = 0.1 # noise level (can be changed; from Laeo Crnkovic-Rubsamen: 0.01)
    if scaled_noise == True:
        sigma = np.diag([1./gis_time, 1./thc_time, 1./wais_time, 1./amaz_time])*noise # diagonal uncorrelated noise
    else:
        n = 4 # number of investigated tipping elements
        sigma = np.diag([1]*n)*noise # diagonal uncorrelated noise
else:
    sigma = None
    ews_calculate = False
    print("Early warning signals are not analysed as there is no stochasticity in the tipping elements")

# Set parameters for Early Warning Signal analysis
if ews_calculate == True:
    min_point = 0                                   # minimum number of states after which to start calculating EWS
    detrend_windows = np.arange(10000, 18000, 4000) # the length of the detrending window
    step_size = 10                                  # the number of states between each EWS calculation
    bws = np.arange(200,320, 60)                    # the bandwidth for filtering timeseries
    
    # 0: GIS, 1: THC, 2: WAIS, 3: AMAZ
    tip_elem = 2                                    # the tipping element to test the sensitivity of

#######################INTEGRATION PARAMETERS########################
# Timestep to integration; it is also possible to run integration until equilibrium
timestep = 0.1
#t_end in each integration loop given in years; also possible to use equilibrate method
t_end = 1.0/conv_fac_gis # simulation length in "real" years

################################# MAIN #################################
#Create Earth System
earth_system = earth_system(gis_time, thc_time, wais_time, amaz_time,
                            limits_gis, limits_thc, limits_wais, limits_amaz,
                            pf_wais_to_gis, pf_thc_to_gis, pf_gis_to_thc,
                            pf_wais_to_thc, pf_gis_to_wais, pf_thc_to_wais, pf_thc_to_amaz)

################################# MAIN LOOP #################################
#for kk in plus_minus_links: # astridg commented out for fast checking 
for kk in plus_minus_links:
    print("Wais to Thc:{}".format(kk[0]))
    print("Thc to Amaz:{}".format(kk[1]))
    
    try:
        os.stat("{}/sensitivity".format(long_save_name))
    except:
        os.mkdir("{}/sensitivity".format(long_save_name))

    try:
        os.stat("{}/sensitivity/network_{}_{}".format(long_save_name, kk[0], kk[1]))
    except:
        os.mkdir("{}/sensitivity/network_{}_{}".format(long_save_name, kk[0], kk[1]))

    try:
        os.stat("{}/sensitivity/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    except:
        os.mkdir("{}/sensitivity/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    
    #save starting conditions
    np.savetxt("{}/sensitivity/network_{}_{}/{}/empirical_values.txt".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4)), sys_var)

    for strength in coupling_strength:
        print("Coupling strength: {}".format(strength))
        
        """
        for GMT_file in GMT_files:
            #print(GMT_file)
            parts = re.split("_|Tlim|Tpeak|tconv|.txt", GMT_file)
            T_lim  = int(parts[-6])
            T_peak = int(parts[-4])
            t_conv = int(parts[-2])

            GMT = np.loadtxt(GMT_file).T[-1]
        """
        for GMT in GMTs:
            
            T_start = GMT_mins[0]
            T_end = GMT_maxs[0]
            t_dur = t_durs[0]

            print("T_start: {}°C".format(T_start))
            print("T_end: {}°C".format(T_end))
            print("t_dur: {}yrs".format(t_dur))
            
            output = []
            states = []; 

            for t in range(0, int(duration)):
                
                effective_GMT = GMT[t]

                #get back the network of the Earth system
                net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

                # initialize state
                if t == 0:
                    initial_state = [-1, -1, -1, -1]
                else:
                    initial_state = [ev.get_timeseries()[1][-1, 0], ev.get_timeseries()[1][-1, 1], ev.get_timeseries()[1][-1, 2], ev.get_timeseries()[1][-1, 3]]
                ev = evolve(net, initial_state)
                # plotter.network(net)

                ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                
                if ews_calculate == True:
                    
                    # save all states for ews analysis (relevant if t_end > timestep)
                    # if t !=0, do not count the initial state, which is the last state of the previous year
                    if t == 0:
                        states = np.append(states, ev.get_timeseries()[1][:, tip_elem])
                    else:
                        states = np.append(states, ev.get_timeseries()[1][1:, tip_elem])
                    
                #saving structure
                output.append([t,
                               ev.get_timeseries()[1][-1, 0],
                               ev.get_timeseries()[1][-1, 1],
                               ev.get_timeseries()[1][-1, 2],
                               ev.get_timeseries()[1][-1, 3],
                               net.get_number_tipped(ev.get_timeseries()[1][-1]),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[0]].count(True),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[1]].count(True),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[2]].count(True),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[3]].count(True)
                               ])
            
            # Test sensitivity of autocorrelation and variance to detrending windows and filtering bandwiths 
            tip_t = np.abs(GMT_array-sys_var[tip_elem]).argmin()        # Check if this value makes sense from first plotting the states
            tau_autocorr = np.empty((len(detrend_windows), len(bws)))
            tau_variance = np.empty((len(detrend_windows), len(bws)))
            
            for i in range(len(detrend_windows)):
                
                detrend_window = int(detrend_windows[i])
                print("Detrending window: ", detrend_window)
                
                for j in range(len(bws)):

                    bw = int(bws[j])
                    print("Filtering bandwidth: ", bw)
                    
                    last_point = 0
                    autocorr = []; variance = []; 

                    # Calculate autocorrelation and variance
                    # If number of states != number of years, tip_t must be modified
                    # There also needs to be one specific tip_t for each tipping element

                    autocorr, last_point0, ann_AC = calc_autocorrelation(states[:tip_t], last_point, autocorr,
                                                                        detrend_window, min_point, step_size, bw)

                    variance, last_point0, ann_var = calc_variance(states[:tip_t], last_point, variance,
                                                                    detrend_window, min_point, step_size, bw)

                    # Calculate Kendall tau correlation for autocorrelation and variance

                    Tau_AC, p_value = kendalltau(autocorr, np.arange(len(autocorr)))
                    #Tau_var, p_value = kendalltau(variance, np.arange(len(variance)))
                    tau_autocorr[i, j] = Tau_AC
                    #tau_variance[i, j] = Tau_var
            

            #necessary for break condition
            if len(output) != 0:
                #saving structure
                data = np.array(output)
                if scaled_noise == True:
                    np.savetxt("{}/sensitivity/network_{}_{}/{}/feedbacks_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.txt".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), tip_elem, T_start, T_end, t_dur, strength, noise), data)
                else:
                    np.savetxt("{}/sensitivity/network_{}_{}/{}/feedbacks_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.txt".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), tip_elem, T_start, T_end, t_dur, strength, noise), data)
                time = data.T[0]
                state_gis = data.T[1]
                state_thc = data.T[2]
                state_wais = data.T[3]
                state_amaz = data.T[4]

                # Plot Kendall tau for autocorrelation
                fig, ax = plt.subplots()
                ax.set_title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                ax.set_ylabel("Detrending window")
                ax.set_xlabel("Filtering bandwidth")
                X, Y = np.meshgrid(bws, detrend_windows)
                CS = ax.contourf(X, Y, tau_autocorr)
                cbar = fig.colorbar(CS)
                cbar.set_label(r"Kendall $\tau$ at $T_{crit}$")
                fig.tight_layout()
                if scaled_noise == True:
                    fig.savefig("{}/sensitivity/network_{}_{}/{}/AC_elem{}_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), tip_elem, T_start, T_end, t_dur, strength, noise))
                else:
                    fig.savefig("{}/sensitivity/network_{}_{}/{}/AC_elem{}_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), tip_elem, T_start, T_end, t_dur, strength, noise))
                #plt.show()
                fig.clf()
                plt.close()

                # Plot Kendall tau for variance
                fig, ax = plt.subplots()
                ax.set_title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                ax.set_ylabel("Detrending window")
                ax.set_xlabel("Filtering bandwidth")
                X, Y = np.meshgrid(bws, detrend_windows)
                CS = ax.contourf(X, Y, tau_variance)
                cbar = fig.colorbar(CS)
                cbar.set_label(r"Kendall $\tau$ at $T_{crit}$")
                fig.tight_layout()
                if scaled_noise == True:
                    fig.savefig("{}/sensitivity/network_{}_{}/{}/var_elem{}_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), tip_elem, T_start, T_end, t_dur, strength, noise))
                else:
                    fig.savefig("{}/sensitivity/network_{}_{}/{}/var_elem{}_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), tip_elem, T_start, T_end, t_dur, strength, noise))
                #plt.show()
                fig.clf()
                plt.close()
                
                    
"""
    # it is necessary to limit the amount of saved files
    # --> compose one pdf file for each network setting and remove the other time-files
    current_dir = os.getcwd()
    os.chdir("{}/feedbacks/network_{}_{}/{}/".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4)))
    pdfs = np.array(np.sort(glob.glob("feedbacks_*.pdf"), axis=0))
    if len(pdfs) != 0.:
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        os.system("rm feedbacks_*.pdf")
        merger.write("feedbacks_complete.pdf")
        print("Complete PDFs merged")
    os.chdir(current_dir)
"""
print("Finish")
#end = time.time()
#print("Time elapsed until Finish: {}s".format(end - start))

