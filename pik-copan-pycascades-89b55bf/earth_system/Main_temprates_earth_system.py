# Add modules directory to path
import os
import sys
import re
import json

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

# private imports from sys.path
#from evolve import evolve # astridg commented out
from evolve_sde import evolve # astridg edit

#private imports for earth system
from earth_sys.timing import timing
from earth_sys.functions_earth_system import global_functions
from earth_sys.earth import earth_system

#measure time
#start = time.time()

#############################GLOBAL SWITCHES#########################################
time_scale = True               # time scale of tipping is incorporated
plus_minus_include = False      # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
stochasticity = True            # Gaussian noise is included in the tipping element evolution  
scaled_noise = False            # noise levels for each tipping element is scaled by the respective timescale    
ews_calculate = True            # Early Warning Signals are calculated; requires stochasticity == True
                                # for the purpose of saving computational time, this switch is by default always off in this program
                                # EWS detection can be calculated in another program
#####################################################################
long_save_name = "results"
ensemble_members = np.arange(1,101) # an array with all ensemble member to produce 

#######################GLOBAL VARIABLES##############################
# drive coupling strength
coupling_strength = np.array([0, 0.25])
# drive global mean temperature (GMT) above pre-industrial
T_end = 4.                                                                                      # final GMT increase [deg] 
T_rates = [1./10000, 1./5000, 1./2500, 1./1000, 1./500, 1./300, 1./200, 1./100, 1./50, 1./10]   # GMT increase rate [deg/yr]
GMT_arrays = []
start_points = []
for T_rate in T_rates:
    increase_length = int(T_end/T_rate)
    increase_arr = np.linspace(0, T_end, increase_length)
    
    # add buffer time with GMT=0 to allow for sufficient timeseries length
    start_point = int(increase_length/2 + 2000)
    start_points.append(start_point)
    buffer_arr = np.zeros(start_point)

    # add buffer time with GMT=T_end to allow all tipping elements to tip
    stable_arr = np.ones(4000)*T_end
    GMT_arr = np.concatenate((np.concatenate((buffer_arr, increase_arr)), stable_arr))
    GMT_arrays.append(GMT_arr)

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
All tipping times are computed ion comparison to the Amazon rainforest tipping time.
"""
if time_scale == True:
    print("Compute calibration timescale")
    #function call for absolute timing and time conversion
    #time_props = timing(tau_gis, tau_thc, tau_wais, tau_amaz)
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
    noise = 0.005   # noise level (very changeable; from Laeo Crnkovic-Rubsamen: 0.01)
    n = 4           # number of investigated tipping elements
    if scaled_noise == True:
        sigma = np.diag([1./gis_time, 1./thc_time, 1./wais_time, 1./amaz_time])*noise # diagonal uncorrelated noise
    else:
        sigma = np.diag([1]*n)*noise # diagonal uncorrelated noise
else:
    sigma = None
    if ews_calculate == True:
        ews_calculate = False
        print("Early warning signals are not analysed as there is no stochasticity in the tipping elements")

# Set parameters for Early Warning Signal analysis
tip_threshold = -1./np.sqrt(3)                  # the analytical state value for element to reach tipping

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
for kk in plus_minus_links:
    print("Wais to Thc:{}".format(kk[0]))
    print("Thc to Amaz:{}".format(kk[1]))
    
    try:
        os.stat("{}/feedbacks".format(long_save_name))
    except:
        os.mkdir("{}/feedbacks".format(long_save_name))

    try:
        os.stat("{}/feedbacks/network_{}_{}".format(long_save_name, kk[0], kk[1]))
    except:
        os.mkdir("{}/feedbacks/network_{}_{}".format(long_save_name, kk[0], kk[1]))

    try:
        os.stat("{}/feedbacks/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    except:
        os.mkdir("{}/feedbacks/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    
    #save starting conditions
    np.savetxt("{}/feedbacks/network_{}_{}/{}/empirical_values.txt".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4)), sys_var)
    
    for member in ensemble_members:

        try:
            os.stat("{}/feedbacks/network_{}_{}/{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4), str(member).zfill(3) ))
        except:
            os.mkdir("{}/feedbacks/network_{}_{}/{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4), str(member).zfill(3) ))


        for strength in coupling_strength:
            print("Coupling strength: {}".format(strength))
            
            for T_ind in range(len(T_rates)):
                
                T_rate = T_rates[T_ind]
                GMT = GMT_arrays[T_ind] # the GMT array corresponding to a certain temperature rate
                start_point = start_points[T_ind] # the start point in the GMT array of the temperature increase
                duration = len(GMT)
                print("Temperature rate: {}°C/yr".format(T_rate))
                print("Length of simulation: {}yr".format(duration))
                t = 0               # Start time
                effective_GMT = GMT[t]   # Start temperature
                
                output = []

                states = np.empty((n, duration)) # the states of tipping elements
                state_tipped = [False, False, False, False] # whether or not an element is in a tipped state
                tip_t = [np.nan, np.nan, np.nan, np.nan] # the time at which an element tips
                tip_counter = [0, 0, 0, 0] # counter for finding a true tipping point

                # set up the network of the Earth System
                net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

                # initialize state
                initial_state = [-1, -1, -1, -1]
                ev = evolve(net, initial_state)
                
                # evolve
                ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                
                for elem in range(n):
                    # store states for analysis
                    states[elem, t] = ev.get_timeseries()[1][-1, elem]

                # saving structure
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

                for t in range(1, duration):
                    
                    # increase GMT by set rate
                    effective_GMT = GMT[t]

                    # get back the network of the Earth system
                    net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

                    # initialize state
                    initial_state = [ev.get_timeseries()[1][-1, 0], ev.get_timeseries()[1][-1, 1], ev.get_timeseries()[1][-1, 2], ev.get_timeseries()[1][-1, 3]]
                    ev = evolve(net, initial_state)
                    
                    # evolve
                    ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                    
                    for elem in range(n):

                        states[elem, t] = ev.get_timeseries()[1][-1, elem]
                        
                        # check if the element is in a tipped state by checking if it is above the tipping threshold for a consequetive 10 years
                        if state_tipped[elem] == False:
                            if states[elem, t] >= tip_threshold:
                                state_tipped[elem] = True
                                tip_t[elem] = t
                                tip_counter[elem] = 1
                        else:
                            if tip_counter[elem] < 10:
                                if states[elem,t] < tip_threshold:
                                    state_tipped[elem] = False
                                    tip_counter[elem] = 0
                                else:
                                    tip_counter[elem] += 1

                        
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
                
                #necessary for break condition
                if len(output) != 0:
                    #saving structure
                    data = np.array(output)
                    # save output to keep the whole timeseries
                    np.savetxt("{}/feedbacks/network_{}_{}/{}/{}/feedbacks_Tend{}_Trate{}_d{:.2f}_n{}.txt".format(long_save_name, 
                                    kk[0], kk[1], str(mc_dir).zfill(4), str(member).zfill(3), T_end, T_rate, strength, noise), data)
                    # save only the states of the tipping element up until the tipping
                    state_data = [list(states[0,:tip_t[0]]), list(states[1,:tip_t[1]]),  list(states[2,:tip_t[2]]), list(states[3,:tip_t[3]])]
                    with open("{}/feedbacks/network_{}_{}/{}/{}/states_tstart{}_Tend{}_Trate{}_d{:.2f}_n{}.json".format(long_save_name,
                                    kk[0], kk[1], str(mc_dir).zfill(4), str(member).zfill(3), start_point, T_end, T_rate, strength, noise), "w") as f:
                        f.write(json.dumps(state_data)) 

                    # Plotting structure
                    time = data.T[0]
                    colors = ['c','b','k','g']
                    labels = ['GIS', 'THC', 'WAIS', 'AMAZ']

                    fig = plt.figure()
                    plt.grid(True)
                    plt.title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                    for elem in range(n):
                        plt.plot(time, data.T[elem+1], label=labels[elem], color=colors[elem])
                    plt.xlabel("Time [yr]")
                    plt.ylabel("System feature f [a.u.]")
                    plt.legend(loc='best')
                    ax2 = plt.gca().twinx()
                    ax2.plot(time, GMT, color='r')
                    ax2.grid(False)
                    ax2.set_ylabel("$\Delta$GMT")
                    ax2.yaxis.label.set_color('r')
                    plt.tight_layout()
                    fig.savefig("{}/feedbacks/network_{}_{}/{}/{}/feedbacks_Tend{}_Trate{}_d{:.2f}_n{}.png".format(long_save_name,
                                    kk[0], kk[1], str(mc_dir).zfill(4), str(member).zfill(3), T_end, T_rate, strength, noise))
                    plt.clf()
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

