# Add modules directory to path
import sys

sys.path.append('')

# global imports
import numpy as np
# private imports from sys.path
from core.coupling import linear_coupling
from core.tipping_element import cusp
from core.tipping_network import tipping_network
from earth_sys.functions_earth_system_no_enso import global_functions



import matplotlib
matplotlib.use('Agg') #otherwise use 'pdf' instead of 'Agg'
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
sns.despine()


"""
Here Saverio's net zero climate model is put in
"""
class nzmodel():
    def __init__(self, NS, a, OF, Kappa1, Kappa2, K2, K3, n, K1, 
        Eta, a1, a2, a3, a4, xref, ScF, phi, phi1, phi2, phi3, phi4, 
        epsilon, epsilon1, epsilon2, epsilon3, epsilon4, 
        tau1, tau11, tau12, tau13, tau14, mu, mu1, mu2, mu3, mu4):

        self._NS = NS
        self._a = a
        self._OF = OF
        self._Kappa1 = Kappa1
        self._Kappa2 = Kappa2
        self._K2 = K2
        self._K3 = K3
        self._n = n
        self._K1 = K1

        self._Eta = Eta
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._a4 = a4
        self._xref = xref
        self._ScF = ScF
        self._phi = phi
        self._phi1 = phi1
        self._phi2 = phi2
        self._phi3 = phi3
        self._phi4 = phi4

        self._epsilon = epsilon
        self._epsilon1 = epsilon1
        self._epsilon2 = epsilon2
        self._epsilon3 = epsilon3
        self._epsilon4 = epsilon4

        self._tau1 = tau1
        self._tau11 = tau11
        self._tau12 = tau12
        self._tau13 = tau13
        self._tau14 = tau14

        self._mu = mu
        self._mu1 = mu1
        self._mu2 = mu2
        self._mu3 = mu3
        self._mu4 = mu4


    """
    netzero function
    """

    ### add the EWS variable to end of inputs
    ### add a switch to turn off EWS
    def NetZeroFunDamage(self, t, E0, E0_ref, xin, yin, zin, D, effective_GMT, limit_closest, ind_closest, switch_unc, switch_unc_barrett, care, unc_temp, EWS_switch, EWS_slope):
        #define arrays to store timeseries
        E_arr = []
        x_arr = []
        y_arr = []
        D_arr = []
        z_arr = []

        #this is timestep 1
        E_arr.append(E0)
        x_arr.append(xin)
        y_arr.append(yin)
        D_arr.append(D)
        z_arr.append(zin)

        Tf_start = t-1
        Tf_end = t



        #knowledge about closeness of tipping point (assumed that percet knowledge is achievable)
        alpha = 1 #linearity of how knowledge is translated into action (probably it is rather 2 than 1 judging on Scott Barrett's papers)
        if EWS_switch == True:
            kappa_know = EWS_slope
        elif switch_unc == True:
            kappa_know = (effective_GMT/global_functions.unc_limits(ind_closest, unc_temp))**alpha
        else:
            kappa_know = (effective_GMT/limit_closest)**alpha #this function only takes into account the actual threshold and no uncertainty in it


        #care of the fate of tipping elements (maybe roughly translatable into perceived impact)
        care_perceived = care*0.0035*17.77 #(self._mu*yin), where yin=E_CL,0


        #Euler solving differential equation
        for tt in range(Tf_start, Tf_end):
            i = 1

            E_i = E0_ref + self._a1*(1-np.exp(-self._a2*tt))
            E_arr.append(E_i)

            x_i = x_arr[i-1] + self._Eta*(E_arr[i-1]-y_arr[i-1]) - (0 + self._a3*(1-np.exp(-self._a4*x_arr[i-1]))) - self._OF
            x_arr.append(x_i)

            if switch_unc_barrett == True: #take into account the barrett uncertainty
                y_i = y_arr[i-1] + self._K1*global_functions.unc_limits_barrett(ind_closest, unc_temp)*z_arr[i-1] - self._K2*y_arr[i-1]
            else: #do NOT take into account the barrett uncertainty
                y_i = y_arr[i-1] + self._K1*z_arr[i-1] - self._K2*y_arr[i-1]
            y_arr.append(y_i)

            D_i = self._phi1*np.log(x_arr[i-1]/self._xref) + self._epsilon1*(np.log(x_arr[i-1]/self._xref))**2
            D_arr.append(D_i)

            #z_i = z_arr[i-1] + self._tau11*(1-D_arr[i]/(1+D_arr[i]))*(D_arr[i]-self._mu1*y_arr[i-1] + kappa_know_barrett*care_perceived*global_functions.unc_limits_barrett(ind_closest, switch_unc)) # the last part computes the uncertainty following Barrett et al., 2021
            #### add in the EWS variable multiplied by the kappa_know*care_perceived 
            z_i = z_arr[i-1] + self._tau11*(1-D_arr[i]/(1+D_arr[i]))*(D_arr[i]-self._mu1*y_arr[i-1] + kappa_know*care_perceived)
            z_arr.append(z_i)

 

        E_arr = np.array(E_arr)
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        D_arr = np.array(D_arr)
        z_arr = np.array(z_arr)



        #convert CO2 to GMT
        gmt = []
        for i in range(0, len(x_arr)):
            gmt.append(global_functions.co2_gmt(x_arr[i]))
        gmt = np.array(gmt)

        
        return gmt, E_arr, x_arr, y_arr, z_arr, D_arr




"""
    def NetZeroFunDamage(self, t, E0, xin, yin, zin, D):
        #define arrays to store timeseries
        E_arr = []
        x_arr = []
        y_arr = []
        D_arr = []
        z_arr = []

        #this is timestep 1
        E_arr.append(E0)
        x_arr.append(xin)
        y_arr.append(yin)
        D_arr.append(D)
        z_arr.append(zin)

        Tf = t
        #Tf = 2
        #Tf = 50000
        #Euler solving differential equation
        for i in range(1, Tf):
            E_i = E0 + self._a1*(1-np.exp(-self._a2*i))
            E_arr.append(E_i)

            x_i = x_arr[i-1] + self._Eta*(E_arr[i-1]-y_arr[i-1]) - (0 + self._a3*(1-np.exp(-self._a4*x_arr[i-1]))) - self._OF
            x_arr.append(x_i)

            y_i = y_arr[i-1] + self._K1*z_arr[i-1] - self._K2*y_arr[i-1]
            y_arr.append(y_i)

            D_i = self._phi1*np.log(x_arr[i-1]/self._xref) + self._epsilon1*(np.log(x_arr[i-1]/self._xref))**2
            D_arr.append(D_i)

            z_i = z_arr[i-1] + self._tau11*(1-D_arr[i]/(1+D_arr[i]))*(D_arr[i]-self._mu1*y_arr[i-1])
            z_arr.append(z_i)


        E_arr = np.array(E_arr)
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        D_arr = np.array(D_arr)
        z_arr = np.array(z_arr)



        #convert CO2 to GMT
        gmt = []
        for i in range(0, len(x_arr)):
            gmt.append(global_functions.co2_gmt(x_arr[i]))
        gmt = np.array(gmt)


        
        #print(x_arr)
        #print(gmt)
        
        #Plotting
        #fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        #T_final = 1000 #Tf
        #ax0.plot(np.arange(0, T_final, 1), gmt[0:T_final], color="blue", label="Moderately incentivized clean energy")

        #ax0.set_xlabel("Time [yrs]")
        #ax0.set_ylabel("$\Delta$ GMT [°C]")


        #fig.tight_layout()
        #fig.savefig("test_figures/scenarios.png")
        #fig.savefig("test_figures/scenarios.pdf")
        #fig.clf()
        #plt.close()
        #die
        
        
        return gmt, E_arr, x_arr, y_arr, z_arr, D_arr
"""