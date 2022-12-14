import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.ar_model import AutoReg
import scipy.stats as st

"""
def autocorrelation(y): 
    y_mean = np.mean(y)
    y_res = y - y_mean
    sum1 = 0.0
    for i in range(1, len(y)):
        sum1 += y_res[i]*y_res[i-1]
    sum2 = sum (y_res**2)
    r = sum1/sum2 
    
    return r
"""

def autocorrelation(y):

    mod = AutoReg(y, lags=[1], old_names = False).fit()
    r = mod.params[1]

    return r


def calc_autocorrelation(residuals, step_size,
                         detrend_window):
    
    autocorr = []

    for i in range(0, len(residuals)-detrend_window, step_size):
        # Calculate correlation coefficient with lag 1
        coeff_lag1 = autocorrelation(residuals[i:i+detrend_window])
        autocorr.append(coeff_lag1)
    
    autocorr = np.array(autocorr)

    return autocorr


def calc_variance(residuals, step_size,
                  detrend_window):

    variance = []

    for i in range(0, len(residuals)-detrend_window, step_size):
        # Calculate correlation coefficient with lag 1
        var = np.var(residuals[i:i+detrend_window])
        variance.append(var)
    
    variance = np.array(variance)

    return variance

def even_number(a):
    if (len(a) % 2) == 0:
        return a
    else:
        return a[:-1]
        print ("The length of the series is an odd number")

def delta(k, N):
    if k == 0 or k == N/2:
        return 1
    else:
        return 0

def discrete_Fourier(a):
    N = len(a)
    k_list = np.arange(0, N/2+1)
    a_k = []
    for k in k_list:
        s = 0
        for j in range(N):
            s += a[j] * np.exp(2*np.pi*1j*j*k/N)
        a_k.append( (2-delta(k,N))/N * s )
    return a_k

def random_phase(a_k):

    N_k = len(a_k)
    r_k = []
    r_k.append(0)
    for k in range(1,N_k-1):
        theta = np.random.uniform(0,2*np.pi)
        r_k.append(abs(a_k[k]) * np.exp(1j*theta))
    theta = np.random.uniform(0,2*np.pi)
    r_k.append( 2**0.5 * abs(a_k[N_k-1]) * np.cos(theta) )
    return r_k

def inverse_discrete_Fourier(r_k, N):

    N_k = len(r_k)
    r = []

    for j in range(N):
        s = 0
        for k in range(N_k):
            s += r_k[k] * np.exp (-2*np.pi*1j*j*k/N)
        r.append(s.real)

    return r

################################
# Boers, N. Observation-based early-warning signals for a collapse of the Atlantic Meridional Overturning Circulation. 
# Nat. Clim. Chang. 11, 680???688 (2021). https://doi.org/10.1038/s41558-021-01097-4
###############################

def fourrier_surrogates(ts, ns):
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def kendall_tau_test(ts, ns, tau):
    tlen = ts.shape[0]
    tsf = ts - ts.mean()
    nts = fourrier_surrogates(tsf, ns)
    stat = np.zeros(ns)
    tlen = nts.shape[1]
    for i in range(ns):
        stat[i] = st.kendalltau(np.arange(tlen), nts[i])[0]
    p = 1 - st.percentileofscore(stat, tau) / 100.
    return p
