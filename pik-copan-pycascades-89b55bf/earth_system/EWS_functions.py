import numpy as np
from scipy.ndimage import gaussian_filter1d


def autocorrelation(y): 
    
    y_mean = np.mean(y)
    y_res = y - y_mean
    sum1 = 0.0
    for i in range(1, len(y)):
        sum1 += y_res[i]*y_res[i-1]
    sum2 = sum (y_res**2)
    r = sum1/sum2 
    
    return r

def calc_autocorrelation_linear(states, start_point, autocorr,
                         detrend_window, min_point, step_size):

    start_point = max(start_point, min_point)
    autocorr_tmp = []

    for i in range(start_point, len(states)-detrend_window, step_size):
        # Detrend the state values within the detrend window
        # for each node (should these be different bc of timescales?)
        trend = np.polyval(np.polyfit(np.arange(detrend_window),
                                      states[i : i+detrend_window], 1),
                                      np.arange(detrend_window))
        detrended = states[i : i+detrend_window] - trend


        # Calculate correlation coefficient with lag 1
        coeff_lag1 = autocorrelation(detrended)
        autocorr_tmp = np.append(autocorr_tmp,coeff_lag1)

    autocorr = np.append(autocorr,autocorr_tmp)
    ann_mean = np.mean(autocorr_tmp)
    next_point = i+step_size

    return autocorr, next_point, ann_mean

def calc_autocorrelation(states, start_point, autocorr,
                         detrend_window, min_point, step_size, bw):

    start_point = max(start_point, min_point)
    autocorr_tmp = []

    for i in range(start_point, len(states)-detrend_window, step_size):
        filtered = gaussian_filter1d( states[i : i+detrend_window], bw )
        detrended = states[i : i+detrend_window] - filtered


        # Calculate correlation coefficient with lag 1
        coeff_lag1 = autocorrelation(detrended)
        autocorr_tmp = np.append(autocorr_tmp,coeff_lag1)

    autocorr = np.append(autocorr,autocorr_tmp)
    ann_mean = np.mean(autocorr_tmp)
    next_point = i+step_size

    return autocorr, next_point, ann_mean

def calc_variance_linear(states, start_point, variance,
                  detrend_window, min_point, step_size):

    start_point = max(start_point, min_point)
    variance_tmp = []

    for i in range(start_point, len(states)-detrend_window, step_size):
        # Detrend the state values within the detrend window
        # for each node (should these be different bc of timescales?)
        trend = np.polyval(np.polyfit(np.arange(detrend_window),
                                      states[i : i+detrend_window], 1),
                                      np.arange(detrend_window))
        detrended = states[i : i+detrend_window] - trend


        # Calculate correlation coefficient with lag 1
        var = np.var(detrended)
        variance_tmp = np.append(variance_tmp,var)

    variance = np.append(variance,variance_tmp)
    ann_mean = np.mean(variance_tmp)
    next_point = i+step_size

    return variance, next_point, ann_mean

def calc_variance(states, start_point, variance,
                  detrend_window, min_point, step_size, bw):

    start_point = max(start_point, min_point)
    variance_tmp = []

    for i in range(start_point, len(states)-detrend_window, step_size):
        # Detrend the state values within the detrend window
        # for each node (should these be different bc of timescales?)
        filtered = gaussian_filter1d(states[i : i+detrend_window], bw)
        detrended = states[i : i+detrend_window] - filtered


        # Calculate correlation coefficient with lag 1
        var = np.var(detrended)
        variance_tmp = np.append(variance_tmp,var)

    variance = np.append(variance,variance_tmp)
    ann_mean = np.mean(variance_tmp)
    next_point = i+step_size

    return variance, next_point, ann_mean

"""
def calc_autocorrelation_as_laeos(states, start_point, autocorr,
                         detrend_window, min_point, step_size):

    start_point = max(start_point, min_point)
    autocorr_tmp = []

    for i in range(start_point, len(states)-detrend_window, step_size):
        #print("Start point: ", start_point)
        #print("End of for loop: ", len(states)-detrend_window)
        #print(i)
        # Detrend the state values within the detrend window
        # for each node (should these be different bc of timescales?)
        trend = np.polyval(np.polyfit(np.arange(detrend_window),
                                      states[i : i+detrend_window], 1),
                                      np.arange(detrend_window))
        detrended = states[i : i+detrend_window] - trend


        # Calculate correlation coefficient with lag 1
        coeff_lag1 = np.corrcoef(detrended[:-1], detrended[1:])[0,1]
        autocorr_tmp = np.append(autocorr_tmp,coeff_lag1)

    autocorr = np.append(autocorr,autocorr_tmp)
    ann_mean = np.mean(autocorr_tmp)
    next_point = i+step_size

    return autocorr, next_point, ann_mean

"""
