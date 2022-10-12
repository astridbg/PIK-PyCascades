import numpy as np

def calc_autocorrelation(states, start_point, autocorr, 
                         detrend_window, min_point, step_size):
    
    start_point = max(start_point, min_point)

    autocorr_tmp = []
    for i in range(start_point, len(states)-detrend_window, step_size):
        print("Start point: ", start_point)
        print("End of for loop: ", len(states)-detrend_window)
        print(i)
        # Detrend the state values within the detrend window
        # for each node (should these be different bc of timescales?)
        trend = np.polyval(np.polyfit(np.arange(detrend_window),
                                      states[i : i+detrend_window], 1),
                                      np.arange(detrend_window))
        detrended = states[i : i+detrend_window] - trend

        # Calculate correlation coefficient with lag 1
        coeff_lag1 = np.corrcoef(detrended[:-1], detrended[1:])[0,1]
        autocorr_tmp.append(coeff_lag1)
     
    autocorr = np.append(autocorr, autocorr_tmp)
    last_point = i+detrend_window

    return autocorr, last_point



