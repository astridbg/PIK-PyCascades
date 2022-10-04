import numpy as np
from sklearn.linear_model import LinearRegression

'''
this function is for finding the AC array up to this point
@cuspArray is the full movement of the tipping element up to this point
@currentStartingPoint has the offset calculated inside the function
'''
def findAcRunning(cuspArray:np.ndarray, windowSize, minimumPoint, regWindow, currentStartingPoint,  AcArray:np.ndarray, slopeArray:np.ndarray, meanAcArray:np.ndarray, EwsVisible:np.ndarray):
    # settings: 
    stepSize = 10          # how much the sampling window is shifted every iteration
    
    if(cuspArray.size < minimumPoint):
        return [0, 0, 0, 0]


    # so that the calculation of AC is smooth, need to step back full window and 
    # then start at the next step
            # only calculate the slope after passing the minimum point

    startingPoint = max(currentStartingPoint, minimumPoint-windowSize)
    slope = 0

    for i in range(startingPoint, cuspArray.size-windowSize, stepSize):
        currentWindow = cuspArray[i : i + windowSize]
        detrendedWindow = currentWindow - np.polyval(np.polyfit(np.arange(currentWindow.size)
            , currentWindow, 1), np.arange(currentWindow.shape[0]))

        prevWindow = detrendedWindow[:-1]
        nextWindow = detrendedWindow[1:]
        AcArray = np.append(AcArray, np.corrcoef(prevWindow, nextWindow)[0,1])

        slope = 0

        if(AcArray.size > regWindow):
            chunkAcArray = AcArray[AcArray.size - regWindow:]
            meanAcArray = np.append(meanAcArray, np.average(chunkAcArray))
            # x = np.reshape(np.arange(0, len(chunkAcArray), 1),(-1,1))
            # model = LinearRegression().fit(x, chunkAcArray)
            # slope = model.coef_
            # slopeArray = np.append(slopeArray, slope)

            runningMean = np.average(meanAcArray[-1000:])
            runningStd = np.std(meanAcArray[-1000:])
            if(np.average(meanAcArray[-10:]) > runningMean + 1.2 * runningStd):
                EwsVisible = np.append(EwsVisible, 1)
                if(np.average(meanAcArray[-10:]) > runningMean + 2 * runningStd):
                    EwsVisible = np.append(EwsVisible, 2)
            else: 
                EwsVisible = np.append(EwsVisible, 0)
    
    lastPoint = cuspArray.size - windowSize + stepSize


    return AcArray, slopeArray, lastPoint, slope, meanAcArray, EwsVisible
# function for calculating the running mean of any array given the window of which to average and the array
def findRunningMean(AcArray:np.ndarray, meanWindow:int):
    meanAcArray = np.array([])
    
    for i in range(meanWindow, np.size(AcArray)-meanWindow):
        meanAcArray = np.append(meanAcArray, np.average(AcArray[i-meanWindow:i+meanWindow]))

    return meanAcArray