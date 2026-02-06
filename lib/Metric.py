import math
import numpy as np
from scipy.stats import norm



def MSE(prediction_samples, labels):
    N = labels.shape[0]
    return np.sum(np.square(prediction_samples - labels)), N

def MAE(prediction_samples, labels):
    N = labels.shape[0]
    # print(np.abs(prediction_samples - labels).shape)
    return np.sum(np.abs(prediction_samples - labels)), N

def MAPE(prediction_samples, labels):
    N = labels.shape[0]
    return np.sum(np.abs((prediction_samples - labels)/labels)), N
    
def MSPE(prediction_samples, labels):
    N = labels.shape[0]
    return np.sum(np.square((prediction_samples - labels)/labels)), N

def MAFE(prediction_samples, labels):
    N = labels.shape[0]
    return np.sum(np.square((prediction_samples - labels)/prediction_samples)), N

def QLIKE(prediction_samples, labels):
    N = labels.shape[0]
    return np.sum(labels/ prediction_samples - np.log(labels/ prediction_samples) - 1), N