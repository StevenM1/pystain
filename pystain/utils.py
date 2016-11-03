import numpy as np
import scipy as sp
from scipy import ndimage

def smooth_within_mask(data, mask, sigma):
    
    
    new_data = data.copy()
    new_data[~mask] = np.nan

    V=new_data.copy()
    V[new_data!=new_data]=0
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

    W=0*new_data.copy()+1
    W[new_data!=new_data]=0
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma)

    new_data=VV/WW
    
    new_data[~mask] = data[~mask]

    return new_data
