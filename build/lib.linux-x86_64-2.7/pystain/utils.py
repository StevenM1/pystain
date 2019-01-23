import numpy as np
import scipy as sp
from scipy import ndimage
import pandas as pd

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
    
    new_data[~mask] = np.nan

    return new_data

def get_gradients_3D(dat, mask):
    """ Function to get gradients of a masked array in 3 dimensions. Unfortunately loops a lot so slow """
    dat_gradients_x = np.full_like(dat, np.nan)
    dat_gradients_y = np.full_like(dat, np.nan)
    dat_gradients_z = np.full_like(dat, np.nan)
    
    n_aisles = dat.shape[0]
    n_row = dat.shape[1]
    n_col = dat.shape[2]
    
    # Loop over aisles
    for aisle in range(n_aisles):
        # Loop over rows to get row gradients within each aisle
        for row in range(n_row):
            this_row = dat[aisle,row,:]
            
            # which data in row are not nan?
            which_idx = np.arange(len(dat[aisle,row,:]))[mask[aisle,row,:]]
            if len(which_idx) < 2:  # if full row is masked, skip
                continue

            dat_gradients_x[aisle,row,which_idx] = np.gradient(this_row[mask[aisle,row,:]])
        
        # columns within aisles
        for col in range(n_col):
            this_col = dat[aisle,:,col]
            which_idx = np.arange(len(dat[aisle,:,col]))[mask[aisle,:,col]]
            if len(which_idx) < 2:  # if full column is masked, skip
                continue

            dat_gradients_y[aisle,which_idx,col] = np.gradient(this_col[mask[aisle,:,col]])

    # loop over rows&columns for aisles gradients
    for row in range(n_row):
        for col in range(n_col):
            this_aisle = dat[:,row,col]
            which_idx = np.arange(len(dat[:,row,col]))[mask[:,row,col]]
            if len(which_idx) < 2:  # if full aisle is masked, skip
                continue

            dat_gradients_z[which_idx,row,col] = np.gradient(this_aisle[mask[:,row,col]])
            
    return [dat_gradients_z, dat_gradients_y, dat_gradients_x]