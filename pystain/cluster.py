import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy as sp
from sklearn import mixture


class StainCluster(object):
    
    
    
    def __init__(self, stain_dataset, interpolate_missing_slices=True, fwhm=0.15):
        
        self.stain_dataset = stain_dataset
        
        if interpolate_missing_slices:
            self.stain_dataset.interpolate_stains()
        
        
        self.gmm = None
        self.fwhm = fwhm
        

        
    
    def make_feature_vector(self, thr=3, use_cache=True):
        
        cache_fn = '/home/gdholla1/data/post_mortem/cached_feature_vectors/%s_%s.hdf5' % (self.stain_dataset.subject_id, self.fwhm)
        
        if use_cache and os.path.exists(cache_fn):
            h5_file = h5py.File(cache_fn)
            self.feature_vector = h5_file['feature_vector'].value
            h5_file.close()
            
        else:
        
            X = np.zeros_like(dataset.data)

            print 'Smoothing with FWHM of %.2f mm' % self.fwhm

            sigma = self.fwhm / 2.335
            sigma_xy = sigma / self.stain_dataset.xy_resolution # 
            sigma_z = sigma / self.stain_dataset.z_resolution # 

            mask = (cluster.stain_dataset.mask.value.sum(-1) > 3)

            for stain_idx in np.arange(X.shape[-1]):
                print ' *  %s' % self.stain_dataset.slice_available.columns[stain_idx]
                data = self.stain_dataset.data.value[..., stain_idx]
                X[..., stain_idx] = self._smooth_within_mask(data, mask, [sigma_z, sigma_xy, sigma_xy])


            self.feature_vector = X[mask]

            h5_file = h5py.File(cache_fn)
            h5_file.create_dataset('feature_vector', data=self.feature_vector)
            h5_file.close()
        
    
    
    def get_dataframe(self):
        
        if self.feature_vector is not None:
            return pandas.DataFrame(self.feature_vector, columns=self.stain_dataset.slice_available.columns)
        
        else:
            raise Exception("First generate feature vector")
    
    
    @staticmethod
    def _smooth_within_mask(data, mask, sigma):
        
        data[~mask] = np.nan
        
        V=data.copy()
        V[data!=data]=0
        VV=sp.ndimage.gaussian_filter(V,sigma=sigma)

        W=0*data.copy()+1
        W[data!=data]=0
        WW=sp.ndimage.gaussian_filter(W,sigma=sigma)

        data=VV/WW
        
        return data
    
    def test(self):
        print 'yo'

