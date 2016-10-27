import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy as sp
from sklearn import mixture
import os
import seaborn as sns
import pandas
import scipy as sp
from scipy import ndimage


class StainCluster(object):
    
    
    
    def __init__(self, stain_dataset, interpolate_missing_slices=True, fwhm=0.15):
        
        self.stain_dataset = stain_dataset
        
        if interpolate_missing_slices:
            self.stain_dataset.interpolate_stains()
        
        
        self.gmm = None
        self.fwhm = fwhm
        self.thr = 3
        
        self.mask = (self.stain_dataset.mask.value.sum(-1) > self.thr)
        self._com = None
        
        self.stains = stain_dataset.slice_available.columns.tolist()
        self.slices = stain_dataset.slice_available.index.tolist()
        
        
        self._active_slices = None
        

    @property
    def center_of_mass(self):
        if self._com is None:
            self._com = ndimage.center_of_mass(self.mask)
        return self._com
    
    @property
    def active_slices(self):
        if self._active_slices is None:            
            indices = self.mask.reshape(self.mask.shape[:1] + (np.prod(self.mask.shape[1:]),)).any(1)
            self._active_slices = np.array(self.slices)[indices]
            
        return self._active_slices    
    
    def make_feature_vector(self, use_cache=True):
        
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


            for stain_idx in np.arange(X.shape[-1]):
                print ' *  %s' % self.stains[stain_idx]
                data = self.stain_dataset.data.value[..., stain_idx]
                X[..., stain_idx] = self._smooth_within_mask(data, self.mask, [sigma_z, sigma_xy, sigma_xy])


            self.feature_vector = X[self.mask]
            
            os.remove(cache_fn)
            h5_file = h5py.File(cache_fn)
            h5_file.create_dataset('feature_vector', data=self.feature_vector)
            h5_file.close()
        
    
    
    def get_dataframe(self):
        
        if self.feature_vector is not None:
            return pandas.DataFrame(self.feature_vector, columns=self.stains)
        
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
    
    
    def cluster_gmm(self, n_components=[1,2,3]):

        self.gmms = [mixture.GMM(n_components=n, covariance_type='full') for n in n_components]
        
        if np.isnan(self.feature_vector).any():
            print 'Removing %d rows with NaNs' % np.isnan(self.feature_vector).any(1).sum()
            self.feature_vector  = self.feature_vector[~np.isnan(self.feature_vector).any(1),:]
        
        for gmm in self.gmms:
            print "Clustering with %d components" % gmm.n_components
            gmm.fit(self.feature_vector)
            
        max_n_components = np.max(n_components)
        self.cluster_palette = sns.color_palette('husl', max_n_components)
        self.cluster_cmaps = [sns.light_palette((i * 360 / max_n_components, 90, 60), 256, input="husl", as_cmap=True) for i in np.arange(max_n_components)]
        self.cluster_n_components = n_components
        self.cluster_predictions = {}
    
    def plot_cluster_means(self, n_components=None):
        
        if n_components is None:
            gmms = [self.gmms[np.argmax(self.cluster_n_components)]]
        elif type(n_components) is int:
            gmms = [self.gmms[self.cluster_n_components.index(n_components)]]
        else:
            gmms = [self.gmms[self.cluster_n_components.index(n)] for n in n_components]
        
            
        for gmm in gmms:
            f = pandas.DataFrame(gmm.means_, columns=self.stains)
            f['cluster'] = ['Cluster %d (w=%.2f)' % (i +1, gmm.weights_[i]) for i in np.arange(gmm.n_components)]
            f = pandas.melt(f, var_name='stain', id_vars=['cluster'])

            sns.factorplot('stain', 'value', 'cluster', data=f, kind='bar', aspect=3, palette=self.cluster_palette)
            
            
    def get_cluster_predictions(self, n_components=None):
        
        if n_components is None:
            n_components = self.cluster_n_components[-1]
            
        if n_components not in self.cluster_predictions.keys():
        
            gmm = self.gmms[self.cluster_n_components.index(n_components)]

            cluster_map = np.zeros(self.mask.shape +(gmm.n_components,))
            cluster_map[:] = np.nan
            cluster_map[self.mask, :] = gmm.predict_proba(self.feature_vector)
            
            self.cluster_predictions[n_components] = cluster_map            
            
        return self.cluster_predictions[n_components]
            
        
    def plot_cluster_probability(self, orientation='coronal', component=1, n_components=None, slice=None, mask_thr=None, **kwargs):
        
        if component < 1:
            raise Exception('Component indices start at 1!')
        
        
        if slice is None:
            if orientation == 'coronal':
                slice = self.get_nearest_active_slice(self.center_of_mass[0])
            elif orientation == 'sagittal':
                slice = self.center_of_mass[2]
            elif orientation == 'axial':
                slice = self.center_of_mass[1]
                
                
        if n_components is None:
            n_components = np.max(self.cluster_n_components)
                
                
        cluster_probs = self.get_cluster_predictions(n_components)
        
        if mask_thr is not None:
            cluster_probs = np.ma.masked_less_equal(cluster_probs, mask_thr)
        
        gmm = self.gmms[self.cluster_n_components.index(n_components)]
        
        if orientation == 'coronal':
            
            if slice not in self.active_slices:
                print '*** Warning *** Slice %d not used in clustering' % slice
            
            slice = self.slices.index(slice)
            
            
            xlim, ylim = self.stain_dataset.xlim, self.stain_dataset.zlim            
            xlim = xlim[0]-self.stain_dataset.crop_margin, xlim[1]+self.stain_dataset.crop_margin
            ylim = ylim[0]-self.stain_dataset.crop_margin, ylim[1]+self.stain_dataset.crop_margin
            
            extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_z_coordinate(ylim[1]), self.get_z_coordinate(ylim[0])
            
            plt.imshow(cluster_probs[slice, ylim[0]:ylim[1], xlim[0]:xlim[1], component-1], 
#                        origin='lower', 
                       cmap=self.cluster_cmaps[component-1], 
                       interpolation='nearest',
                       extent=extent,
                       aspect=1,
                       **kwargs)
            
            plt.title('Coronal slice %d (%.2fmm)\n' % (self.stain_dataset.slice_available.index[slice], self.get_slice_coordinate(self.stain_dataset.slice_available.index[slice])))

            
        elif orientation == 'axial':
            xlim = self.stain_dataset.xlim
            xlim = xlim[0]-self.stain_dataset.crop_margin, xlim[1]+self.stain_dataset.crop_margin
            
            extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_slice_coordinate(self.slices[0]), self.get_slice_coordinate(self.slices[-1])
            
            plt.imshow(cluster_probs[:, slice, xlim[0]:xlim[1], component-1], 
                       origin='lower', 
                       cmap=self.cluster_cmaps[component-1], 
                       interpolation='nearest',
                       extent=extent,
                       aspect=1,
                       **kwargs)            
            
            
            plt.title('Axial slice %d (%.2fmm)\n' % (slice, self.get_z_coordinate(slice)))
        
        elif orientation == 'sagittal':
#             plt.imshow(cluster_probs[:, :, slice, component-1].T, origin='lower', cmap=self.cluster_cmaps[component-1], aspect=self.stain_dataset.xy_resolution/self.stain_dataset.z_resolution, interpolation='nearest', **kwargs)

            zlim = self.stain_dataset.zlim
            zlim = zlim[0]-self.stain_dataset.crop_margin, zlim[1]+self.stain_dataset.crop_margin
            
            extent = self.get_slice_coordinate(self.slices[0]), self.get_slice_coordinate(self.slices[-1]), self.get_z_coordinate(zlim[1]), self.get_z_coordinate(zlim[0])
            
            plt.imshow(cluster_probs[::-1, zlim[0]:zlim[1], slice, component-1].T, 
#                        origin='lower', 
                       cmap=self.cluster_cmaps[component-1], 
                       interpolation='nearest',
                       extent=extent,
                       aspect=1,
                       **kwargs)     
            
#             zlim = self.stain_dataset.zlim
#             plt.ylim(zlim[0]-self.stain_dataset.crop_margin, zlim[1]+self.stain_dataset.crop_margin, )
            
            plt.title('Sagittal slice %d (%.2fmm)' % (slice, self.get_x_coordinate(slice)))            
            
        plt.axis('on')
        sns.despine()
        
    
    
    def get_proportional_slice(self, q, orientation='coronal'):
        
        assert((q >= 0) and (q <= 1))
        
        if orientation == 'coronal':
            return self.get_nearest_active_slice(coordinate=self.slices[0] + q * (self.slices[-1] - self.slices[0]))
        
        if orientation == 'sagittal':
            return int(self.stain_dataset.xlim[0] + q * (self.stain_dataset.xlim[1] - self.stain_dataset.xlim[0]))
        
        if orientation == 'axial':
            return int(self.stain_dataset.zlim[0] + q * (self.stain_dataset.zlim[1] - self.stain_dataset.zlim[0]))
        
        
    def drop_slices(self, slices):
        """ Removes slices from the mask and the feature vector"""
        
        if not hasattr(self, 'feature_vector'):
            raise Exception("First get the feature vector")
        
        slices = [self.slices.index(slice) for slice in slices]
        
        print slices
        
        feature_vector_tmp = np.zeros_like(self.stain_dataset.data)
        feature_vector_tmp[self.mask, :] = self.feature_vector
        
        self.mask[slices, ...] = False
        
        self.feature_vector = feature_vector_tmp[self.mask, ...]
        
        self._active_slices = None
        
        
        if hasattr(self, 'dropped_slices'):
            self.dropped_slices += slices
        else:
            self.dropped_slices = slices
        
        
    def get_incomplete_slices(self):        
        nans = np.isnan(self.stain_dataset.data)
        slices = np.where(nans.reshape((nans.shape[0], np.prod(nans.shape[1:]))).any(1))[0]
        
        print 'Found %d incomplete slices (out of %d)' % (len(slices), len(self.slices))
        
        slices = [self.slices[s] for s in slices]
        return slices
        
        
    def drop_stain(self, stain):
        if not hasattr(self, 'feature_vector'):
            raise Exception("First get the feature vector")
            
        stain_idx = self.stains.index(stain)
            
        self.stain_dataset.data = np.delete(self.stain_dataset.data, stain_idx, -1)
        self.feature_vector = np.delete(self.feature_vector, stain_idx, -1)
        
        self.stains.pop(stain_idx)
        
        print "Dropped stain %s" % stain
        
        if hasattr(self, 'dropped_stains'):
            self.dropped_stains.append(stain)
        else:
            self.dropped_stains = [stain]
            
            
    def normalize_feature_vector(self):        
        self.feature_vector = (self.feature_vector - self.feature_vector.mean(0)) / self.feature_vector.std(0)

    
    
    def get_slice_coordinate(self, slice):        
        return (slice - self.slices[0]) / 50 * self.stain_dataset.z_resolution
    
    
    def get_x_coordinate(self, x):
        return (x - self.stain_dataset.xlim[0]) * self.stain_dataset.xy_resolution
    
    def get_z_coordinate(self, z):
        return -(z - self.stain_dataset.zlim[1]) * self.stain_dataset.xy_resolution    
    
    
    def get_nearest_active_slice(self, index=None, coordinate=None):    
        
        assert((index is None) or (coordinate is None))

        if index is not None:
            coordinate = self.slices[int(np.round(index))]
        
        return self.find_nearest(self.active_slices, coordinate)
    
    
    @staticmethod
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
