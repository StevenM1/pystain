import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy as sp
from sklearn import mixture
import os


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
        

    @property
    def center_of_mass(self):
        if self._com is None:
            self._com = ndimage.center_of_mass(self.mask)
        return self._com
    
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
                print ' *  %s' % self.stain_dataset.slice_available.columns[stain_idx]
                data = self.stain_dataset.data.value[..., stain_idx]
                X[..., stain_idx] = self._smooth_within_mask(data, self.mask, [sigma_z, sigma_xy, sigma_xy])


            self.feature_vector = X[self.mask]
            
            os.remove(cache_fn)
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
    
    def plot_cluster_means(self):
        for gmm in cluster.gmms:
            f = pandas.DataFrame(gmm.means_, columns=dataset.slice_available.columns)
            f['cluster'] = np.arange(gmm.n_components) + 1
            f = pandas.melt(f, id_vars=['cluster'])

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
        
        if slice is None:
            if orientation == 'coronal':
                slice = self.center_of_mass[0]
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
            plt.imshow(cluster_probs[slice, :, :, component-1], origin='lower', cmap=self.cluster_cmaps[component-1], interpolation='nearest', **kwargs)
            xlim, ylim = self.stain_dataset.xlim, self.stain_dataset.zlim
            
            plt.xlim(xlim[0]-self.stain_dataset.crop_margin, xlim[1]+self.stain_dataset.crop_margin, )
            plt.ylim(ylim[0]-self.stain_dataset.crop_margin, ylim[1]+self.stain_dataset.crop_margin, )
            
#             plt.contour(self.mask[slice, :, :], levels=[0, 1], colors='k', linewidths=[0.5], alpha=0.5)
            
            plt.title('Coronal slice %d, component %d\n(w=%.2f)' % (self.stain_dataset.slice_available.index[slice], component, gmm.weights_[component-1]))
            
            plt.plot([xlim[0] + 25, xlim[0]+25 + 1. / self.stain_dataset.xy_resolution], [ylim[0] + 100, ylim[0] + 100], c='k')
            plt.text(xlim[0] + .5 / self.stain_dataset.xy_resolution, ylim[0] + 125, '1 cm')
            
        elif orientation == 'axial':
            plt.imshow(cluster_probs[:, slice, :, component-1], origin='lower', cmap=self.cluster_cmaps[component-1], aspect=self.stain_dataset.z_resolution/self.stain_dataset.xy_resolution, interpolation='nearest', **kwargs)
            
            xlim = self.stain_dataset.xlim
            plt.xlim(xlim[0]-self.stain_dataset.crop_margin, xlim[1]+self.stain_dataset.crop_margin)
            
            plt.title('Axial slice %d, component %d' % (slice, component))
        
        elif orientation == 'sagittal':
            plt.imshow(cluster_probs[:, :, slice, component-1].T, origin='lower', cmap=self.cluster_cmaps[component-1], aspect=self.stain_dataset.xy_resolution/self.stain_dataset.z_resolution, interpolation='nearest', **kwargs)
            
            zlim = self.stain_dataset.zlim
            plt.ylim(zlim[0]-self.stain_dataset.crop_margin, zlim[1]+self.stain_dataset.crop_margin, )
            
            plt.title('Sagittal slice %d, component %d' % (slice, component))            
            
        plt.axis('off')
        
    
    
    def get_proportional_slice(self, q, orientation='coronal'):
        
        assert((q >= 0) and (q <= 1))
        
        if orientation == 'coronal':
            return int(q * (cluster.stain_dataset.data.shape[0] - 1))
        
        if orientation == 'sagittal':
            return int(self.stain_dataset.xlim[0] + q * (self.stain_dataset.xlim[1] - self.stain_dataset.xlim[0]))
        
        if orientation == 'axial':
            return int(self.stain_dataset.zlim[0] + q * (self.stain_dataset.zlim[1] - self.stain_dataset.zlim[0]))
        
        
                
        
