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
        
        #self.stains = stain_dataset.slice_available.columns.tolist()
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
    
    def make_feature_vector(self, save_directory_intensity, save_directory_gradient, save_directory_gradient_2D, use_cache=True):
        
        if use_cache:
            h5_file = self.stain_dataset.h5file
            X = h5_file['data_smoothed_0.3_thr_3'][:].copy()

            # get gradient matrices
            gradient_X = h5_file['data_smoothed_0.3_thr_3_gradient_image'][:].copy()
            gradient_X_2D = h5_file['data_smoothed_0.3_thr_3_gradient_image_2D'][:].copy()
        else:
            from .utils import get_gradients_3D
            # Do not use cached data, but recalculate smoothing
            X = np.zeros_like(self.stain_dataset.data)
            gradient_X = np.zeros_like(self.stain_dataset.data)
            gradient_X_2D = np.zeros_like(self.stain_dataset.data)
            
            print 'Smoothing with FWHM of %.2f mm' % self.fwhm

            sigma = self.fwhm / 2.335
            sigma_xy = sigma / self.stain_dataset.xy_resolution # 
            sigma_z = sigma / self.stain_dataset.z_resolution # 

            # All slices concat. into a single vector -> file
            for stain_idx in np.arange(X.shape[-1]):
                # Re-smooth original data
                
                print ' *  %s' % self.stain_dataset.stains[stain_idx]
                data = self.stain_dataset.data.value[..., stain_idx]
                X[..., stain_idx] = self._smooth_within_mask(data, self.mask, [sigma_z, sigma_xy, sigma_xy])
                 
                # Re-calculate gradient magnitudes
                # We are using a modified version of the gradient estimate. 
                # In our previous attempt we used the standard version but that did not take the large
                #  change of gradient at the borders into account. Therefore Steven Miletic build his own
                #  version that can exclude the outer borders.
                # The function is implemented in utils as get_gradients_3D
                d = get_gradients_3D(X[..., i], self.stain_dataset.thresholded_mask)
                dslice, dz, dx = d
                dslice /= self.stain_dataset.z_resolution
                dz /= self.stain_dataset.xy_resolution
                dx /= self.stain_dataset.xy_resolution

                gradient_X[..., i] = np.sqrt(dslice**2 + dz**2 + dx**2)

                
                # Get gradients in 2D (only in-plane of the slices, ignore cross-slice gradients)
                d = np.gradient(X[..., i])
                
                dslice, dz, dx = d
                dz /= self.stain_dataset.xy_resolution
                dx /= self.stain_dataset.xy_resolution

                gradient_X_2D[..., i] = np.sqrt(dz**2 + dx**2)

        self.feature_vector = X[self.mask]
        # Save data into partitions for cross-validation below, both intensity and gradient data (same naming convention,
        # different directories)
        lookup_dicts = [{'name': 'intensity', 'data': X, 'save_directory': save_directory_intensity},
                       {'name': 'gradient', 'data': gradient_X, 'save_directory': save_directory_gradient},
                       {'name': 'gradient_2D', 'data': gradient_X_2D, 'save_directory': save_directory_gradient_2D}]
        
        for lookup_dict in lookup_dicts:
            X = lookup_dict['data']
            save_directory = lookup_dict['save_directory']
            print('Saving %s data to cross-validation partitioned dataframes...' %lookup_dict['name'])
  
            ### Save full dataset for AIC/BIC calculationx
            tmp = pandas.DataFrame(X[self.mask.copy()], columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_All-Data-In-Mask.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))

            ### Save three cross-validation partitions:
            # CV dataset 1: train on every odd slice, test on every even slice.
            CV_idx_set1_1 = np.arange(0, X.shape[0], 2)
            CV_idx_set1_2 = np.arange(1, X.shape[0], 2)

            # Select slices from X
            mask1_1 = self.mask.copy()
            mask1_1[~CV_idx_set1_1,:,:] = False
            X_set1_1 = X[mask1_1]
            mask1_2 = self.mask.copy()
            mask1_2[~CV_idx_set1_2,:,:] = False
            X_set1_2 = X[mask1_2]

            tmp = pandas.DataFrame(X_set1_1, columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_CV_set1_1.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))
            tmp = pandas.DataFrame(X_set1_2, columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_CV_set1_2.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))

            # CV dataset 2: train on every even slice and test on every even slice
            # Skip every second slice to decorrelate train and test set
            CV_idx_set2_1 = np.arange(0, X.shape[0], 4)
            CV_idx_set2_2 = np.arange(2, X.shape[0], 4)

            # Select slices from X
            mask2_1 = self.mask.copy()
            mask2_1[~CV_idx_set2_1,:,:] = False
            X_set2_1 = X[mask2_1]
            mask2_2 = self.mask.copy()
            mask2_2[~CV_idx_set2_2,:,:] = False
            X_set2_2 = X[mask2_2]
            tmp = pandas.DataFrame(X_set2_1, columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_CV_set2_1.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))
            tmp = pandas.DataFrame(X_set2_2, columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_CV_set2_2.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))

            # CV dataset 3: train on every odd slice and test on every odd slice.
            # Skip every second slice to decorrelate train and test set
            CV_idx_set3_1 = np.arange(1, X.shape[0], 4)
            CV_idx_set3_2 = np.arange(3, X.shape[0], 4)
            # Select slices from X
            mask3_1 = self.mask.copy()
            mask3_1[~CV_idx_set3_1,:,:] = False
            X_set3_1 = X[mask3_1]
            mask3_2 = self.mask.copy()
            mask3_2[~CV_idx_set3_2,:,:] = False
            X_set3_2 = X[mask3_2]
            tmp = pandas.DataFrame(X_set3_1, columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_CV_set3_1.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))
            tmp = pandas.DataFrame(X_set3_2, columns=self.stain_dataset.stains)
            tmp.to_pickle(os.path.join(save_directory, '%s_%s_CV_set3_2.pkl' %(self.stain_dataset.subject_id, str(self.fwhm))))
            

        self.vmin = np.nanmin(self.feature_vector, 0)
        self.vmax = np.nanmax(self.feature_vector, 0)
        
    
    
    def get_dataframe(self):
        
        if self.feature_vector is not None:
            return pandas.DataFrame(self.feature_vector, columns=self.stain_dataset.stains)
        
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
            f = pandas.DataFrame(gmm.means_, columns=self.stain_dataset.stains)
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
            
            extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0])
            
            plt.imshow(cluster_probs[:, slice, xlim[0]:xlim[1], component-1], 
                       origin='upper', 
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
            
            extent = self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0]), self.get_z_coordinate(zlim[1]), self.get_z_coordinate(zlim[0])
            
            plt.imshow(cluster_probs[::-1, zlim[0]:zlim[1], slice, component-1].T, 
                       origin='upper',
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


    def plot_stain_cluster(self, orientation='coronal', component=1, n_components=None, slice=None, stain='SMI32', linewidth=3, mask_thr=0.5, cmap=plt.cm.gray,  **kwargs):
            
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

            
            if orientation == 'coronal':
                

                vmin = self.vmin[self.stain_dataset.stains.index(stain)]
                vmax = self.vmax[self.stain_dataset.stains.index(stain)]

                self.stain_dataset.plot_coronal_slice(slice=slice, cmap=cmap, fwhm=self.fwhm, stain=stain, vmin=vmin, vmax=vmax)
                
                if slice not in self.active_slices:
                    print '*** Warning *** Slice %d not used in clustering' % slice
                
                slice = self.slices.index(slice)
                
                
                xlim = self.stain_dataset.xlim[0] - self.stain_dataset.crop_margin, self.stain_dataset.xlim[1] + self.stain_dataset.crop_margin
                zlim = self.stain_dataset.zlim[0] - self.stain_dataset.crop_margin, self.stain_dataset.zlim[1] + self.stain_dataset.crop_margin

                extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_z_coordinate(zlim[1]), self.get_z_coordinate(zlim[0])            

                
                for component in np.arange(n_components):
                    cluster_mask = cluster_probs[slice, zlim[0]:zlim[1], xlim[0]:xlim[1], component]
                    cluster_mask = (cluster_mask > mask_thr)
                    cluster_mask = ndimage.binary_erosion(cluster_mask, iterations=linewidth)

                    
                    plt.contour(cluster_mask, 
                               origin='upper', 
                               extent=extent,
                               aspect=1,
                                colors=[self.cluster_palette[component]],
                                levels=[0, 2],
                                linewidth=[linewidth],
                               **kwargs)
                    
                plt.title('Coronal slice %d (%.2fmm)\n' % (self.stain_dataset.slice_available.index[slice], self.get_slice_coordinate(self.stain_dataset.slice_available.index[slice])))
                
                
                

            if orientation == 'axial':
                
                self.stain_dataset.plot_axial_slice(slice=slice, cmap=cmap, stain=stain)
                
                
                xlim = self.stain_dataset.xlim
                xlim = xlim[0]-self.stain_dataset.crop_margin, xlim[1]+self.stain_dataset.crop_margin
                
                extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0])
                
                for component in np.arange(n_components):
                    print component
                    
                    cluster_mask = cluster_probs[:, slice, xlim[0]:xlim[1], component]
                    cluster_mask = (cluster_mask > mask_thr)
                    cluster_mask = ndimage.binary_erosion(cluster_mask, iterations=linewidth)
                    
                    print cluster_mask.sum()
                    
                    plt.contour(cluster_mask, 
                               origin='upper', 
                               extent=extent,
                               aspect=1,
                                colors=[self.cluster_palette[component]],
                                levels=[0, 2],
                                linewidth=[linewidth],
                               **kwargs)                
                
                plt.title('Axial slice %d (%.2fmm)\n' % (slice, self.get_z_coordinate(slice)))
    # 
                
    #         elif orientation == 'axial':
    #             xlim = self.stain_dataset.xlim
    #             xlim = xlim[0]-self.stain_dataset.crop_margin, xlim[1]+self.stain_dataset.crop_margin
                
    #             extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0])
                
    #             plt.imshow(cluster_probs[:, slice, xlim[0]:xlim[1], component-1], 
    #                        origin='upper', 
    #                        cmap=self.cluster_cmaps[component-1], 
    #                        interpolation='nearest',
    #                        extent=extent,
    #                        aspect=1,
    #                        **kwargs)            
                
                
    #             plt.title('Axial slice %d (%.2fmm)\n' % (slice, self.get_z_coordinate(slice)))
            
    #         elif orientation == 'sagittal':
    # #             plt.imshow(cluster_probs[:, :, slice, component-1].T, origin='lower', cmap=self.cluster_cmaps[component-1], aspect=self.stain_dataset.xy_resolution/self.stain_dataset.z_resolution, interpolation='nearest', **kwargs)

    #             zlim = self.stain_dataset.zlim
    #             zlim = zlim[0]-self.stain_dataset.crop_margin, zlim[1]+self.stain_dataset.crop_margin
                
    #             extent = self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0]), self.get_z_coordinate(zlim[1]), self.get_z_coordinate(zlim[0])
                
    #             plt.imshow(cluster_probs[::-1, zlim[0]:zlim[1], slice, component-1].T, 
    #                        origin='upper',
    #                        cmap=self.cluster_cmaps[component-1], 
    #                        interpolation='nearest',
    #                        extent=extent,
    #                        aspect=1,
    #                        **kwargs)     
                
    # #             zlim = self.stain_dataset.zlim
    # #             plt.ylim(zlim[0]-self.stain_dataset.crop_margin, zlim[1]+self.stain_dataset.crop_margin, )
                
    #             plt.title('Sagittal slice %d (%.2fmm)' % (slice, self.get_x_coordinate(slice)))            
                
            plt.axis('on')
            sns.despine()
        
    
    
    def get_proportional_slice(self, q, orientation='coronal'):
        
        assert((q >= 0) and (q <= 1))
        
        if orientation == 'coronal':
            return self.get_nearest_active_slice(coordinate=self.slices[-1] - q * (self.slices[-1] - self.slices[0]))
        
        if orientation == 'sagittal':
            return int(self.stain_dataset.xlim[0] + q * (self.stain_dataset.xlim[1] - self.stain_dataset.xlim[0]))
        
        if orientation == 'axial':
            return int(self.stain_dataset.zlim[1] + q * (self.stain_dataset.zlim[0] - self.stain_dataset.zlim[1]))
        
        
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
            
        stain_idx = self.stain_dataset.stains.index(stain)
            
        self.stain_dataset.data = np.delete(self.stain_dataset.data, stain_idx, -1)
        self.feature_vector = np.delete(self.feature_vector, stain_idx, -1)
        
        self.stain_dataset.stains.pop(stain_idx)
        
        print "Dropped stain %s" % stain
        
        if hasattr(self, 'dropped_stains'):
            self.dropped_stains.append(stain)
        else:
            self.dropped_stains = [stain]
            
            
    def normalize_feature_vector(self):        
        self.feature_vector = (self.feature_vector - self.feature_vector.mean(0)) / self.feature_vector.std(0)

    
    
    def get_slice_coordinate(self, slice):        
        return (self.slices[-1] - slice) / 50 * self.stain_dataset.z_resolution
    
    
    def get_x_coordinate(self, x):
        return (x - self.stain_dataset.xlim[0]) * self.stain_dataset.xy_resolution
    
    def get_z_coordinate(self, z):
        return (self.stain_dataset.zlim[1] - z) * self.stain_dataset.xy_resolution    
    
    
    def get_nearest_active_slice(self, index=None, coordinate=None):    
        
        assert((index is None) or (coordinate is None))

        if index is not None:
            coordinate = self.slices[int(np.round(index))]
        
        return self.find_nearest(self.active_slices, coordinate)
    
    
    @staticmethod
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]

max_n_components = 6

cluster_colors = [sns.light_palette((i * 360 / max_n_components, 90, 60), 256, input="husl") for i in np.arange(max_n_components)]
cluster_cmaps = [sns.light_palette((i * 360 / max_n_components, 90, 60), 256, input="husl", as_cmap=True) for i in np.arange(max_n_components)]

def plot_clusters_coronal(slice, stain, ds, model, n_clusters=2):
    
    
    if slice is None:
        slice = np.abs(ds.slice_available.index.values - int(ds.center_of_mass[0])).argmin()
        slice = ds.slice_available.iloc[slice].name
        
    df = ds.smoothed_dataframe    
    n = 10000    
    stepsize = df.shape[0] / n
    df = df[~df[stain].isnull()]
    x = df[stain].values[::stepsize]
    
    im = ds.get_coronal_slice(slice, stain)
    im -= x.min()
    im /= x.max()
    

    
    cluster_probs = model.w * exgauss_pdf(im[..., np.newaxis], model.mu, model.nu, model.sigma)
    
    cluster_probs_norm = cluster_probs / cluster_probs.sum(-1)[..., np.newaxis]
    cluster_labels = np.argmax(cluster_probs_norm, -1)
    
    xlim = ds.xlim[0] - ds.crop_margin, ds.xlim[1] + ds.crop_margin
    zlim = ds.zlim[0] - ds.crop_margin, ds.zlim[1] + ds.crop_margin
    extent = ds.get_x_coordinate(xlim[0]), ds.get_x_coordinate(xlim[1]), ds.get_z_coordinate(zlim[1]), ds.get_z_coordinate(zlim[0])
    
    
    for i in xrange(n_clusters):
        tmp = np.ma.masked_array(cluster_probs_norm[..., i], cluster_labels != i)
        plt.imshow(tmp[zlim[0]:zlim[1], xlim[0]:xlim[1]], cmap=cluster_cmaps[i], vmin=1./n_clusters, vmax=1., extent=extent, interpolation='nearest')
    

def plot_clusters_axial(slice, stain, ds, model, n_clusters=2):
    
    
    if slice is None:
        slice = ds.center_of_mass[1]
        
    df = ds.smoothed_dataframe    
    n = 10000    
    stepsize = df.shape[0] / n
    df = df[~df[stain].isnull()]
    x = df[stain].values[::stepsize]
    
    im = ds.get_axial_slice(slice, stain)
    im -= x.min()
    im /= x.max()
    

    
    cluster_probs = model.w * exgauss_pdf(im[..., np.newaxis], model.mu, model.nu, model.sigma)
    
    cluster_probs_norm = cluster_probs / cluster_probs.sum(-1)[..., np.newaxis]
    cluster_labels = np.argmax(cluster_probs_norm, -1)
    
    xlim = ds.xlim[0] - ds.crop_margin, ds.xlim[1] + ds.crop_margin
    extent = ds.get_x_coordinate(xlim[0]), ds.get_x_coordinate(xlim[1]), ds.get_slice_coordinate(ds.slices[-1]), ds.get_slice_coordinate(ds.slices[0])
    
    
    for i in xrange(n_clusters):
        tmp = np.ma.masked_array(cluster_probs_norm[..., i], cluster_labels != i)
        plt.imshow(tmp[:, xlim[0]:xlim[1]], cmap=cluster_cmaps[i], vmin=1./n_clusters, vmax=1., extent=extent, interpolation='nearest')
    

def plot_clusters_sagittal(slice, stain, ds, model, n_clusters=2):
    
    
    if slice is None:
        slice = ds.center_of_mass[2]
        
    df = ds.smoothed_dataframe    
    n = 10000    
    stepsize = df.shape[0] / n
    df = df[~df[stain].isnull()]
    x = df[stain].values[::stepsize]
    
    im = ds.get_sagittal_slice(slice, stain)
    im -= x.min()
    im /= x.max()
    

    
    cluster_probs = model.w * exgauss_pdf(im[..., np.newaxis], model.mu, model.nu, model.sigma)
    
    cluster_probs_norm = cluster_probs / cluster_probs.sum(-1)[..., np.newaxis]
    cluster_labels = np.argmax(cluster_probs_norm, -1)
    
    ylim = ds.zlim[0] - ds.crop_margin, ds.zlim[1] + ds.crop_margin
    extent = ds.get_slice_coordinate(ds.slices[-1]), ds.get_slice_coordinate(ds.slices[0]), ds.get_z_coordinate(ylim[1]), ds.get_z_coordinate(ylim[0]),
    
    
    for i in xrange(n_clusters):
        tmp = np.ma.masked_array(cluster_probs_norm[..., i], cluster_labels != i)
        plt.imshow(tmp[::-1, ylim[0]:ylim[1]].T, origin='upper', cmap=cluster_cmaps[i], vmin=1./n_clusters, vmax=1., extent=extent, interpolation='nearest')
    


def exgauss_pdf(x, mu, sigma, nu):

    nu = 1./nu

    p1 = nu / 2. * np.exp((nu/2.)  * (2 * mu + nu * sigma**2. - 2. * x))


    p2 = sp.special.erfc((mu + nu * sigma**2 - x)/ (np.sqrt(2.) * sigma))

    return p1 * p2

def mixed_exgauss_likelihood(x, w, mu, sigma, nu):

    # Create indiviudal
    pdfs = w * exgauss_pdf(x[:, np.newaxis], mu, nu, sigma)

    ll = np.sum(np.log(np.sum(pdfs, 1)))

    if ((np.isnan(ll)) | (ll == np.inf)):
        return -np.inf


    return ll

def input_optimizer(pars, x, n_clusters):

    pars = np.array(pars)

    if np.sum(pars[:n_clusters-1]) > 1:
        return np.inf

    pars = np.insert(pars, n_clusters-1, 1 - np.sum(pars[:n_clusters-1]))

    if np.any(pars[:n_clusters] < 0.05):
        return np.inf

    w = pars[:n_clusters][np.newaxis, :]
    mu = pars[n_clusters:n_clusters*2][np.newaxis, :]
    nu = pars[n_clusters*2:n_clusters*3][np.newaxis, :]
    sigma = pars[n_clusters*3:n_clusters*4][np.newaxis, :]

    return -mixed_exgauss_likelihood(x, w, mu, sigma, nu)


def _fit(input_args, disp=False, popsize=100, **kwargs):

    sp.random.seed()

    x, n_clusters = input_args

    weight_bounds = [(1e-3, 1)] * (n_clusters - 1)
    mu_bounds = [(-1., 2.5)] * n_clusters
    nu_bounds = [(1e-3, 2.5)] * n_clusters
    sigma_bounds = [(1e-3, 2.5)] * n_clusters

    bounds = weight_bounds + mu_bounds + nu_bounds + sigma_bounds

    result = sp.optimize.differential_evolution(input_optimizer, bounds, (x, n_clusters), polish=True, disp=disp, maxiter=500, popsize=popsize, **kwargs)
    result = sp.optimize.minimize(input_optimizer, result.x, (x, n_clusters), method='SLSQP', bounds=bounds, **kwargs)

    return result

class SimpleExgaussMixture(object):


    def __init__(self, data, n_clusters):

        self.data = data
        self.n_clusters = n_clusters
        self.n_parameters = n_clusters * 4 - 1
        self.likelihood = -np.inf

        self.previous_likelihoods = []
        self.previous_pars = []


    def get_likelihood_data(self, data):
        
        return mixed_exgauss_likelihood(data, self.w, self.mu, self.sigma, self.nu)
    
    def get_bic_data(self, data):
        likelihood = self.get_likelihood_data(data)
        return - 2 * likelihood + self.n_parameters * np.log(data.shape[0])
        
        
    def get_aic_data(self, data):
        likelihood = self.get_likelihood_data(data)
        return 2 * self.n_parameters - 2  * likelihood
    

    def _fit(self, **kwargs):
        return _fit((self.data, self.n_clusters), **kwargs)



    def fit(self, n_tries=1, **kwargs):
        for run in np.arange(n_tries):

            result = self._fit(**kwargs)
            self.previous_likelihoods.append(-result.fun)

            if -result.fun > self.likelihood:

                pars = result.x
                pars = np.insert(pars, self.n_clusters-1, 1 - np.sum(pars[:self.n_clusters-1]))

                self.w = pars[:self.n_clusters][np.newaxis, :]
                self.mu = pars[self.n_clusters:self.n_clusters*2][np.newaxis, :]
                self.nu = pars[self.n_clusters*2:self.n_clusters*3][np.newaxis, :]
                self.sigma = pars[self.n_clusters*3:self.n_clusters*4][np.newaxis, :]

                self.likelihood = -result.fun

        self.aic = 2 * self.n_parameters - 2 * self.likelihood
        self.bic = - 2 * self.likelihood + self.n_parameters * np.log(self.data.shape[0])



    def fit_multiproc(self, n_tries=4, n_proc=4, disp=False):

        pool = Pool(n_proc)

        print 'starting pool'
        results = pool.map(_fit, [(self.data, self.n_clusters)] * n_tries)
        print 'ready'

        print results



        pool.close()

        for result in results:
            self.previous_likelihoods.append(-result.fun)
            self.previous_pars.append(result.x)

            if -result.fun > self.likelihood:

                pars = result.x
                pars = np.insert(pars, self.n_clusters-1, 1 - np.sum(pars[:self.n_clusters-1]))

                self.w = pars[:self.n_clusters][np.newaxis, :]
                self.mu = pars[self.n_clusters:self.n_clusters*2][np.newaxis, :]
                self.nu = pars[self.n_clusters*2:self.n_clusters*3][np.newaxis, :]
                self.sigma = pars[self.n_clusters*3:self.n_clusters*4][np.newaxis, :]

                self.likelihood = -result.fun

        self.aic = 2 * self.n_parameters - 2 * self.likelihood
        self.bic = - 2 * self.likelihood + self.n_parameters * np.log(self.data.shape[0])

    def plot_fit(self, colorized=False):
        # Create indiviudal pds

        t = np.linspace(0, self.data.max(), 1000)
        pdfs = self.w * exgauss_pdf(t[:, np.newaxis], self.mu, self.nu, self.sigma)

        sns.distplot(self.data, kde=False, norm_hist=True, color='grey')
        
        if colorized:
            for i in xrange(self.n_clusters):
                plt.plot(t, pdfs[:, i], c=cluster_colors[i][255])
        else:            
            plt.plot(t, pdfs, c='k', alpha=0.5)

        plt.plot(t, np.sum(pdfs, 1), c='k', lw=2)
        
        plt.xlim(-.05, 1.)


def sort_model_clusters(model):
    idx = np.argsort(model.mu + model.nu)
    
    model.mu[0] = model.mu[0][idx]
    model.nu[0] = model.nu[0][idx]
    model.sigma[0] = model.sigma[0][idx]
    model.w = model.w[0][idx]
    
    return model
