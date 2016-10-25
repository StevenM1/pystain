import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import h5py
import scipy as sp
from scipy import ndimage

class StainDataset(object):
    
    xy_resolution = (40.5 / 2874)
    z_resolution = 0.3
    base_dir = os.path.join(os.environ['HOME'], 'data/post_mortem/new_data_format')
    
    crop_margin = 50
    
    
    def __init__(self, subject_id):
        
        self.subject_id = subject_id
        h5file = os.path.join(self.base_dir, str(subject_id), 'images.hdf5')
        print h5file
        self.h5file = h5py.File(h5file)
        self.data = self.h5file['data']
        self.mask = self.h5file['mask']
        
        self.data_pandas = pandas.read_pickle(os.path.join(self.base_dir, str(subject_id), 'data.pandas'))
        self.mask_pandas = pandas.read_pickle(os.path.join(self.base_dir, str(subject_id), 'masks.pandas'))
        
        self.slice_available = ~self.data_pandas.pivot_table('subject_id', 'slice', 'stain').isnull()
        
        
        self._center_of_mass = None
        
        self._xlim = None
        self._zlim = None
        
        
    def _get_index_slice(self, slice):
        return self.slice_available.index.get_loc(slice)

    
    def _get_index_stain(self, stain):
        return self.slice_available.columns.get_loc(stain)
    
    @property
    def center_of_mass(self):
        
        if self._center_of_mass is None:
            self._center_of_mass = ndimage.center_of_mass(self.mask.value.sum(-1))
            
        return self._center_of_mass
    
    def get_limits(self):
        
        _, zs, xs, _ = np.where(self.mask.value > 0)
        
        self._xlim = np.min(xs), np.max(xs)
        self._zlim = np.min(zs), np.max(zs)
        
    
    @property
    def xlim(self):
        
        if self._xlim is None:
            self.get_limits()
            
        return self._xlim
    
    
    @property
    def zlim(self):
        
        if self._zlim is None:
            self.get_limits()
            
        return self._zlim    
    
    
    def get_coronal_slice(self, slice, stain=None):
        
        if stain == None:
            
            
            return self.data[self._get_index_slice(slice), ...]
        
        else:
            return self.data[self._get_index_slice(slice), ..., self._get_index_stain(stain)]

        
    def get_axial_slice(self, slice, stain=None):
        
        if stain == None:
            return self.data[:, slice, :, ...]
        
        else:
            return self.data[:, slice, :, self._get_index_stain(stain)]

        
        
        
    def get_coronal_mask(self, slice, thr=3):
        
        mask = self.mask[self._get_index_slice(slice), :, :, :].sum(-1) >= thr
        
        if mask.sum() == 0:
            n_masks = (self.mask_pandas.slice == slice).sum()
            print 'Warning: only %d masks available for coronal slice %d' % (n_masks, slice)
        
        return mask 

    def get_axial_mask(self, slice, thr=3):

        mask = self.mask[:, slice, :, :].sum(-1)        
        if mask.sum() == 0:
            n_masks = mask.max()
            print 'Warning: only %d masks available for axial slice %d' % (n_masks, slice)

        mask = mask >= thr    

        return mask

    def plot_axial_slice(self, slice=None, stain='SMI32', outline_color='green', fwhm=0.15, plot_mask=True, thr=3, cmap=plt.cm.hot, crop=False, **kwargs):
        
        
        sigma = fwhm / 2.335
        
        sigma_x = sigma / self.xy_resolution 
        sigma_y = sigma / self.z_resolution 
        
        print sigma_x, sigma_y
        
        im = self.get_axial_slice(slice, stain)
        
        if np.isnan(im).any():
            print 'yo'
            V=im.copy()
            V[im!=im]=0
            VV=sp.ndimage.gaussian_filter(V,sigma=[sigma_y, sigma_x])

            W=0*im.copy()+1
            W[im!=im]=0
            WW=sp.ndimage.gaussian_filter(W,sigma=[sigma_y, sigma_x])
            
            im=VV/WW

        else:
            print sigma_x, sigma_y
            im = sp.ndimage.gaussian_filter(im, sigma=[sigma_y, sigma_x])
        
        
        plt.imshow(im, origin='lower', cmap=cmap, aspect=self.z_resolution/self.xy_resolution, interpolation='nearest')
        plt.axis('off')
        
        if plot_mask:
            m = self.get_axial_mask(slice, thr=thr)            
            plt.contour(m, origin='lower', colors=[outline_color], levels=[0,1])

        yticklabels = []
        for i in plt.yticks()[0]:
            
            if i > 0 and i < len(self.slice_available):
                yticklabels.append(self.slice_available.iloc[int(i)].name)
            else:
                yticklabels.append('')
                
        _ = plt.gca().set_yticklabels(yticklabels)
    
        if crop:
            plt.xlim(self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin)
        
        
    def plot_coronal_slice(self, slice=None, stain='SMI32', outline_color='green', fwhm=0.15, cmap=plt.cm.hot, plot_mask=True, crop=True, mask_out=False, thr=3, **kwargs):
        
        if slice == None:
            slice = np.abs(self.slice_available.index.values - int(self.center_of_mass[1])).argmin()
            slice = self.slice_available.iloc[slice].name

        sigma = fwhm / 2.335
        
        sigma_x = sigma / self.xy_resolution # left-right
        sigma_y = sigma / self.xy_resolution # anterior-posterior 
        
        im = self.get_coronal_slice(slice, stain)
        im = sp.ndimage.gaussian_filter(im, sigma=[sigma_x, sigma_y])


        if mask_out or plot_mask:
            m = self.get_coronal_mask(slice, thr=thr)            

            if mask_out:
                im = np.ma.masked_array(im, 
            if plot_mask:
                plt.contour(m, origin='lower', colors=[outline_color], levels=[0,1])
        
        plt.imshow(im, origin='lower', cmap=cmap, aspect=1, interpolation='nearest')
        plt.axis('off')
        
        plt.title('y = %d' % slice)
        
        if crop:
            plt.xlim(self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin)
            plt.ylim(self.zlim[0] - self.crop_margin, self.zlim[1] + self.crop_margin)


    def interpolate_stain(self, stain):
        
        slices_available = self.slice_available[self.slice_available[stain]].index.values
        slices_not_available = self.slice_available[~self.slice_available[stain]].index.values

        if len(slices_not_available) == 0:
            print "All slices available for stain %s!" % stain
        else:
            print 'Slices that are not available for stain %s:' % stain
        
        for slice in slices_not_available:
            slice_minus_one = slice - 50
            slice_plus_one = slice + 50
            if (slice_minus_one in slices_available) & (slice_plus_one in slices_available):
                print ' * slice %s' % slice + ' (can be interpolated)'        
                new_slice = 0.5 * self.get_coronal_slice(slice_minus_one, stain) + 0.5 * self.get_coronal_slice(slice_plus_one, stain)
                
                self.data[self._get_index_slice(slice), ..., self._get_index_stain(stain)] = new_slice
            else:
                print ' * slice %s' % slice + ' (can NOT be interpolated)'
                #self.data[self._get_index_slice(slice), ...] = np.nan
                


               
    
        
    def interpolate_stains(self, stains=None):

        if stains is None:
            stains = self.slice_available.columns
        
        for stain in stains:
            print " *** %s ***" % stain
            self.interpolate_stain(stain) 



