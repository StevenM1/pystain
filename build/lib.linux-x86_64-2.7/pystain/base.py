import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import h5py
import scipy as sp
from scipy import ndimage
from .utils import smooth_within_mask
import pandas

class StainDataset(object):
    
    xy_resolution = (40.5 / 2874)
    z_resolution = 0.3
    base_dir = os.path.join(os.environ['HOME'], 'data/post_mortem/new_data_format')
    
    crop_margin = 50
    
    
    def __init__(self, subject_id, thr=3, fwhm=0.30):
        
        self.subject_id = subject_id
        h5file = os.path.join(self.base_dir, str(subject_id), 'images.hdf5')
        print h5file
        self.h5file = h5py.File(h5file)
        self.data = self.h5file['data']
        self.mask = self.h5file['mask']
        
        self.data_pandas = pandas.read_pickle(os.path.join(self.base_dir, str(subject_id), 'data.pandas'))
        self.mask_pandas = pandas.read_pickle(os.path.join(self.base_dir, str(subject_id), 'masks.pandas'))
        
        self.slice_available = ~self.data_pandas.pivot_table('subject_id', 'slice', 'stain').isnull()

        self.slices = self.slice_available.index.tolist()
        self.stains = self.slice_available.columns.tolist()

        self.thr = thr
        self.fwhm = fwhm
        
        self.thresholded_mask = self.mask.value.sum(-1) > thr
        
        self._smoothed_dataframe = None

        # Get smoothed data
        key = 'data_smoothed_%s_thr_%s' % (fwhm, thr)
        
        if key not in self.h5file.keys():
            print '%s not cached' % key
            
            self.interpolate_stains()
            self.smoothed_data = np.zeros_like(self.data)

            sigma = self.fwhm / 2.335
            sigma_xy = sigma / self.xy_resolution # 
            sigma_z = sigma / self.z_resolution # 
            
            for i, stain in enumerate(self.stains):
                print "Smoothing %s" % stain
                
                self.smoothed_data[..., i] = smooth_within_mask(self.data[..., i], self.thresholded_mask, [sigma_z, sigma_xy, sigma_xy])

            
                        
            
            self.h5file.create_dataset(key, data=self.smoothed_data)
            self.smoothed_data = self.h5file[key]
            self.get_vminmax()
            self.h5file.create_dataset(key + '_vmin', data=self._vmin)
            self.h5file.create_dataset(key + '_vmax', data=self._vmax)
            self.h5file.flush()

            
        else:
            self.smoothed_data = self.h5file[key]
            self._vmin = self.h5file[key + '_vmin'].value
            self._vmax = self.h5file[key + '_vmax'].value

        # mask details
        self._center_of_mass = None
        
        self._xlim = None
        self._zlim = None
        
        
    def _get_index_slice(self, slice):
        return self.slice_available.index.get_loc(slice)

    
    def _get_index_stain(self, stain):
        return self.stains.index(stain)
    
    @property
    def center_of_mass(self):
        
        if self._center_of_mass is None:
            self._center_of_mass = list(ndimage.center_of_mass(self.mask.value.sum(-1)))
            self._center_of_mass[0] = self.slices[int(np.round(self._center_of_mass[0]))]
            self._center_of_mass = tuple(self._center_of_mass)    
            
        return self._center_of_mass
    
    @property
    def smoothed_dataframe(self):
        
        if self._smoothed_dataframe is None:
            self._smoothed_dataframe = pandas.DataFrame(self.smoothed_data.value[self.thresholded_mask], columns=self.stains)
            
        return self._smoothed_dataframe

    def get_limits(self):
        
        _, zs, xs = np.where(self.thresholded_mask > 0)
        
        self._xlim = np.min(xs), np.max(xs)
        self._zlim = np.min(zs), np.max(zs)
        
    
    @property
    def xlim(self):
        
        if self._xlim is None:
            self.get_limits()
            
        return self._xlim
    
    @property 
    def vmin(self):
        if self._vmin is None:
            self.get_vminmax()
            
        return self._vmin

    @property
    def vmax(self):
        
        if self._vmax is None:
            self.get_vminmax()
            
        return self._vmax
    
    @property
    def zlim(self):
        
        if self._zlim is None:
            self.get_limits()
            
        return self._zlim    

    @property
    def gradient_image(self, gradient_type='gradient'):

        if gradient_type == 'gradient':
            key = 'data_smoothed_%s_thr_%s_gradient_image' % (self.fwhm, self.thr)
        elif gradient_type == 'gradient_2D':
            key = 'data_smoothed_%s_thr_%s_gradient_image_2D' % (self.fwhm, self.thr)
        else:
            raise(IOError('I dont get the gradient type. Either "gradient" or "gradient_2D"'))

        
        if not key in self.h5file.keys():
            self.get_gradient_images()

        return self.h5file[key]


    def get_vminmax(self, percentiles=(0, 99)):
        print 'calculating vmin'
        self._vmin = np.nanpercentile(self.smoothed_data.value[self.thresholded_mask, ...], percentiles[0], 0)

        print 'calculating vmax'
        self._vmax = np.nanpercentile(self.smoothed_data.value[self.thresholded_mask, ...], percentiles[1], 0)
   
    def get_gradient_images(self):
        from .utils import get_gradients_3D
        
        gradient_magnitude_smoothed = np.zeros_like(self.smoothed_data)
        
        for i in xrange(len(self.stains)):
            print 'Calculcating gradient of %s' % self.stains[i]
            # We are using a modified version of the gradient estimate. 
            # In our previous attempt we used the standard version but that did not take the large
            #  change of gradient at the borders into account. Therefore Steven Miletic build his own
            #  version that can exclude the outer borders.
            # The function is implemented in utils as get_gradients_3D
            d = get_gradients_3D(self.smoothed_data[..., i], self.thresholded_mask)
            dslice, dz, dx = d
            dslice /= self.z_resolution
            dz /= self.xy_resolution
            dx /= self.xy_resolution
            
            gradient_magnitude_smoothed[..., i] = np.sqrt(dslice**2 + dz**2 + dx**2)

        key = 'data_smoothed_%s_thr_%s_gradient_image' % (self.fwhm, self.thr)

        self.h5file.create_dataset(key, data=gradient_magnitude_smoothed)
        self.h5file.flush()
                
    def get_gradient_images_2D(self):
        gradient_magnitude_smoothed = np.zeros_like(self.smoothed_data)
        
        for i in xrange(len(self.stains)):
            print 'Calculcating gradient of %s' % self.stains[i]
            # Note that we are using a standard function where we only look in plane within a given 
            # slice. 
            d = np.gradient(self.smoothed_data[..., i])
            dslice, dz, dx = d
            dz /= self.xy_resolution
            dx /= self.xy_resolution
            
            gradient_magnitude_smoothed[..., i] = np.sqrt(dz**2 + dx**2)

        key = 'data_smoothed_%s_thr_%s_gradient_image_2D' % (self.fwhm, self.thr)

        self.h5file.create_dataset(key, data=gradient_magnitude_smoothed)
        self.h5file.flush()     
    
    def get_coronal_slice(self, slice, stain=None, smoothed=True, gradient=False):

        if gradient:
            data = self.gradient_image
        else:
            if smoothed:
                data = self.smoothed_data
            else:
                data = self.data
        
        if stain == None:
            return data[self._get_index_slice(slice), ...]
        
        else:
            return data[self._get_index_slice(slice), ..., self._get_index_stain(stain)]

        
    def get_axial_slice(self, slice, stain=None, smoothed=True, gradient=False):
        
        if gradient:
            data = self.gradient_image
        else:
            if smoothed:
                data = self.smoothed_data
            else:
                data = self.data

        if stain == None:
            return data[:, slice, :, ...]
        
        else:
            return data[:, slice, :, self._get_index_stain(stain)]


    def get_sagittal_slice(self, slice, stain=None, smoothed=True, gradient=False):
        
        if gradient:
            data = self.gradient_image
        else:
            if smoothed:
                data = self.smoothed_data
            else:
                data = self.data

        if stain == None:
            return data[:, :, slice, ...]
        
        else:
            return data[:, :, slice, self._get_index_stain(stain)]

        
        
    def get_coronal_mask(self, slice):
        
        #mask = self.mask[self._get_index_slice(slice), :, :, :].sum(-1) >= thr
        mask = self.thresholded_mask[self._get_index_slice(slice), ...]
        
        if mask.sum() == 0:
            n_masks = (self.mask_pandas.slice == slice).sum()
            print 'Warning: only %d masks available for coronal slice %d' % (n_masks, slice)
        
        return mask 

    def get_axial_mask(self, slice):

        #mask = self.mask[:, slice, :, :].sum(-1)        
        mask = self.thresholded_mask[:, slice, :]
        if mask.sum() == 0:
            n_masks = self.mask[:, slice, :, :].sum(-1).max()
            print 'Warning: only %d masks available for axial slice %d' % (n_masks, slice)

        return mask


    def get_sagittal_mask(self, slice):

        mask = self.thresholded_mask[:, :, slice]
        if mask.sum() == 0:
            n_masks = self.mask[:, :, slice, :].sum(-1).max()
            print 'Warning: only %d masks available for axial slice %d' % (n_masks, slice)

        return mask

    def plot_slice(self, orientation='coronal', **kwargs):
            
        if orientation == 'coronal':
            self.plot_coronal_slice(**kwargs)
        elif orientation == 'axial':
            self.plot_axial_slice(**kwargs)
        elif orientation == 'sagittal':
            self.plot_sagittal_slice(**kwargs)


    def plot_axial_slice(self, slice=None, stain='SMI32', image=None, gradient=False, outline_color='black', cmap=plt.cm.hot, plot_mask=False, crop=True, smoothed=True, mask_out=True, **kwargs):

            if slice is None:
                slice = self.center_of_mass[1]
            
            if (image == 'gradient_image') or gradient:
                im = self.get_axial_slice(slice, stain, smoothed=smoothed, gradient=True)
            elif image is None:
                im = self.get_axial_slice(slice, stain, smoothed=smoothed)

            if len(im.shape) != 2:
                im = image[:, slice, :]

            mask = self.get_axial_mask(slice)

            if mask_out:
                im = np.ma.masked_array(im, ~mask)
            
            plt.title('Axial slice %d (%.2fmm)\n' % (slice, self.get_z_coordinate(slice)))

            if crop:

                xlim = self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin
                extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0])

                vmin = self.vmin[self._get_index_stain(stain)]
                vmax = self.vmax[self._get_index_stain(stain)]

                plt.imshow(im[:, xlim[0]:xlim[1]], cmap=cmap, aspect=1, extent=extent, interpolation='nearest', vmin=vmin,vmax=vmax)
                
                
            if plot_mask:
                plt.contour(mask[:, xlim[0]:xlim[1]], origin='upper', colors=[outline_color], levels=[0,1], extent=extent)
        
        
    def plot_coronal_slice(self, slice=None, image=None, gradient=False, stain='SMI32', outline_color='black', cmap=plt.cm.hot, plot_mask=False, crop=True, smoothed=True, mask_out=True, **kwargs):
            
            if slice is None:
                slice = np.abs(self.slice_available.index.values - int(self.center_of_mass[0])).argmin()
                slice = self.slice_available.iloc[slice].name

            if image is None:

                if image == 'gradient_image':
                    gradient = True
                
                image = self.get_coronal_slice(slice, stain, smoothed=smoothed, gradient=gradient)

                if len(image.shape) != 2:
                    image = image[:, slice, :]

            
            mask = self.get_coronal_mask(slice)

            print image.shape, mask.shape

            if mask_out:
                image = np.ma.masked_array(image, ~mask)
            
            
            plt.title('Coronal slice %d (%.2fmm)\n' % (slice, self.get_slice_coordinate(slice)))
            

            
            if crop:
                
                
                xlim = self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin
                zlim = self.zlim[0] - self.crop_margin, self.zlim[1] + self.crop_margin
                
                extent = self.get_x_coordinate(xlim[0]), self.get_x_coordinate(xlim[1]), self.get_z_coordinate(zlim[1]), self.get_z_coordinate(zlim[0])
                
                if smoothed: 
                    vmin = self.vmin[self._get_index_stain(stain)]
                    vmax = self.vmax[self._get_index_stain(stain)]
                else:
                    vmin = None
                    vmax = None

                plt.imshow(image[zlim[0]:zlim[1], xlim[0]:xlim[1]], cmap=cmap, aspect=1, extent=extent, interpolation='nearest', vmin=vmin, vmax=vmax, **kwargs)
                
                
            if plot_mask:
                plt.contour(mask[zlim[0]:zlim[1]:, xlim[0]:xlim[1]], origin='upper', colors=[outline_color], levels=[0,1], extent=extent)
    #             plt.imshow(mask[zlim[0]:zlim[1]:, xlim[0]:xlim[1]])


    def plot_sagittal_slice(self, slice=None, image=None, gradient=False, stain='SMI32', outline_color='black', cmap=plt.cm.hot, plot_mask=False, crop=True, smoothed=True, mask_out=True, **kwargs):
            
            if slice == None:
                slice = self.center_of_mass[2]

            mask = self.get_sagittal_mask(slice)

            if image is None:
                if image == 'gradient_image':
                    gradient = True
                
                image = self.get_sagittal_slice(slice, stain, smoothed=smoothed, gradient=gradient)

                if len(image.shape) != 2:
                    image = image[:, slice, :]

                if mask_out:
                    image = np.ma.masked_array(image, ~mask)
            
            plt.title('Sagittal slice %d (%.2fmm)\n' % (slice, self.get_x_coordinate(slice)))

            if crop:
                ylim = self.zlim[0] - self.crop_margin, self.zlim[1] + self.crop_margin
            
                if smoothed:
                    vmin = self.vmin[self._get_index_stain(stain)]
                    vmax = self.vmax[self._get_index_stain(stain)]
                else:
                    vmin = None
                    vmax = None

                extent = self.get_slice_coordinate(self.slices[-1]), self.get_slice_coordinate(self.slices[0]), self.get_z_coordinate(ylim[1]), self.get_z_coordinate(ylim[0]), 
                plt.imshow(image[::-1, ylim[0]:ylim[1]].T, origin='upper', cmap=cmap, aspect=1, extent=extent, interpolation='nearest', vmin=vmin,vmax=vmax)
                
                
            if plot_mask:
                plt.contour(mask[::-1, ylim[0]:ylim[1]].T, origin='upper', colors=[outline_color], levels=[0,1], extent=extent)


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
                new_slice = 0.5 * self.get_coronal_slice(slice_minus_one, stain, smoothed=False) + 0.5 * self.get_coronal_slice(slice_plus_one, stain, smoothed=False)
		print new_slice.dtype, new_slice.shape
		print self.data.dtype, self.data.shape
		
                
                self.data[self._get_index_slice(slice), ..., self._get_index_stain(stain)] = new_slice
            else:
                print ' * slice %s' % slice + ' (can NOT be interpolated)'
                #self.data[self._get_index_slice(slice), ...] = np.nan
                
    def get_slice_coordinate(self, slice):        
        return (self.slices[-1] - slice) / 50 * self.z_resolution
    
    
    def get_x_coordinate(self, x):
        return (x - self.xlim[0]) * self.xy_resolution
    
    def get_z_coordinate(self, z):
        return (self.zlim[1] - z) * self.xy_resolution    
               
    
        
    def interpolate_stains(self, stains=None):

        if stains is None:
            stains = self.slice_available.columns
        
        for stain in stains:
            print " *** %s ***" % stain
            self.interpolate_stain(stain) 



    def get_proportional_slice(self, q, orientation='coronal'):
        
        assert((q >= 0) and (q <= 1))
        
        if orientation == 'coronal':
            coordinate = self.slices[-1] - q * (self.slices[-1] - self.slices[0])
            return self.get_nearest_active_slice(coordinate=coordinate)
        
        if orientation == 'sagittal':
            return int(self.xlim[0] + q * (self.xlim[1] - self.xlim[0]))
        
        if orientation == 'axial':
            return int(self.zlim[1] + q * (self.zlim[0] - self.zlim[1]))


    def get_nearest_active_slice(self, index=None, coordinate=None):    
        
        assert((index is None) or (coordinate is None))

        if index is not None:
            coordinate = self.slices[int(np.round(index))]
        
        return self.find_nearest(self.slices, coordinate)
    
    
    @staticmethod
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
