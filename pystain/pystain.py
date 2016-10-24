import os
import numpy as np
import pandas

class StainDataset(object):
    
    xy_resolution = (40.5 / 2874)
    z_resolution = 0.3
    base_dir = '/home/gdholla1/data/post_mortem/new_data_format/'
    base_dir = os.path.join(os.environ['HOME'], 'post_mortem/new_data_format')
    
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
        return dataset.slice_available.index.get_loc(slice)

    
    def _get_index_stain(self, stain):
        return dataset.slice_available.columns.get_loc(stain)
    
    @property
    def center_of_mass(self):
        
        if self._center_of_mass is None:
            self._center_of_mass = ndimage.center_of_mass(dataset.mask.value.sum(-1))
            
        return self._center_of_mass
    
    def get_limits(self):
        
        _, zs, xs, _ = np.where(dataset.mask.value > 0)
        
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
        
        if not self.slice_available.ix[slice, stain]:
            print 'WARNING: slice {stain} for slice {slice} not available'.format(**locals())
        
        return self.mask[self._get_index_slice(slice), :, :, :].sum(-1) >= thr
    
    def get_axial_mask(self, slice, thr=3):
        return self.mask[:, slice, :, :].sum(-1) >= thr    
        
        
    def plot_axial_slice(self, slice=None, stain='SMI32', fwhm=0.15, plot_mask=True, cmap=plt.cm.hot, **kwargs):
        
        
        sigma = fwhm / 2.335
        
        sigma_x = sigma / self.xy_resolution # anterior-posterior 
        sigma_y = sigma / self.z_resolution # inferior-superior
        
        print sigma_x, sigma_y
        
        im = self.get_sagital_slice(slice, stain)
        
        if np.isnan(im).any():
            V=im.copy()
            V[im!=im]=0
            VV=sp.ndimage.gaussian_filter(V,sigma=[sigma_x, sigma_y])

            W=0*im.copy()+1
            W[im!=im]=0
            WW=sp.ndimage.gaussian_filter(W,sigma=[sigma_x, sigma_y])
            
            im=VV/WW
        
#         im = sp.ndimage.gaussian_filter(im, sigma=[sigma_x, sigma_y])
        
        print im.shape
        
        plt.imshow(im, origin='lower', cmap=cmap, aspect=self.z_resolution/self.xy_resolution, interpolation='nearest')
        plt.axis('off')
        
        if plot_mask:
            m = self.get_sagital_mask(slice)            
            plt.contour(m, origin='lower', colors=['green'], levels=[0,1])
        
        
    def plot_coronal_slice(self, slice=None, stain='SMI32', fwhm=0.15, cmap=plt.cm.hot, plot_mask=True, crop=True, **kwargs):
        
        if slice == None:
            slice = np.abs(self.slice_available.index.values - int(self.center_of_mass[1])).argmin()
            slice = self.slice_available.iloc[slice].name

        sigma = fwhm / 2.335
        
        sigma_x = sigma / self.xy_resolution # left-right
        sigma_y = sigma / self.xy_resolution # anterior-posterior 
        
        im = self.get_coronal_slice(slice, stain)
        im = sp.ndimage.gaussian_filter(im, sigma=[sigma_x, sigma_y])
        
        plt.imshow(im, origin='lower', cmap=cmap, aspect=1, interpolation='nearest')
        plt.axis('off')
        
        
        if plot_mask:
            m = self.get_coronal_mask(slice)            
            plt.contour(m, origin='lower', colors=['green'], levels=[0,1])
        
        plt.title('y = %d' % slice)
        
        if crop:
            plt.xlim(self.xlim[0] - self.crop_margin, self.xlim[1] + self.crop_margin)
            plt.ylim(self.zlim[0] - self.crop_margin, self.zlim[1] + self.crop_margin)
        
