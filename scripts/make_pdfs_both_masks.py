import h5py
import glob
import pandas
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import skimage
from skimage import feature
import scipy as sp
from scipy import ndimage
from natsort import natsorted


masks = h5py.File('/home/gdholla1/data/post_mortem/masks.hd5f')
subject_ids = masks.keys()

subject_ids = ['14037']

for subject_id in subject_ids[:]:
    print subject_id
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages('/home/gdholla1/projects/pm_quant/pdfs_both_masks/%s.pdf' % (subject_id))
    
    data = h5py.File('/home/gdholla1/data/post_mortem/%s.hd5f' % str(subject_id), 'r')
    
    slices = natsorted(data.keys())
    stains = natsorted(np.unique(np.concatenate([e.keys() for e in data.values()])))

    n_slices = len(slices)
    n_stains = len(stains)
    

    for i, slice in enumerate(slices[:]):
        print slice
        fig, axs = plt.subplots(1, n_stains)
        fig.set_size_inches(n_stains * 10, 6)

        for ax in axs.ravel():
            ax.set_axis_off()
        
        
        if 'SMI32' in data[slice].keys():
            im = data[slice]['SMI32'].value
            
            
            diff_mag = feature.canny(im.astype(float), sigma=25, use_quantiles=True, low_threshold=0.7, high_threshold=0.8)
            diff_mag = diff_mag > 0
            diff_mag = ndimage.binary_dilation(diff_mag, iterations=1)        
            diff_mag = np.ma.masked_equal(diff_mag, 0)

        for j, stain in enumerate(stains[:]):
            print stain
            plt.sca(axs[j])
            plt.title('{stain} - z = {slice}'.format(**locals()))
            
            if stain in data[slice].keys():
                im = data[slice][stain].value
                im = ndimage.gaussian_filter(im, 10)
                plt.imshow(im.T, origin='lower', cmap=plt.cm.hot)
                
                if 'SMI32' in data[slice].keys():
                    plt.imshow(diff_mag.T, origin='lower', cmap=plt.cm.hot, vmin=0, vmax=1)
                
                if (slice in masks[subject_id].keys()) and ('PARV' in masks[subject_id][slice].keys()):
                    for key, mask in masks[subject_id][slice]['PARV'].items():
                        plt.contour(mask.value.T, origin='lower', linewidths=1, colors=['blue'], levels=[0, 1])                
            
                if (slice in masks[subject_id].keys()) and ('SMI32' in masks[subject_id][slice].keys()):
                    for key, mask in masks[subject_id][slice]['SMI32'].items():
                        print key, mask.value.sum()
                        plt.contour(mask.value.T, origin='lower', linewidths=1, colors=['green'], levels=[0, 1])                
        plt.savefig(pdf, format='pdf')
        plt.close(fig)
                    
    data.close()
    pdf.close()
    
    
masks.close()
