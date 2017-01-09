
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from pystain import StainDataset
import os
import numpy as np

import seaborn as sns
sns.set_context('poster')
sns.set_style('whitegrid')

subject_ids = [12104, 13095, 14037, 14051, 14069, 15033, 15035, 15055]

def cmap_hist(data, bins=None, cmap=plt.cm.hot, vmin=None, vmax=None):
    n, bins, patches = plt.hist(data, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # scale values to interval [0,1]
    col = (bin_centers - vmin) / vmax

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))


for subject_id in subject_ids[1:]:
    for fwhm in [0.15, 0.3]:
        dataset = StainDataset(subject_id, fwhm=fwhm)
        dataset.get_vminmax((0, 99))

        d = '/home/gdholla1/data/post_mortem/visualize_stains_v1/%s/' % (subject_id)
        if not os.path.exists(d):
            os.makedirs(d) 

        fn = os.path.join(d, 'stains_%s.pdf' % fwhm)
        pdf = PdfPages(fn)

        for i, stain in enumerate(dataset.stains):
            print 'Plotting %s' % stain
            plt.figure()
            data = dataset.smoothed_data.value[dataset.thresholded_mask, i]
            data = data[~np.isnan(data)]
            bins = np.linspace(0, dataset.vmax[i], 100)
            cmap_hist(data, bins, plt.cm.hot, vmin=dataset.vmin[i], vmax=dataset.vmax[i])
            plt.title(stain)
            plt.savefig(pdf, format='pdf')

            plt.close(plt.gcf())

            plt.figure()

            if not os.path.exists(d):
                os.makedirs(d)

            for i, orientation in enumerate(['coronal', 'axial', 'sagittal']):
                for j, q in enumerate([.25, .5, .75]):
                    ax = plt.subplot(3, 3, i + j*3 + 1)
                    slice = dataset.get_proportional_slice(q, orientation)
                    dataset.plot_slice(slice=slice, stain=stain, orientation=orientation, cmap=plt.cm.hot)
                    ax.set_anchor('NW')

                    
            plt.gcf().set_size_inches(20, 20)
            plt.suptitle(stain)
            plt.savefig(pdf, format='pdf')
            plt.close(plt.gcf())
        
        plt.figure()
        print 'Plotting correlation matrix'
        sns.heatmap(np.round(dataset.smoothed_dataframe.corr(), 2), annot=True)
        plt.savefig(pdf, format='pdf')
        plt.close(plt.gcf())

        pdf.close()
