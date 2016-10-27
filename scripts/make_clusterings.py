import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pystain
from pystain import StainDataset, StainCluster

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas
import numpy as np
import seaborn as sns
import os
import pickle as pkl


mask_thr = 0.5

sns.set_style('whitegrid')


subject_ids = [15055, 13054, 13061, 13072, 13074, 13095, 14037, 14051, 14069, 15033, 15035, 15052]
#subject_ids = [15055]

max_n_components = 5

for subject_id in subject_ids:
    for fwhm in [0.15, 0.3]:

        print subject_id, fwhm

        d = '/home/gdholla1/data/post_mortem/clusterings_v1/%s_%s/' % (subject_id, fwhm)

        if not os.path.exists(d):
            os.makedirs(d)

        try:

            dataset = StainDataset(subject_id)
            cluster = StainCluster(dataset, interpolate_missing_slices=False, fwhm=fwhm)
            cluster.make_feature_vector()
            cluster.drop_stain('TH')
            incomplete_slices = cluster.get_incomplete_slices()
            cluster.drop_slices(incomplete_slices)
            cluster.normalize_feature_vector()


            fn = os.path.join(d, 'incomplete_slices.txt')
            f = open(fn, 'w')
            f.writelines([' * %d\n\r' % e for e in incomplete_slices])
            f.close()
            

            cluster.cluster_gmm(range(1, max_n_components+1))
            
            fn = os.path.join(d, 'gmms.pkl')
            pkl.dump(cluster.gmms, open(fn, 'w'))

            bics = [gmm.bic(cluster.feature_vector) for gmm in cluster.gmms]
            
            fn = os.path.join(d, 'clusters.pdf')

            pdf = PdfPages(fn)
            
            for n_components in xrange(1, max_n_components+1):
                print "Making pictures for %d components" % n_components

                cluster.plot_cluster_means(n_components)
                plt.savefig(pdf, format='pdf')
                plt.close(plt.gcf())

                for i in xrange(1, n_components+1):
                    print "Plotting component %d" % i

                    for j, q in enumerate([0.25, 0.5, 0.75]):
                        slice_coronal = cluster.get_proportional_slice(q)
                        slice_axial = cluster.get_proportional_slice(q, 'axial')
                        slice_sagittal = cluster.get_proportional_slice(q, 'sagittal')  

                        plt.subplot(3,3,(3*j)+1)
                        cluster.plot_cluster_probability(component=i, n_components=n_components,slice=slice_coronal, mask_thr=mask_thr)
                        xticks = np.arange(0, np.ceil(cluster.get_x_coordinate(cluster.stain_dataset.xlim[1])))
                        plt.xticks(xticks)
                        yticks = np.arange(0, np.ceil(cluster.get_z_coordinate(cluster.stain_dataset.zlim[0])))
                        plt.yticks(yticks)

                        plt.subplot(3,3,(3*j)+2)
                        cluster.plot_cluster_probability('axial', component=i, n_components=n_components,slice=slice_axial, mask_thr=mask_thr)        
                        xticks = np.arange(0, np.ceil(cluster.get_x_coordinate(cluster.stain_dataset.xlim[1])))
                        plt.xticks(xticks)
                        yticks = np.arange(0, np.ceil(cluster.get_slice_coordinate(cluster.slices[0])))
                        plt.yticks(yticks)

                        plt.subplot(3,3,(3*j)+3)
                        cluster.plot_cluster_probability('sagittal', component=i, n_components=n_components,slice=slice_sagittal, mask_thr=mask_thr)                

                        xticks = np.arange(0, np.ceil(cluster.get_slice_coordinate(cluster.slices[0])))
                        plt.xticks(xticks)
                        yticks = np.arange(0, np.ceil(cluster.get_z_coordinate(cluster.stain_dataset.zlim[0])))
                        plt.yticks(yticks)

                plt.gcf().set_size_inches(25, 25)


                plt.suptitle('%d clusters, BIC=%s' % (n_components, bics[n_components-1]), fontsize=24)

                plt.savefig(pdf, format='pdf')
                plt.close(plt.gcf())

            pdf.close()

        except Exception as e:
            print "Error with %s/%s: %s" % (subject_id, fwhm, e)
