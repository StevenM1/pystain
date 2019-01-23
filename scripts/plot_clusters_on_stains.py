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


#subject_ids = [12062, 12104, 13095, 14037, 14051, 14069, 15033, 15035, 15055]
#subject_ids = [15055]
subject_ids = [12104, 13095, 14037, 14051, 14069, 15033, 15035, 15055]

max_n_components = 5
mapper = pandas.read_pickle('/home/gdholla1/data/post_mortem/clusterings_v1/group_clusters.pandas')

for subject_id in subject_ids[:]:
    for fwhm in [0.15, 0.3]:
        #try:
        d = '/home/gdholla1/data/post_mortem/clusterings_v1/%s_%s/' % (subject_id, fwhm)

        if not os.path.exists(d):
            os.makedirs(d)

        dataset = StainDataset(subject_id)
        cluster = StainCluster(dataset, interpolate_missing_slices=False, fwhm=fwhm)
        cluster.make_feature_vector()
        cluster.drop_stain('TH')
        incomplete_slices = cluster.get_incomplete_slices()
        cluster.drop_slices(incomplete_slices)
        cluster.normalize_feature_vector()

        fn = os.path.join(d, 'gmms.pkl')
        cluster.gmms = pkl.load(open(fn, 'r'))


        cluster.cluster_palette = sns.color_palette('husl', max_n_components)
        cluster.cluster_cmaps = [sns.light_palette((i * 360 / max_n_components, 90, 60), 256, input="husl", as_cmap=True) for i in np.arange(max_n_components)]
        cluster.cluster_n_components = range(1, max_n_components+1)
        cluster.cluster_predictions = {}            

        bics = [gmm.bic(cluster.feature_vector) for gmm in cluster.gmms]

        fn = os.path.join(d, 'clusters_on_stains.pdf')

        pdf = PdfPages(fn)


        sns.set_style('whitegrid')

        for n_components in [1,2,3,4,5]:
        #for n_components in [1,2]:

            print subject_id, fwhm, n_components
            new_gmms_idx = mapper.ix[str(subject_id), fwhm, n_components].group_cluster_idx.tolist()
            
            
            cluster.gmms[n_components-1].means_ = cluster.gmms[n_components-1].means_[new_gmms_idx]
            cluster.gmms[n_components-1].covars_ = cluster.gmms[n_components-1].covars_[new_gmms_idx]
            cluster.gmms[n_components-1].weights_ = cluster.gmms[n_components-1].weights_[new_gmms_idx]

            cluster.plot_cluster_means(n_components)
            plt.savefig(pdf, format='pdf')
            plt.close(plt.gcf())

            for q in [0.1, 0.3, 0.5, 0.75, 0.9][:]:
                slice = cluster.get_proportional_slice(q)
                print q, slice
                
                plt.figure()
                
                for i, stain in enumerate(cluster.stains):
                    print i, stain
                    plt.subplot(1, len(cluster.stains), i+1)
                    cluster.plot_stain_cluster(slice=slice, n_components=n_components, mask_thr=.5, linewidths=2, stain=stain)
                    title = plt.gca().get_title() + ' %s, %d clusters' % (stain, n_components)
                    plt.title(title)
                    
                plt.gcf().set_size_inches(11 * 10, 10)
                plt.savefig(pdf, format='pdf')
                plt.close(plt.gcf())


        pdf.close()


    #try:
    #except Exception as e:
        #print "Error with %s/%s: %s" % (subject_id, fwhm, e)


