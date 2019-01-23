import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

max_n_components = 6
fwhm = 0.3

sns.set_style('whitegrid')
sns.set_context('poster')

import os
import pandas

import numpy as np
import itertools

from multiprocessing import Pool

import scipy as sp
from scipy import optimize

import pickle as pkl
from pystain import StainDataset

from pystain.cluster import plot_clusters_axial, sort_model_clusters, SimpleExgaussMixture, plot_clusters_coronal, plot_clusters_sagittal

cluster_colors = [sns.light_palette((i * 360 / max_n_components, 90, 60), 256, input="husl") for i in np.arange(max_n_components)]
cluster_cmaps = [sns.light_palette((i * 360 / max_n_components, 90, 60), 256, input="husl", as_cmap=True) for i in np.arange(max_n_components)]

subject_ids = [13095, 14037, 14051, 14069, 15033, 15035, 15055]
n_subj = len(subject_ids)

dss = [StainDataset(subject_id, fwhm=fwhm) for subject_id in subject_ids]

stains = [u'CALR', u'FER', u'GABRA3', u'GAD6567', u'MBP', u'PARV', u'SERT', u'SMI32', u'SYN', u'TH', u'TRANSF', u'VGLUT1']


for stain in stains[:]:

    fn = os.path.join('/home/gdholla1/data/post_mortem/ml_clusters_visualized/', '%s.pdf' % stain) 
    pdf = PdfPages(fn)

    for n_clusters in xrange(1, 7):
        fns = ['/home/gdholla1/data/post_mortem/ml_clusters_v2/{subject_id}_{fwhm}_{stain}_{n_clusters}.pkl'.format(**locals()) for subject_id in subject_ids]
        fns = [fn if os.path.exists(fn) else None for fn in fns]
        models = [pkl.load(open(fn)) if fn is not None else None for fn in fns ]

        models= [sort_model_clusters(model) if model is not None else None for model in models]
        

        fig = plt.figure(figsize=(40, 60))
        for i, subject_id in enumerate(subject_ids[:]):
            
            if models[i] is not None:
            
                plt.subplot(19, len(subject_ids), i+1)
                models[i].plot_fit(colorized=True)
                plt.title(subject_id)
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                ## *** CORONAL ***
                # Posterior slice
                slice = dss[i].get_proportional_slice(0.25)
                
                plt.subplot(19, len(subject_ids), i + 1 + 1 * n_subj)    
                dss[i].plot_coronal_slice(slice=slice, stain=stain)
                plt.title('Posterior coronal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 2 * n_subj)    
                plot_clusters_coronal(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                # Middle slice
                slice = None
                
                plt.subplot(19, len(subject_ids), i + 1 + 3 * n_subj)
                dss[i].plot_coronal_slice(slice, stain=stain)   
                plt.title('Middle coronal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 4 * n_subj)    
                plot_clusters_coronal(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                # ANTERIOR SLICE
                slice = dss[i].get_proportional_slice(0.75)
                
                plt.subplot(19, len(subject_ids), i + 1 + 5 * n_subj)    
                dss[i].plot_coronal_slice(slice=slice, stain=stain)
                plt.title('Anterior coronal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 6 * n_subj)    
                plot_clusters_coronal(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])   
                
                
                ## ***SAGITTAL ***
                # MEDIAL
                slice = dss[i].get_proportional_slice(0.25, orientation='sagittal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 7 * n_subj)    
                dss[i].plot_sagittal_slice(slice=slice, stain=stain)
                plt.title('Medial sagittal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 8 * n_subj)    
                plot_clusters_sagittal(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                # MIDDLE
                slice = None
                
                plt.subplot(19, len(subject_ids), i + 1 + 9 * n_subj)    
                dss[i].plot_sagittal_slice(slice=slice, stain=stain)
                plt.title('Middle sagittal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 10 * n_subj)    
                plot_clusters_sagittal(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                # LATERAL
                slice = dss[i].get_proportional_slice(0.75, orientation='sagittal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 11 * n_subj)    
                dss[i].plot_sagittal_slice(slice=slice, stain=stain)
                plt.title('Lateral sagittal')
                
                plt.subplot(19, len(subject_ids), i + 1 + 12 * n_subj)    
                plot_clusters_sagittal(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                
                #  *** AXIAL ****
                
                # INFERIOR
                slice = dss[i].get_proportional_slice(0.25, orientation='axial')
                
                plt.subplot(19, len(subject_ids), i + 1 + 13 * n_subj)    
                dss[i].plot_axial_slice(slice=slice, stain=stain)
                plt.title('Inferior axial')
                
                plt.subplot(19, len(subject_ids), i + 1 + 14 * n_subj)    
                plot_clusters_axial(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([]) 
                
                # MIDDLE
                slice = None
                
                plt.subplot(19, len(subject_ids), i + 1 + 15 * n_subj)    
                dss[i].plot_axial_slice(slice=slice, stain=stain)
                plt.title('Middle axial')
                
                plt.subplot(19, len(subject_ids), i + 1 + 16 * n_subj)    
                plot_clusters_axial(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([])
                
                # SUPERIOR
                slice = dss[i].get_proportional_slice(0.75, orientation='axial')
                
                plt.subplot(19, len(subject_ids), i + 1 + 17 * n_subj)    
                dss[i].plot_axial_slice(slice=slice, stain=stain)
                plt.title('Superior axial')
                
                plt.subplot(19, len(subject_ids), i + 1 + 18 * n_subj)    
                plot_clusters_axial(slice, stain, dss[i], models[i], n_clusters)    
                plt.gca().yaxis.set_ticklabels([])
                plt.gca().xaxis.set_ticklabels([]) 

        plt.savefig(pdf, format='pdf')
        plt.close(plt.gcf())

    pdf.close()

