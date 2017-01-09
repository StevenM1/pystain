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


import pymc3 as pm
from pymc3.distributions.continuous import ExGaussian
from pymc3.backends import Text

import theano.tensor as tt
import scipy as sp
import pandas
import shutil

subject_ids = [12104, 13095, 14037, 14051, 14069, 15033, 15035, 15055]

burnin = 20000
n_samples = 50000
trace_size = n_samples - burnin

def cmap_hist(data, bins=None, cmap=plt.cm.hot, vmin=None, vmax=None):
    
    if bins is None:
        bins = np.linspace(vmin, vmax, 100)
    
    n, bins, patches = plt.hist(data, bins=bins, normed=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # scale values to interval [0,1]
    col = (bin_centers - vmin) / vmax

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))

def exgauss_pdf(x, mu, sigma, nu):
    
    nu = 1./nu
    
    p1 = nu / 2. * np.exp((nu/2.)  * (2 * mu + nu * sigma**2 - 2 * x))
    p2 = sp.special.erfc((mu + nu * sigma**2 - x)/ (np.sqrt(2) * sigma))
    return p1 * p2

def plot_trace_uni(trace, dataset, stain, **kwargs):
    
    t = np.linspace(0, dataset.vmax[dataset._get_index_stain(stain)], 100)
    
    pdf1 = exgauss_pdf(t, trace['mu'], trace['sigma'], trace['nu'])
    
    l1 = plt.plot(t, pdf1, color=sns.color_palette()[0], label='cluster 1', **kwargs)
    
    return l1[0]
    



for subject_id in subject_ids[1:2]:
    print subject_id
    dataset = StainDataset(subject_id, fwhm=0.3)

    df = dataset.smoothed_dataframe
    df = df[~df.isnull().any(1)]

    d = '/home/gdholla1/data/post_mortem/cluster_bayes_v1/{subject_id}/'.format(**locals())

    if not os.path.exists(d):
        os.makedirs(d)
        
    pdf_fn = os.path.join(d, 'clusters_uni.pdf')
    pdf = PdfPages(pdf_fn)

    results = []

    n = 10000
    stepsize = df.shape[0] / n

    for stain in dataset.stains:
        print subject_id, stain


        x = df[stain].values[::stepsize]
        
        with pm.Model() as uni_model:   
            nu = pm.HalfNormal('nu', tau=5.)
            mu = pm.Normal('mu', mu=0, tau=1/10.)
            sigma = pm.HalfNormal('sigma', tau=1./10.)

            exgauss = ExGaussian('exgauss', mu=mu, sigma=sigma, nu=nu, observed=x)
            
            trace_fn = os.path.join(d, '{subject_id}_{stain}_uni'.format(**locals()))
            
            if os.path.exists(trace_fn):
                shutil.rmtree(trace_fn)
            
            db = Text(trace_fn)

        
    #         start = pm.find_MAP()
            step = pm.Metropolis()
            
            trace_ = pm.sample(n_samples, step, trace=db)
            trace = trace_[burnin:]
            
            try:
                print 'Finding MAP'
                MAP = pm.find_MAP()
                results.append(MAP)
            except Exception:
                MAP = {}
                results.append(MAP)
                print "Problem with finding MAP"

            try:
                print 'Calculating DIC'
                dic = pm.stats.dic(trace)
            except Exception:   
                dic = np.nan
                print "Problem with DIC"
            
            try:
                print 'Calculating BPIC'
                bpic = pm.stats.bpic(trace)
            except Exception:
                bpic = np.nan
                print "Problem with BPIC"
            
            try:
                print 'Calculating WAIC'
                waic = pm.stats.waic(trace)
            except Exception:
                waic = np.nan
                print "Problem with WAIC"


            results[-1].update({'subject_id':subject_id,
                                'stain':stain,
                                'bpic':bpic,
                                'dic':dic,
                                'waic':waic})

            
            try: 
                plt.figure()
                pm.traceplot(trace)
                plt.suptitle('{subject_id} - {stain} (one cluster)'.format(**locals()))
                plt.savefig(pdf, format='pdf')

                plt.figure()
                cmap_hist(x, vmin=dataset.vmin[dataset._get_index_stain(stain)], vmax=dataset.vmax[dataset._get_index_stain(stain)])

                for i in np.arange(0, trace_size, trace_size/500):
                    plot_trace_uni(trace[i], dataset, stain, alpha=0.1, lw=3)

                sns.despine()
                
                plt.title('{subject_id} - {stain} (one cluster)'.format(**locals()))
                plt.savefig(pdf, format='pdf')
                
                
                plt.close(plt.gcf())
            except Exception:
                print "Problem with plotting"
            
            
        results_pandas = pandas.DataFrame(results)
        results_pandas.to_pickle(os.path.join(d, 'results_uni.pandas'))
            
    pdf.close()


