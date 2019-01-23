import re
import pandas
import glob
import h5py
import scipy as sp
from scipy import ndimage
import natsort
import numpy as np

fns = glob.glob('/home/gdholla1/data/post_mortem/STACKED_SLIDES/*/*')
reg = re.compile('.*/(?P<subject_id>[0-9]{5})_PNG/(?P<stain>[A-Za-z0-9]+)_(?P<slice>[0-9]+)_[0-9]+_(?P<id>[0-9]+)\.png')

df = pandas.DataFrame([reg.match(fn).groupdict() for fn in fns if reg.match(fn)])
df['subject_id'] = df['subject_id'].astype(int)
df['slice'] = df['slice'].astype(int)
df['fn'] = [fn for fn in fns if reg.match(fn)]
df['id'] = df['id'].astype(int)


df = df.drop_duplicates(['subject_id', 'slice', 'stain'], keep='last')


def correct_stain(stain):
    
    if stain == 'Cal':
        return 'CALR'
    
    if stain == 'FERR':
        return 'FER'
    
    if stain == 'Ferr':
        return 'FER'
    
    if stain == 'GABRA':
        return 'GABRA3'
    
    if stain == 'GAD':
        return 'GAD6567'
    
    if stain == 'GAD6557':
        return 'GAD6567'
    
    if stain == 'PV':
        return 'PARV'    
    
    if stain == 'SMI3':
        return 'SMI32' 
    
    if stain == 'Syn':
        return 'SYN'     

    if stain == 'TF':
        return 'TRANSF'
    
    
    return stain

df['stain'] = df.stain.map(correct_stain).astype(str)

df.to_pickle('/home/gdholla1/data/post_mortem/data.pandas')


reg3 = re.compile('/home/gdholla1/data/post_mortem/MasksForGilles/(?P<subject_id>[0-9]{5})_RegMasks_(?P<rater>[A-Z]+)/(?P<stain>[A-Z0-9a-z_]+)_(?P<slice>[0-9]+)_([0-9]+)_(?P<id>[0-9]+)\.png')
fns = glob.glob('/home/gdholla1/data/post_mortem/MasksForGilles/*_RegMasks_*/*_*_*_*.png')
masks = pandas.DataFrame([reg3.match(fn).groupdict() for fn in fns])
masks['fn'] = fns
masks['subject_id'] = masks['subject_id'].astype(int)
masks['slice'] = masks['slice'].astype(int)

masks.set_index(['subject_id', 'slice', 'stain', 'rater'], inplace=True)
masks.sort_index(inplace=True)

masks.to_pickle('/home/gdholla1/data/post_mortem/masks.pandas')

mask_stains = ['PARV', 'SMI32']
raters_a = ['KH', 'MT']

raters_b = ['MCK', 'AA']

import os

for subject_id, d in df.groupby(['subject_id']):
    print subject_id
    
    slices = natsort.natsorted(d.slice.unique())
    
    print slices
    
    stains = natsort.natsorted(d.stain.unique())
    resolution = ndimage.imread(d.fn.iloc[0]).shape

    data_array = np.zeros((len(slices),) + resolution + (len(stains),))
    data_array[:] = np.nan
    
    
    print 'Storing data'
    for idx, row in d.iterrows():
        
        slice_idx = slices.index(row['slice'])
        stain_idx = stains.index(row['stain'])
        
        data_array[slice_idx, ..., stain_idx] = ndimage.imread(row.fn)
        
    mask_array = np.zeros((len(slices),) + resolution + (4,))
    
    
    print 'Storing masks'
    for idx, row in masks.ix[subject_id].reset_index().iterrows():
        
        slice_idx = slices.index(row['slice'])
        
        if row.rater in raters_a:
            last_idx = mask_stains.index(row.stain) * 2 + raters_a.index(row.rater)
        else:
            last_idx = mask_stains.index(row.stain) * 2 + raters_b.index(row.rater)
        
        im = ndimage.imread(row.fn)
        mask_array[slice_idx, ..., last_idx] = im > np.percentile(im, 70)
        
        
    print 'Creating HDF5 file'
    p = '/home/gdholla1/data/post_mortem/new_data_format/%s/' % subject_id
    
    if not os.path.exists(p):
        os.makedirs(p)
    
    new_file = h5py.File(os.path.join(p, 'images.hdf5' % subject_id), )
    new_file.create_dataset('data', data=data_array)
    new_file.create_dataset('mask', data=mask_array.astype(bool))
    new_file.close()
    
    d.to_pickle(os.path.join(p, 'data.pandas'))
    masks.ix[subject_id].reset_index().to_pickle(os.path.join(p, 'masks.pandas'))



