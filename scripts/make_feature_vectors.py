import glob
import pystain
from pystain import StainDataset, StainCluster

fns = glob.glob('/home/gdholla1/data/post_mortem/new_data_format/*')

subject_ids = [int(fn.split('/')[-1]) for fn in fns]


for subject_id in subject_ids:
    dataset = StainDataset(subject_id)
    cluster = StainCluster(dataset, fwhm=0.3)
    cluster.make_feature_vector()
