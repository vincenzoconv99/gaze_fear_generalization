import numpy as np
import scipy.io as sio
from os.path import join, isdir
from os import makedirs

def save_event_features(data, dataset, filename, type='unknown', method='unknown', dset='train'):
	# Define the saving directory
	dir_name = 'features/'+dataset+'_'+type+'_'+method + '/' + dset
	if not isdir(dir_name):
		makedirs(dir_name)
	np.save(join(dir_name, filename), data)