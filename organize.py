from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


dataset_home = './'
subdirs = ['training/', 'testing/', 'validation/']
for subdir in subdirs:
	labeldirs = ['espicula/', 'normal/', 'piscada/', 'ruido/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)

seed(1)
src_directory = 'original_signals/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'training/'
	if random() < 0.05:
		dst_dir = 'testing/'
	elif random() < 0.2:
		dst_dir = 'validation/'

	if file.startswith('Esp¡cula'):
		dst = dataset_home + dst_dir + 'espicula/'  + file
		copyfile(src, dst)
	elif file.startswith('Normal'):
		dst = dataset_home + dst_dir + 'normal/'  + file
		copyfile(src, dst)
	elif file.startswith('Piscada'):
		dst = dataset_home + dst_dir + 'piscada/'  + file
		copyfile(src, dst)
	elif file.startswith('Ruido'):
		dst = dataset_home + dst_dir + 'ruido/'  + file
		copyfile(src, dst)
