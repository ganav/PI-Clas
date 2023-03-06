import os,cv2
import numpy as np
import csv
import glob, math
import os,sys,split_name_path,edit_path
import os.path
from subprocess import call

data_csv = []
path = '.\\archive\\'
dataset = path + 'thermal images UL\\'
sets = ['valid','test','train']
ext = '.jpg'
train_size = 0.7
test_size = 0.2

class_folders = glob.glob(dataset + '*')

for img_class in class_folders:
		
	class_files = glob.glob(img_class + '/*'+ext)
	length = len(class_files)

	valid = class_files[math.ceil(length*train_size)+math.ceil(length*test_size):]
	test = class_files[math.ceil(length*train_size):math.ceil(length*train_size)+math.ceil(length*test_size)]
	train = class_files[:math.ceil(length*train_size)]

	sets_ = [valid,test,train]

	for set_ in range(len(sets_)):

		for img_path in sets_[set_]:
			name, path1 = split_name_path.f(img_path)
			img = cv2.imread(img_path,1)
				
			if img is None:
				print('this image is NONE: ', img_path)
				continue
				
			#create dataset folder
			path2 = path + sets[set_] +'\\'
			if not os.path.exists(path2):
				os.mkdir(path2)

			path3 =  path2 + path1 + '\\'
			path4 =  path3 + name

			if not os.path.exists(path3):
				os.mkdir(path3)
			cv2.imwrite(path4,img)
	
print('done..')
