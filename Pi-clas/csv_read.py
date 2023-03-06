import csv
import numpy as np
import random
import glob
import os.path
import sys
import pandas as pd
import operator
from collections import Counter
import numpy as np
import sys,os
import codecs
import tensorflow as tf

def save_(name,data):
	with open(name+'.csv', 'w', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerows(data)#remove the last candle which is not closed yet

def app_(name,data):
	with open(name+'.csv', 'a+', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerow(data)#remove the last candle which is not closed yet

#def normalize(input_data,mm):
	#return (input_data/ mm).astype(np.float32)

def normalize(input_data):
	data_set_d_ = input_data-np.min(input_data)# ? check if it works wrong with dim=4
	data_set_d = data_set_d_/np.max(data_set_d_)
	return data_set_d

def get_data(L_path):
	with open(L_path, 'r') as fin:
		reader = csv.reader(fin)
		#train = list(reader)# loads str

		# load float
		tr,pr=[],[]
		# time, open, high, low, close, tick_volume, spread, real_volume
		#  0     1     2     3      4       5           6        7
		for row in reader:

			tr.append(np.float_(row[0]))# read only 4 colums (open, high, low, close,)
			pr.append(np.float_(row[1]))# read only 4 colums (open, high, low, close,)

	return tr,pr

def collect(csv_data):
	data = [row[3] for row in csv_data]
	return data

'''def print_head(data,rows):
	print('first %d lines:' %(rows))
	for i in range(rows):
		print(data[i])
'''
#read data from csv file
def read_csv(data_path,currency_name,time_step,pp,dp):

	csv_name = data_path + currency_name + time_step +'.csv'
	print('path of data: ', csv_name)

	csv_data=get_data(csv_name)# get all data
	ll = len(csv_data)
	print('total number of data: ', ll)

	d = int(ll*dp)# divide data 
	#data = csv_data[d:]
	data = csv_data[:d]

	#p = int(len(data)*pp)# divide data for train and test sets

	#train = data[:p]
	#test = data[p:]

	return data,len(data)

def oneHot(data,data_column):

	i = 0

	if data_column == 'close':

		if data[0]<data[1]:# increased
			i = 2
		elif data[0]>data[1]:# decreased
			i = 0
		else: # no change
			i = 1
	else:
		if data[0][3]<data[1][3]:# increased
			i = 2
		elif data[0][3]>data[1][3]:# decreased
			i = 0
		else: # no change
			i = 1

	indx=tf.one_hot(i, 3)
	indx = indx[np.newaxis,...]
	indx = np.array(indx)

	return indx

#load data
def load_data_assist(data,data_column,in_len,out_len,data_type,d_div):

	data = np.array(data)
	dataX = []
	dataY = []
	dataZ = []

	#collect close column
	# open, high, low, close
	#   0     1    2     3
	ll = data.shape[0]

	#print_head(data,3)
	data = normalize(data)

	for i in range(ll-in_len-out_len):
		t = i+in_len
		data_A = data[i:t]
		data_A_ = collect(data_A)# collect close
		data_B = collect(data[t:t+out_len])# collect close

		data_B = np.array(data_B)
		data_A = np.array(data_A)
		data_A_ = np.array(data_A_)
		#print(data_A.shape)
		if data_type == '2d':
			data_A = data_A.reshape((int(in_len*4/d_div), d_div))
		elif len(data_A) != in_len or len(data_B) != out_len:
			#print(len(data_A),data_A.shape,in_len)
			print('data with lack of size is skipped !!!')
			continue 
		#print(data_A.shape)
		#sys.exit()

		data_A = data_A[np.newaxis,...]#add 1 more dimension
		#data_B = data_B[np.newaxis,...]#add 1 more dimension
		data_A = data_A[np.newaxis,...]#add 1 more dimension
		#data_B = data_B[np.newaxis,...]#add 1 more dimension
		#print(data_A.shape)

		data_A = np.moveaxis(data_A, 0, -1)#reorder the shape of dimension
		#data_B = np.moveaxis(data_B, 0, -1)#reorder the shape of dimension

		#print(data_A.shape)
		#sys.exit()
		dataX.append(data_A)
		dataZ.append(data_A_)	
		dataY.append(data_B)	
	return dataX, dataY, dataZ

def load_data(train,data_column,in_len,out_len,data_type,d_div):
	
	trainX, trainY, trainX_ = load_data_assist(train,data_column,in_len,out_len,data_type,d_div)

	print('train dataX: ',len(trainX))
	print('train dataX_: ',len(trainX_))
	print('train dataY: ',len(trainY))

	return trainX, trainY, trainX_



