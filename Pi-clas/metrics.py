from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score,accuracy_score
from csv_read import get_data, app_
import numpy as np


def main():

	y_true, y_pred = get_data('acc.csv')
	mat = confusion_matrix(y_true, y_pred)

	'''
	to get tp,fp,tn,fn return MCM from the original source
	MCM = multilabel_confusion_matrix(y_true, y_pred,
	                                      sample_weight=sample_weight,
	                                      labels=labels, samplewise=samplewise)
	                                      '''
	rec,mcm = recall_score(y_true, y_pred, average=None)
	pre = precision_score(y_true, y_pred, average=None)
	f1 = f1_score(y_true, y_pred, average=None)

	rec = rec * 100
	pre = pre * 100
	f1 = f1 * 100


	app_('result',['tn', 'fp', 'fn', 'tp'])

	for i in range(len(mcm)):
		app_('result',[mcm[i][0][0], mcm[i][0][1], mcm[i][1][0], mcm[i][1][1]])


	app_('result',[])#space
	app_('result',['confusion_matrix'])

	for i in range(len(mat)):
		mat2 = []
		for j in range(len(mat)):
			mat2.append(mat[i][j])
		app_('result',mat2)


	app_('result',[])#space
	app_('result',['TPR','PPV','F1', 'ACC'])

	acc_all = []
	for i in range(len(rec)):
		acc = ((mcm[i][0][0]+mcm[i][1][1])/(mcm[i][0][0] + mcm[i][0][1] + mcm[i][1][0] + mcm[i][1][1]))*100
		
		# check zero division
		sum_ofbot = mcm[i][0][0] + mcm[i][0][1] + mcm[i][1][0] + mcm[i][1][1]
		sum_oftop = mcm[i][0][0] + mcm[i][1][1]
		
		if sum_ofbot == 0 or sum_oftop == 0:
			app_('result',[rec[i],pre[i],f1[i],0])
			acc_all.append(0)

		else:
			app_('result',[rec[i],pre[i],f1[i],acc])
			acc_all.append(acc)


	app_('result',[np.mean(rec),np.mean(pre),np.mean(f1),np.mean(acc_all),'average'])

	print('DONE.....')



