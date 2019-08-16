'''This code pertains to paper titled "Prediction of Left Ventricular Mechanics Using Machine Learning".
The paper is authored by Yaghoub Dabiri, Alex Van der Velden, Kevin L. Sack, Jenny S. Choy, Ghassan S. Kassab and Julius M. Guccione, 
and published at Frontiers in Physics, 2019.

Copyright (c) 2019 by University of California San Francisco. All rights reserved.

The codes are provided “as is”, and without any express or implied warranties, including, 
without limitation, the implied warranties of merchantability and fitness for a particular purpose.

Please cite the paper as follows:
Yaghoub Dabiri, Alex Van der Velden, Kevin L. Sack, Jenny S. Choy, Ghassan S. Kassab and Julius M. Guccione, 
Prediction of Left Ventricular Mechanics Using Machine Learning, Frontiers in Physics, 2019.'''

#This file is for S12 but it can be adjusted for S11 and S22 using parameters in Table 1 in the paper.

import xgboost as xgb
import pandas as pd
import numpy as np
import time
import math


start_time = time.process_time()

seed = 7
test_size = 0.0

dataset = np.asarray(pd.read_csv('stress_dataset_train.csv', delimiter=','))
dataset_test = np.asarray(pd.read_csv('stress_dataset_test.csv', delimiter=','))

dataset = np.append(dataset,np.zeros([len(dataset),1]),1)
dataset = np.append(dataset,np.zeros([len(dataset),1]),1)

dataset_test = np.append(dataset_test,np.zeros([len(dataset_test),1]),1)
dataset_test = np.append(dataset_test,np.zeros([len(dataset_test),1]),1)

counter = 0
for row in dataset:
	row[11] = row[5]
	row[12] = row[6]
	r = math.sqrt((row[5])**2 + (row[6])**2)
	teta = math.atan2(row[6],row[5])
	if (teta <0):
		teta = 2*math.pi + teta
	row[5] = r
	row[6] = teta

	row[8] = 1000 * row[8]
	row[9] = 1000 * row[9]
	row[10] = 1000 * row[10]
	counter = counter+1

counter2 = 0
for row2 in dataset_test:
	row2[11] = row2[5]
	row2[12] = row2[6]
	r = math.sqrt((row2[5])**2 + (row2[6])**2)
	teta = math.atan2(row2[6],row2[5])
	if (teta <0):
		teta = 2*math.pi + teta
	row2[5] = r
	row2[6] = teta

	row2[8] = 1000 * row2[8]
	row2[9] = 1000 * row2[9]
	row2[10] = 1000 * row2[10]
	counter2 = counter2+1


x_train = dataset[:,1:8]
y_train = dataset[:,10]

x_test = dataset[0:576,1:8]
y_test = dataset_test[0:576,10] 

model = xgb.XGBRegressor()

param_test1 = {
'n_jobs':[-1],
'learning_rate': [0.1],
'max_depth':[7],
'n_estimators': [1500],
} 

gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1,cv=3,verbose = 1,scoring='r2')

grid_fit = gsearch1.fit(x_train,y_train)

elapsed_time = time.process_time() - start_time

with open('C:/Users/location of file', 'wb') as fp:
    pickle.dump(grid_fit, fp)
