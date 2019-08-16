'''This code pertains to paper titled "Prediction of Left Ventricular Mechanics Using Machine Learning".
The paper is authored by Yaghoub Dabiri, Alex Van der Velden, Kevin L. Sack, Jenny S. Choy, Ghassan S. Kassab and Julius M. Guccione, 
and published at Frontiers in Physics, 2019.

Copyright (c) 2019 by University of California San Francisco. All rights reserved.

The codes are provided “as is”, and without any express or implied warranties, including, 
without limitation, the implied warranties of merchantability and fitness for a particular purpose.

Please cite the paper as follows:
Yaghoub Dabiri, Alex Van der Velden, Kevin L. Sack, Jenny S. Choy, Ghassan S. Kassab and Julius M. Guccione, 
Prediction of Left Ventricular Mechanics Using Machine Learning, Frontiers in Physics, 2019.'''

import xgboost as xgb
import pandas as pd
import numpy as np
import time

start_time = time.process_time()

seed = 7
test_size = 0.0

dataset = np.asarray(pd.read_csv('lvp2_data.csv', delimiter=','))
dataset_test = np.asarray(pd.read_csv('lvp2_test.csv', delimiter=','))

#############################################
#Run,LV_l0,LV_t0,LV_tmax,RV_l0,RV_t0,RV_tmax,t,V,P
x_train = dataset[:,1:8]
y_train = dataset[:,9]

number_of_test = 1
x_test = dataset_test[0+(number_of_test-1)*401:401*number_of_test,1:8]
y_test = dataset_test[0+(number_of_test-1)*401:401*number_of_test,9] 

model = xgb.XGBRegressor()

param_test1 = {
'n_jobs':[-1],
'learning_rate': [0.05],
'max_depth':[15],
'n_estimators': [500],
} 

gsearch = GridSearchCV(estimator = model,param_grid = param_test1,cv=3,verbose = 1,scoring='r2')
grid_fit = gsearch.fit(x_train,y_train)

with open('C:/Users/location of file', 'wb') as fp:
    pickle.dump(grid_fit, fp)
