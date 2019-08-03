
#System stuff
import sys
from itertools import cycle
import os
#Basics
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats, interp
import matplotlib.pyplot as plt
from numpy import linalg as LA
#sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, decomposition, datasets
from skrvm import RVR
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
#Other machinea learing libraries
from xgboost import XGBRegressor, plot_importance



plt.rcParams.update({'font.size': 30})

import json
from pprint import pprint
from os import listdir

reusingModels = False

def R(X1,X2):
    return np.sqrt((float(X1[1])-float(X2[1]))**2+(float(X1[2])-float(X2[2]))**2+(float(X1[3])-float(X2[3]))**2)

Z = {'H':1,'C':6,'N':7,'O':8,'F':9}
Coulomb={}
Output={}
data_path='dsgdb9nsd.xyz/'
num_atoms = 12
property_index = 8
list=listdir(data_path)
count=0
for l in list:
    if count < 4000:
        f = open(data_path+l, "r")
        size=f.readline()
        if int(size) == num_atoms:
            count += 1
            try:
                print(str(count) + ": " + l)
                properties=f.readline()
                properties=properties.split('\t')
                xyz=np.empty([0,5])
                for i in range(int(size)):
                    c=f.readline()
                    c=[c.split('\t')]
                    xyz=np.append(xyz,c,axis=0)
                littleCoulomb=np.empty([num_atoms,num_atoms])
                for x in range(num_atoms):
                    for y in range(num_atoms):
                        if x == y:
                            littleCoulomb[x][y]=Z[xyz[x][0]]**(2.4)/2
                        else:
                            littleCoulomb[x][y]=Z[xyz[x][0]]*Z[xyz[y][0]]/R(xyz[x],xyz[y])
                #Get the property desired
                property=float(properties[property_index])
                #Get the eigenvalues
                Coulomb[l],v = LA.eig(littleCoulomb)
                #Sort the numbers from smallest to largest
                Coulomb[l] = np.sort(Coulomb[l])
                Output[l]=[property]

            except:
                continue

Coulomb_df=pd.DataFrame(Coulomb).transpose()
Output_df=pd.DataFrame(Output).transpose()*10

#Create a new StandardScaler scaler object
scaler = StandardScaler().fit(Coulomb_df)

#Scale the entire input dataset
Coulomb_df = scaler.transform(Coulomb_df)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        Coulomb_df, Output_df, test_size=.2, random_state=None)

reports_df = pd.DataFrame(
    columns=['Name', 'MARE', 'MSE', 'R2'])

for regr_choice in range(5):

    regr_names = ['RF', 'SVM', 'RVM', 'Huber',
                  'XGBOOST']
    regr_objects = [RandomForestRegressor(n_estimators=400, max_depth=1000, random_state=0),
                    svm.SVR(kernel='rbf', epsilon=0.1, verbose=True),
                    RVR(kernel='rbf', n_iter=10000, tol=0.0001, verbose=True),
                    linear_model.HuberRegressor(
                        epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05),
                    XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                 max_depth=400, alpha=10, n_estimators=400)
                    ]

    regr = regr_objects[regr_choice]
    regr_name = regr_names[regr_choice]

    if reusingModels:
        regr = joblib.load('SavedModels_'+regr_name+'.pkl')
    else:
        regr.fit(X_train_scaled, y_train)

    if 'XGB' in regr_name:
        X_scaled_df_XGB = X_test_scaled  # .as_matrix()
        y_predicted = regr.predict(X_scaled_df_XGB)
    else:
        y_predicted = regr.predict(X_test_scaled)

    y_predicted = pd.DataFrame(y_predicted)

    MARE = 0
    for i in range(len(y_predicted)):
        MARE += abs((y_predicted.iloc[i][0] -
                     y_test.iloc[i][0])/y_test.iloc[i][0]*100)
    print("MARE=", MARE/len(y_predicted))

    print(mean_squared_error(y_test, y_predicted))
    print(r2_score(y_test, y_predicted))


    errors_file = open(regr_name+'_Test_Analysis.txt', 'w')
    errors_file.write('MARE\t'+str(MARE/len(y_predicted))+'\n')
    errors_file.write(
        'MSE\t'+str(mean_squared_error(y_test, y_predicted))+'\n')
    errors_file.write('r2\t'+str(r2_score(y_test, y_predicted))+'\n')
    errors_file.close()

    reports_df_row = pd.DataFrame(
        columns=['Name', 'MARE', 'MSE', 'R2'])
    reports_df_row.set_value(0, 'Name', regr_name+'_Test')
    reports_df_row.set_value(0, 'MARE', MARE/len(y_predicted))
    reports_df_row.set_value(0, 'MSE', np.sqrt(
        mean_squared_error(y_test, y_predicted)))
    reports_df_row.set_value(0, 'R2', r2_score(y_test, y_predicted))
    reports_df = reports_df.append(reports_df_row)

    xPlot = y_test
    yPlot = y_predicted
    plt.figure(figsize=(10, 10))
    plt.plot(xPlot, yPlot, 'ro')
    plt.plot(xPlot, xPlot)
    plt.ylabel(regr_name)
    plt.xlabel('DFT')
    plt.savefig('Figs_'+regr_name+'_Correlation_Test', bbox_inches='tight')

    if 'XGB' in regr_name:
        X_scaled_df_XGB = X_train_scaled
        y_predicted = regr.predict(X_scaled_df_XGB)
    else:
        y_predicted = regr.predict(X_train_scaled)

    y_predicted = pd.DataFrame(y_predicted)

    MARE = 0
    for i in range(len(y_train)):
        MARE += abs((y_predicted.iloc[i][0] -
                     y_train.iloc[i][0])/y_train.iloc[i][0]*100)
    print("MARE=", MARE/len(y_predicted))

    print(mean_squared_error(y_train, y_predicted))
    print(MARE/len(y_predicted))
    print(r2_score(y_train, y_predicted))

    errors_file = open(regr_name+'_Train_Analysis.txt', 'w')
    errors_file.write('MARE\t'+str(MARE/len(y_predicted))+'\n')
    errors_file.write(
        'MSE\t'+str(mean_squared_error(y_train, y_predicted))+'\n')
    errors_file.write('r2\t'+str(r2_score(y_train, y_predicted))+'\n')
    errors_file.close()

    reports_df_row = pd.DataFrame(
        columns=['Name', 'MARE', 'MSE', 'R2'])
    reports_df_row.set_value(0, 'Name', regr_name+'_Train')
    reports_df_row.set_value(0, 'MARE', MARE/len(y_predicted))
    reports_df_row.set_value(
        0, 'MSE', mean_squared_error(y_train, y_predicted))
    reports_df_row.set_value(0, 'R2', r2_score(y_train, y_predicted))
    reports_df = reports_df.append(reports_df_row)

    xPlot = y_train
    yPlot = y_predicted
    plt.figure(figsize=(10, 10))
    plt.plot(xPlot, yPlot, 'ro')
    plt.plot(xPlot, xPlot)
    plt.ylabel(regr_name)
    plt.xlabel('DFT')
    plt.savefig('Figs_'+regr_name+'_Correlation_Train', bbox_inches='tight')

    from sklearn.externals import joblib
    joblib.dump(regr, 'SavedModels_'+regr_name+'.pkl')
