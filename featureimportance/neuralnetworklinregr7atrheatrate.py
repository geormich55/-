#1 Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
seed=1

#2 Importing the dataset:
dataset = pd.read_excel(r'C:\Users\Micha\Desktop\python\DATA.xlsx', sheet_name='Fullo1')
data = dataset[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7']]
data.columns = ['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel', 'HeatRate']
datas=data.loc[3:2908, :]
datase=dataset.values[3:2908, :]
Xdatas=datas[['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel']]
Ydatas=datas[['HeatRate']]
X1 = pd.DataFrame(datas, columns=['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel'])
Y1 = pd.DataFrame(datas, columns=['HeatRate'])
#2 Importing a second dataset in order to test data's validity:
dat = pd.read_excel(r'C:\Users\Micha\Desktop\Diplwmatikh ELPE\Σχέσεις\Dataslevhtas.xlsx', sheet_name='Fullo1')
da = dat[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47']]
da.columns = ['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'Gasfuel', 'LHV', 'GasTurbine', 'SelfConsumption', 'Gasflow', 'Gastempinputcondenser', 'Gastempoutputcondenser', 'PlantMW', 'HeatRate', 'HRSGefficiency', 'Condenserefficiency']
datasets=da.loc[3:2908, :]
datasetse=dat.values[3:2908, :]
X2 = pd.DataFrame(datasets, columns=['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'Gasfuel', 'LHV', 'GasTurbine', 'SelfConsumption', 'Gasflow', 'Gastempinputcondenser', 'Gastempoutputcondenser', 'PlantMW', 'HeatRate', 'HRSGefficiency', 'Condenserefficiency'])
I=[]
for i in range(2905):
      n = 0
      if Y1.values[i,0]<4000:
          Y1.values[i,0]=0
          I.append(i)
          n=1
      elif Y1.values[i,0]>10000:
          Y1.values[i,0]=0
          I.append(i)
          n=1
      if ((X2.values[i, 37] * X2.values[i, 38]) / 3600) != 0 and (X2.values[i, 45] / 3600) != 0:
          if abs((((X2.values[i,44])/((X2.values[i,37]*X2.values[i,38])/3600))-(1/(X2.values[i,45]/3600))))>=0.05:
              if n==0:
                  Y1.values[i, 0] = 0
                  I.append(i)

Y1=Y1[Y1!=0]
Y1=Y1.dropna()
a=np.shape(I)[0]
for i in range(2905):
     for j in range(7):
         for k in range(a):
             if i == I[k]:
                 X1.values[i,j]=0
X1=X1[X1!=0]
X1=X1.dropna()
X = X1.loc[:, :].values
Y = Y1.loc[:, :].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1)

# define the model
def larger_model():
# create model
	model = Sequential()
	model.add(Dense(13, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train.astype(float), y_train.astype(float) ,cv=kfold)
estimator = KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=0)
results2 = cross_val_score(estimator, X_train.astype(float), y_train.astype(float), cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Baseline: %.2f (%.2f) MSE" % (results2.mean(), results2.std()))

scaler = StandardScaler()
scaled_Y = scaler.fit_transform(Y)
scaler.mean_
scaler1 = StandardScaler()
scaled_X = scaler1.fit_transform(X1)
scaler1.mean_
scx=pd.DataFrame(scaled_X,columns=X1.columns)
scy=pd.DataFrame(scaled_Y,columns=Y1.columns)

estimator.fit(X.astype(float), Y.astype(float))
prediction = estimator.predict(X_test.astype(float))
#accuracy_score(y_test.astype(float), prediction)
#print("accuracy is: %.2f" % accuracy_score)
print("prediction is: %.2f" % prediction)

pipeline.fit(X.astype(float), Y.astype(float))
prediction2 = pipeline.predict(X_test.astype(float))
#accuracy_score(y_test.astype(float), prediction2)
#print("accuracy is: %.2f" % accuracy_score)
print("prediction is: %.2f" % prediction2)

y_error = y_test.astype(float) - prediction2.astype(float)

import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

p=prediction2.reshape(-1,1)
print("R2 score : %.2f" % r2_score(y_test, p))
print("Mean squared error: %.2f" % mean_squared_error(y_test, p))

er = []
g = 0
for i in range(len(y_test)):
     print("actual=", y_test[i], " observed=", p[i])
     x = (y_test[i] - p[i]) ** 2
     er.append(x)
     g = g + x

x = 0
for i in range(len(er)):
     x = x + er[i]

print("MSE", x / len(er))

v = np.var(er)
print("variance", v)

print("average of errors ", np.mean(er))

m = np.mean(y_test)
print("average of observed values", m)

y = 0
for i in range(len(y_test)):
    y = y + ((y_test[i] - m) ** 2)

print("total sum of squares", y)
print("ẗotal sum of residuals ", g)
print("r2 calculated", 1 - (g / y))

from sklearn.metrics import r2_score
r2_score(y_test.astype(float),prediction2.astype(float))


print("R2 score : %.2f" % r2_score(y_test, p))
print("Mean squared error: %.2f" % mean_squared_error(y_test, p))

import numpy as np
RSS = np.sum((prediction2.astype(float) - y_test.astype(float))**2)
y_mean = np.mean(y_test.astype(float))
TSS = np.sum((y_test.astype(float) - y_mean.astype(float))**2)
R2 = 1 - RSS/TSS
R2

n=X_test.shape[0]
p=X_test.shape[1] - 1

adj_rsquared = 1 - (1 - R2) * ((n - 1)/(n-p-1))
adj_rsquared

scaler = StandardScaler()
scaled_Y = scaler.fit_transform(Y)
scaler.mean_
scaler1 = StandardScaler()
scaled_X = scaler1.fit_transform(X1)
scaler1.mean_
scx=pd.DataFrame(scaled_X,columns=X1.columns)
scy=pd.DataFrame(scaled_Y,columns=Y1.columns)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(scaled_X,scaled_Y,test_size = 0.1)

estimator = KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=0)
results_scaled = cross_val_score(estimator, X_train.astype(float), y_train.astype(float), cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results_scaled.mean(), results_scaled.std()))

estimator.fit(scaled_X.astype(float), scaled_Y.astype(float))
prediction_scaled = estimator.predict(X_test.astype(float))
#accuracy_score(y_test.astype(float), prediction_scaled)
#print("accuracy is: %.2f" % accuracy_score)

prediction_original=scaler.inverse_transform(prediction_scaled)
print("prediction is: %.2f" % prediction_original)

#estimator = KerasRegressor(build_fn=larger_model, train_data =(X_train, y_train) ,validation_data = (X_test,y_test), epochs=100, batch_size=5, verbose=0)
#results3 = cross_val_score(estimator, X_test, y_test)
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=larger_model, train_data =(X_train, y_train) ,validation_data = (X_test,y_test), epochs=50, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
#results4 = cross_val_score(pipeline, X_test, y_test)
#print("Baseline: %.2f (%.2f) MSE" % (results3.mean(), results3.std()))
#print("Baseline: %.2f (%.2f) MSE" % (results4.mean(), results4.std()))

