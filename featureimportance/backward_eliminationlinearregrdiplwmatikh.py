#1 Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# permutation feature importance with knn for regression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

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
          if abs((((X2.values[i, 44]) / ((X2.values[i, 37] * X2.values[i, 38]) / 3600)) - (1 / (X2.values[i, 45] / 3600)))) >= 0.05:
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

# Make standardization
scaler = StandardScaler()
scaled_Y = scaler.fit_transform(Y)
scaled_X = scaler.fit_transform(X)
#3 Encoding the categorical variables:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#5 Splitting the dataset into the Training and Test dataset
#train_set_split: Split arrays or matrices into random train and #test subsets. %20 of the dataset to the test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 6 Fit multiple Linear Regression model to our Train set
from sklearn.linear_model import LinearRegression

# Fit the linear regression model to the training set… We use the fit #method the arguments of the fit method will be training sets
model = LinearRegression().fit(X_train, Y_train)

#Get Results
r_sq = model.score(X_train, Y_train)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
#Predict response
y_pred = model.predict(X_test)
print('predicted response:', y_pred, sep='\n')
prosarmogh=(Y_test-y_pred)/np.std(Y_test-y_pred)
print('predicted adjustment:', prosarmogh, sep='\n')

#for i,v in enumerate(prosarmogh):
	#print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
#pyplot.bar([x for x in range(len(prosarmogh))], prosarmogh)
#pyplot.show()

np.std(X_train)
scaled_coefficients=model.coef_/np.std(X_train)

coefficients=np.resize(model.coef_,(7,))

# summarize feature importance
for i,v in enumerate(coefficients):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(coefficients))], coefficients)
pyplot.show()

coefficients2=np.resize(scaled_coefficients,(7,))

# summarize feature importance
for i,v in enumerate(coefficients2):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(coefficients2))], coefficients2)
pyplot.show()


scaled_Xtrain=X_train/np.std(X_train)

scaled_Xtest=X_test/np.std(X_test)

# Fit the linear regression model to the training set… We use the fit #method the arguments of the fit method will be training sets
model = LinearRegression().fit(scaled_Xtrain, Y_train)

#Get Results
r_sq = model.score(scaled_Xtrain, Y_train)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
#Predict response
y_pred = model.predict(scaled_Xtest)
print('predicted response:', y_pred, sep='\n')

coefficients3=np.resize(model.coef_,(7,))

# summarize feature importance
for i,v in enumerate(coefficients3):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(coefficients3))], coefficients3)
pyplot.show()

#Predict response with another way
#d = model.coef_.dot(np.transpose(X_test))
#y_pred1 = model.intercept_ + np.sum(d.astype('float64'), axis=1)
#print('predicted response:', y_pred1, sep='\n')
# Beta0 has x0=1. Add a column of for the the first term of the #MultiLinear Regression equation.
import statsmodels.formula.api as sm
import statsmodels.api as sm
import numpy as np
#The 0th column contains only 1 in each 1411 rows
X2= np.append(arr = np.ones((1411,1)).astype(int), values = X, axis=1)
X_opt= X2[:, [1,2,3,4,5,6]]
#Optimal X contains the highly impacted independent variables
#OLS: Oridnary Least Square Class. endog is the dependent variable, #exog is the number of observations
Xp=np.array(X_opt, dtype=float)
Yp=np.array(Y, dtype=float)
regressor_OLS=sm.OLS(endog = Yp, exog = Xp).fit()
regressor_OLS.summary()