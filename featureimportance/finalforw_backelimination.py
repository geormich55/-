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
#dataset = pd.read_excel(r'C:\Users\Micha\Desktop\python\DATA.xlsx', sheet_name='Fullo1')
dataset = pd.read_excel(r'DATA.xlsx', sheet_name='Fullo1')
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
X=X1
X3 = X1.iloc[:, :].values
#X = pd.DataFrame(X3, columns=['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel'])
Y = Y1.loc[:, :].values
y = Y.tolist()


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import statsmodels.api as sm

#data = load_boston()
#X = pd.DataFrame(data.data, columns=data.feature_names)
#y = data.target

"""X_opt variable has all the columns of independent variables of matrix X 
in this case we have 5 independent variables"""
X_opt = np.array(X)[:,[0,1,2,3,4,5,6]]

a=np.array(X)[:,[0,1,2,3,4,5,6]]

b=pd.DataFrame(a)

c=sm.add_constant(pd.DataFrame(X_opt))




initial_list=[]
threshold_in=0.01
threshold_out = 0.05
""" Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
included = list(initial_list)
pvalincluded=[]
while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            b=sm.add_constant(pd.DataFrame(X[included+[new_column]]))
            model = sm.OLS(y, b.astype(float)).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
             best_feature = new_pval.index[new_pval.argmin()]
             included.append(best_feature)
             pvalincluded.append(best_pval)
             changed=True
             if True:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))


        # backward step
        a=sm.add_constant(pd.DataFrame(X[included]))
        model = sm.OLS(y, a.astype(float)).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.index[pvalues.argmax()]
            included.remove(worst_feature)
            if True:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break

print('resulting features:')
print(included)
print(('included p-values'))
print(pvalincluded)
print('excluded features:')
print(new_pval)


import matplotlib.pyplot as plt
x = range(3)
x_labels=included
y=pvalincluded
plt.bar(x,y,color='green',align='center')
plt.title('pval for 3 features')
plt.xticks(x,x_labels,rotation='vertical')
plt.show()



#initial_list=[]
#threshold_in=0.9
#threshold_out = 0.9
""" Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
#included = list(initial_list)
#pvalincluded = []
#while True:
        #changed=False
        # forward step
        #excluded = list(set(X.columns)-set(included))
        #new_pval = pd.Series(index=excluded)
        #for new_column in excluded:
            #b=sm.add_constant(pd.DataFrame(X[included+[new_column]]))
            #model = sm.OLS(y, b.astype(float)).fit()
            #new_pval[new_column] = model.pvalues[new_column]
        #best_pval = new_pval.min()
        #if best_pval < threshold_in:
             #best_feature = new_pval.index[new_pval.argmin()]
             #included.append(best_feature)
             #pvalincluded.append(best_pval)
             #changed=True
             #if True:
                #print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))


        # backward step
        #a=sm.add_constant(pd.DataFrame(X[included]))
        #model = sm.OLS(y, a.astype(float)).fit()
        # use all coefs except intercept
        #pvalues = model.pvalues.iloc[1:]
        #worst_pval = pvalues.max() # null if pvalues is empty
        #if worst_pval > threshold_out:
            #changed=True
            #worst_feature = pvalues.index[pvalues.argmax()]
            #included.remove(worst_feature)
            #if True:
                #print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        #if not changed:
            #break

#print('resulting features:')
#print(included)
#print(('included p-values'))
#print(pvalincluded)
#print('excluded features and p-values:')
#print(new_pval)

#import matplotlib.pyplot as plt
#x = range(7)
#x_labels=included
#y=pvalincluded
#plt.bar(x,y,color='green',align='center')
#plt.title('pval for 7 features')
#plt.xticks(x,x_labels,rotation='vertical')
#plt.show()



for i in range(7):
  X2= np.append(arr = np.ones((1411,1)).astype(int), values = X, axis=1)
  if i==0:
     X_opt=X2[:,[3]]
     Xp=np.array(X_opt, dtype=float)
     Yp=np.array(Y, dtype=float)

     """Running the OLS method on X_opt and storing results in regressor_OLS"""
     regressor_OLS=sm.OLS(endog = Yp, exog = Xp).fit()
     regressor_OLS.summary()
     print(regressor_OLS.summary())

     rsquared=sm.OLS(endog = Yp, exog = Xp).fit().rsquared_adj

     print(rsquared)
  if i == 1:
      X_opt = X2[:, [1,3]]
      Xp = np.array(X_opt, dtype=float)
      Yp = np.array(Y, dtype=float)

      """Running the OLS method on X_opt and storing results in regressor_OLS"""
      regressor_OLS = sm.OLS(endog=Yp, exog=Xp).fit()
      regressor_OLS.summary()
      print(regressor_OLS.summary())

      rsquared = sm.OLS(endog=Yp, exog=Xp).fit().rsquared_adj

      print(rsquared)
  if i == 2:
      X_opt = X2[:, [1,3,7]]
      Xp = np.array(X_opt, dtype=float)
      Yp = np.array(Y, dtype=float)

      """Running the OLS method on X_opt and storing results in regressor_OLS"""
      regressor_OLS = sm.OLS(endog=Yp, exog=Xp).fit()
      regressor_OLS.summary()
      print(regressor_OLS.summary())

      rsquared = sm.OLS(endog=Yp, exog=Xp).fit().rsquared_adj

      print(rsquared)
  if i == 3:
      X_opt = X2[:, [1,3,4,5,7]]
      Xp = np.array(X_opt, dtype=float)
      Yp = np.array(Y, dtype=float)

      """Running the OLS method on X_opt and storing results in regressor_OLS"""
      regressor_OLS = sm.OLS(endog=Yp, exog=Xp).fit()
      regressor_OLS.summary()
      print(regressor_OLS.summary())

      rsquared = sm.OLS(endog=Yp, exog=Xp).fit().rsquared_adj

      print(rsquared)

import matplotlib.pyplot as plt
x = range(5)
x_labels=Xp
y=Yp
plt.bar(x,y,color='green',align='center')
plt.title('rsquared for 5 features')
plt.xticks(x,x_labels,rotation='vertical')
plt.show()


#scaler = StandardScaler()
#scaled_Y = scaler.fit_transform(Y)
#scaled_X = scaler.fit_transform(X1)
#scx=pd.DataFrame(scaled_X,columns=X1.columns)



