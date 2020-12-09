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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

#Z=X1[[X1.columns[0]]]

#for i in range(27):
   #Z=pd.concat([Z,X1[[X1.columns[0]]]],axis=1)

Z=[]
Z=pd.DataFrame(Z)
#X3=X1
#h=X3[0:7:]
h=[]
for i in range(7):
    for j in range(7):
        h.append(0)
h=np.array(h)
h=h.reshape(7,7)
for i in range(7):
    for j in range(7):
            if i==j:
               Z=pd.concat([Z,pd.DataFrame(X1.iloc[:,i].values*X1.iloc[:,j].values)],axis=1)
            else:
                if h[i,j]==0 and h[j,i]==0:
                    Z=pd.concat([Z,pd.DataFrame(X1.iloc[:,i].values*X1.iloc[:,j].values)],axis=1)
                    h[i,j]=h[i,j]+1


U=[]
for i in range(1411):
      U.append(i)

s = pd.Series(U)
X1=X1.set_index([s])
Y1=Y1.set_index([s])

Z.columns=['LPStTemp^2', 'LPStTemp*AmbTemp', 'LPStTemp*CondFlow', 'LPStTemp*InletCond', 'LPStTemp*OutletCond', 'LPStTemp*CondTemp', 'LPStTemp*CondLevel', 'AmbTemp^2', 'AmbTemp*CondFlow', 'AmbTemp*InletCond', 'AmbTemp*OutletCond', 'AmbTemp*CondTemp', 'AmbTemp*CondLevel', 'CondFlow^2', 'CondFlow*InletCond', 'CondFlow*OutletCond', 'CondFlow*CondTemp', 'CondFlow*CondLevel', 'InletCond^2', 'InletCond*OutletCond', 'InletCond*CondTemp', 'InletCond*CondLevel', 'OutletCond^2', 'OutletCond*CondTemp', 'OutletCond*CondLevel', 'CondTemp^2', 'CondTemp*CondLevel', 'CondLevel^2']

X =pd.concat([X1,Z],axis=1)

#X = X.loc[:, :].values
Y = Y1.loc[:, :].values



X3 = X1.iloc[:, :].values
#X = pd.DataFrame(X3, columns=['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel'])
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
            pvalincluded.remove(worst_pval)
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
x = range(np.shape(included)[0])
x_labels=included
y=pvalincluded
plt.bar(x,y,color='green',align='center')
plt.title('pval for 9 features')
plt.xticks(x,x_labels,rotation='vertical')
plt.show()

for i in range(np.shape(included)[0]):
    xTitle = included[i]
    yTitle = 'HeatRate'
    title = ''
    title_font = {'fontname': 'Arial', 'size': '30', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '18'}
    # create basic scatterplot
    plt.plot(X[included[i]], Y, 'o')
    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(X[included[i]].astype(float), Y.astype(float), 1)
    # add linear regression line to scatterplot
    plt.plot(X[included[i]], m * X[included[i]] + b)
    # use green as color for individual points
    plt.plot(X[included[i]], Y, 'o', color='green')
    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(X[included[i]].astype(float), Y.astype(float), 1)
    # use red as color for regression line
    plt.plot(X[included[i]], m * X[included[i]] + b, color='red')
    plt.xlabel(xTitle, **axis_font)
    plt.ylabel(yTitle, **axis_font)
    plt.title(title, **title_font)
    plt.show()

X3=X.sort_values(by=included[0])
for i in range(np.shape(included)[0]):
    xTitle = included[i]
    yTitle = 'HeatRate'
    title = ''
    title_font = {'fontname': 'Arial', 'size': '30', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '18'}
    X1=np.array(X3[included[i]])
    X2=X1.reshape(-1,1)
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X2)
    model = LinearRegression().fit(x_, Y)
    y_pred = model.predict(x_)
    plt.plot(X3[included[i]], Y, "bo", markersize=2)
    plt.plot(X3[included[i]], y_pred, "r-", markersize=2)
    plt.xlabel(xTitle, **axis_font)
    plt.ylabel(yTitle, **axis_font)
    plt.title(title, **title_font)
    #m1, m2, b = np.polyfit(X[selected[i]].astype(float), Y.astype(float), 2)
    #plt.plot(X[selected[i]], m1 * X[selected[i]] +m2 * X[selected[i]] + b , "r-", markersize=2)
    #plt.plot(X[selected[i]], x_, "r-", markersize=2)  # p(X) evaluates the polynomial at X
    plt.show()



initial_list=[]
threshold_in=0.9
threshold_out = 0.9
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
pvalincluded = []
dropped = []
pvaldropped = []
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
            #included.remove(worst_feature)
            #pvalincluded.remove(worst_pval)
            if True:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                dropped.append(worst_feature)
                pvaldropped.append(worst_pval)
        if not changed:
            break



A=[]
for i in dropped:
    if i in included:
      A.append(included.index(i))
      included.remove(i)

for i in A:
       pvalincluded.remove(pvalincluded[i])


print('resulting features:')
print(included)
print(('included p-values'))
print(pvalincluded)
print('excluded features and p-values:')
print(new_pval)

import matplotlib.pyplot as plt
x = range(np.shape(included)[0])
x_labels=included
y=pvalincluded
plt.bar(x,y,color='green',align='center')
plt.title('pval for these features')
plt.xticks(x,x_labels,rotation='vertical')
plt.show()
