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
X4= pd.DataFrame(datas, columns=['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel', 'HeatRate'])
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



import statsmodels.formula.api as smf
#import statsmodels.api as sm


"""Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
#remaining = set(X.columns)
#response='HeatRate'
#remaining.remove(response)
#selected = []
#scores = []
#current_score, best_new_score = 0.0, 0.0
#i=0
#while i!=35:
        #i=i+1
        #scores_with_candidates = []
        #for candidate in remaining:
            #formula = "{} ~ {} + 1".format(response,' + '.join(selected + [candidate]))
            #X2 = np.append(arr=np.ones((1411, 1)).astype(int), values=X[selected + [candidate]], axis=1)
            #Xp = np.array(X2, dtype=float)
            #Yp = np.array(Y, dtype=float)
            #score = sm.OLS(endog = Yp, exog = Xp).fit().rsquared_adj
            #scores_with_candidates.append((score, candidate))
        #scores_with_candidates.sort()
        #best_new_score, best_candidate = scores_with_candidates.pop()
        #if current_score < best_new_score:
            #remaining.remove(best_candidate)
            #selected.append(best_candidate)
            #scores.append(best_new_score)
            #current_score = best_new_score
            #print('Add  {:30} with rsquared-value {:.6}'.format(best_candidate, best_new_score))
        #else:
            #remaining.remove(best_candidate)
            #selected.append(best_candidate)
            #scores.append(best_new_score)
            #current_score = best_new_score
            #print('Add but not better {:30} with rsquared-value {:.6}'.format(best_candidate, best_new_score))

#formula = "{} ~ {} + 1".format(response,' + '.join(selected))
#X2 = np.append(arr=np.ones((1411, 1)).astype(int), values=X[selected], axis=1)
#Xp = np.array(X2, dtype=float)
#Yp = np.array(Y, dtype=float)
#model = sm.OLS(endog = Yp, exog = Xp).fit()


#a=np.array(datas)[:,[0,1,2,3,4,5,6,7]]

#b=pd.DataFrame(a)

#h=pd.to_numeric(datas['HeatRate'])

#print(selected)
#print(scores)

#model.rsquared_adj

#model.model._formula_max_endog

#model.summary()


import statsmodels.formula.api as smf
import statsmodels.api as sm


"""Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
remaining = set(X.columns)
response='HeatRate'
#remaining.remove(response)
selected = []
scores = []
current_score, best_new_score = 0.0, 0.0
i=0
while i!=35:
        i=i+1
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,' + '.join(selected + [candidate]))
            X2 = np.append(arr=np.ones((1411, 1)).astype(int), values=X[selected + [candidate]], axis=1)
            Xp = np.array(X2, dtype=float)
            Yp = np.array(Y, dtype=float)
            score = sm.OLS(endog = Yp, exog = Xp).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            scores.append(best_new_score)
            current_score = best_new_score
            print('Add  {:30} with rsquared-value {:.6}'.format(best_candidate, best_new_score))
        else:
            #remaining.remove(best_candidate)
            #selected.append(best_candidate)
            #scores.append(best_new_score)
            #current_score = best_new_score
            print('Add but not better {:30} with rsquared-value {:.6}'.format(best_candidate, best_new_score))

formula = "{} ~ {} + 1".format(response,' + '.join(selected))
X2 = np.append(arr=np.ones((1411, 1)).astype(int), values=X[selected], axis=1)
Xp = np.array(X2, dtype=float)
Yp = np.array(Y, dtype=float)
model = sm.OLS(endog = Yp, exog = Xp).fit()


a=np.array(datas)[:,[0,1,2,3,4,5,6,7]]

b=pd.DataFrame(a)

h=pd.to_numeric(datas['HeatRate'])

print(selected)
print(scores)

model.rsquared_adj

model.model._formula_max_endog

model.summary()

import matplotlib.pyplot as plt
x = range(np.shape(selected)[0])
x_labels=selected
y=scores
plt.bar(x,y,color='green',align='center')
plt.title('rsquared for 1 2 or more features')
plt.xticks(x,x_labels,rotation='vertical')
plt.show()

for i in range(np.shape(selected)[0]):
    xTitle = selected[i]
    yTitle = 'HeatRate'
    title = ''
    title_font = {'fontname': 'Arial', 'size': '30', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '18'}
    # create basic scatterplot
    plt.plot(X[selected[i]], Y, 'o')
    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(X[selected[i]].astype(float), Y.astype(float), 1)
    # add linear regression line to scatterplot
    plt.plot(X[selected[i]], m * X[selected[i]] + b)
    # use green as color for individual points
    plt.plot(X[selected[i]], Y, 'o', color='green')
    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(X[selected[i]].astype(float), Y.astype(float), 1)
    # use red as color for regression line
    plt.plot(X[selected[i]], m * X[selected[i]] + b, color='red')
    plt.xlabel(xTitle, **axis_font)
    plt.ylabel(yTitle, **axis_font)
    plt.title(title, **title_font)
    plt.show()

X3=X.sort_values(by=selected[0])
for i in range(np.shape(selected)[0]):
    xTitle = selected[i]
    yTitle = 'HeatRate'
    title = ''
    title_font = {'fontname': 'Arial', 'size': '30', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '18'}
    X1=np.array(X3[selected[i]])
    X2=X1.reshape(-1,1)
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X2)
    model = LinearRegression().fit(x_, Y)
    y_pred = model.predict(x_)
    plt.plot(X3[selected[i]], Y, "bo", markersize=2)
    plt.plot(X3[selected[i]], y_pred, "r-", markersize=2)
    plt.xlabel(xTitle, **axis_font)
    plt.ylabel(yTitle, **axis_font)
    plt.title(title, **title_font)
    #m1, m2, b = np.polyfit(X[selected[i]].astype(float), Y.astype(float), 2)
    #plt.plot(X[selected[i]], m1 * X[selected[i]] +m2 * X[selected[i]] + b , "r-", markersize=2)
    #plt.plot(X[selected[i]], x_, "r-", markersize=2)  # p(X) evaluates the polynomial at X
    plt.show()

