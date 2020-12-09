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
remaining = set(datas.columns)
response='HeatRate'
remaining.remove(response)
selected = []
scores = []
current_score, best_new_score = 0.0, 0.0
i=0
while i!=7:
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
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            scores.append(best_new_score)
            current_score = best_new_score
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
x = range(7)
x_labels=selected
y=scores
plt.bar(x,y,color='green',align='center')
plt.title('rsquared for 1 2 or more features')
plt.xticks(x,x_labels,rotation='vertical')
plt.show()