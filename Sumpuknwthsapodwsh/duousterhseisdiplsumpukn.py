#1 Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 Importing the dataset:
dataset = pd.read_excel(r'C:\Users\Micha\Desktop\Diplwmatikh ELPE\Σχέσεις\DATAS.xlsx', sheet_name='Fullo1')
data = dataset[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7']]
data.columns = ['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel', 'Condefficiency']
datas=data.loc[3:2908, :]
datase=dataset.values[3:2908, :]
Xdatas=datas[['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel']]
Ydatas=datas[['Condefficiency']]
X1_ = pd.DataFrame(datas, columns=['LPStTemp', 'AmbTemp', 'CondFlow', 'InletCond', 'OutletCond', 'CondTemp', 'CondLevel', 'Condefficiency'])
#2 Importing a second dataset in order to test data's validity:
dat = pd.read_excel(r'C:\Users\Micha\Desktop\Diplwmatikh ELPE\Σχέσεις\Dataslevhtas.xlsx', sheet_name='Fullo1')
da = dat[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47']]
da.columns = ['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'Gasfuel', 'LHV', 'GasTurbine', 'SelfConsumption', 'Gasflow', 'Gastempinputcondenser', 'Gastempoutputcondenser', 'PlantMW', 'HeatRate', 'HRSGefficiency', 'Condenserefficiency']
datasets=da.loc[3:2908, :]
datasetse=dat.values[3:2908, :]
X2 = pd.DataFrame(datasets, columns=['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'Gasfuel', 'LHV', 'GasTurbine', 'SelfConsumption', 'Gasflow', 'Gastempinputcondenser', 'Gastempoutputcondenser', 'PlantMW', 'HeatRate', 'HRSGefficiency', 'Condenserefficiency'])
A = pd.DataFrame(datas, columns=['LPStTemp'])
A1 = pd.DataFrame(datas, columns=['LPStTemp'])
Au = pd.DataFrame(datas, columns=['LPStTemp'])
for i in range(2904):
           A.values[i+1,0]=A1.values[i,0]
           Au.values[i + 1, 0] = A.values[i, 0]
A1['LPStTemp(t-1)'] = A
A1['LPStTemp(t-2)'] = Au
B = pd.DataFrame(datas, columns=['AmbTemp'])
A2 = pd.DataFrame(datas, columns=['AmbTemp'])
Au2 = pd.DataFrame(datas, columns=['AmbTemp'])
for i in range(2904):
    Au2.values[i + 1, 0] = B.values[i, 0]
A1['AmbTemp'] = A2
A1['AmbTemp(t-1)'] = B
A1['AmbTemp(t-2)'] = Au2
C = pd.DataFrame(datas, columns=['CondFlow'])
A3 = pd.DataFrame(datas, columns=['CondFlow'])
Au3 = pd.DataFrame(datas, columns=['CondFlow'])
for i in range(2904):
           C.values[i+1,0]=A3.values[i,0]
           Au3.values[i + 1, 0] = C.values[i, 0]
A1['CondFlow'] = A3
A1['CondFlow(t-1)'] = C
A1['CondFlow(t-2)'] = Au3
D = pd.DataFrame(datas, columns=['InletCond'])
A4 = pd.DataFrame(datas, columns=['InletCond'])
Au4 = pd.DataFrame(datas, columns=['InletCond'])
for i in range(2904):
           D.values[i+1,0]=A4.values[i,0]
           Au4.values[i + 1, 0] = D.values[i, 0]
A1['InletCond'] = A4
A1['InletCond(t-1)'] = D
A1['InletCond(t-2)'] = Au4
E = pd.DataFrame(datas, columns=['OutletCond'])
A5 = pd.DataFrame(datas, columns=['OutletCond'])
Au5 = pd.DataFrame(datas, columns=['OutletCond'])
for i in range(2904):
           E.values[i+1,0]=A5.values[i,0]
           Au5.values[i + 1, 0] = E.values[i, 0]
A1['OutletCond'] = A5
A1['OutletCond(t-1)'] = E
A1['OutletCond(t-2)'] = Au5
F = pd.DataFrame(datas, columns=['CondTemp'])
A6 = pd.DataFrame(datas, columns=['CondTemp'])
Au6 = pd.DataFrame(datas, columns=['CondTemp'])
for i in range(2904):
           F.values[i+1,0]=A6.values[i,0]
           Au6.values[i + 1, 0] = F.values[i, 0]
A1['CondTemp'] = A6
A1['CondTemp(t-1)'] = F
A1['CondTemp(t-2)'] = Au6
G = pd.DataFrame(datas, columns=['CondLevel'])
A7 = pd.DataFrame(datas, columns=['CondLevel'])
Au7 = pd.DataFrame(datas, columns=['CondLevel'])
for i in range(2904):
           G.values[i+1,0]=A7.values[i,0]
           Au7.values[i + 1, 0] = G.values[i, 0]
A1['CondLevel'] = A7
A1['CondLevel(t-1)'] = G
A1['CondLevel(t-2)'] = Au7
H = pd.DataFrame(datas, columns=['Condefficiency'])
A8 = pd.DataFrame(datas, columns=['Condefficiency'])
Au8 = pd.DataFrame(datas, columns=['HeatRate'])
for i in range(2904):
           H.values[i+1,0]=A8.values[i,0]
           Au8.values[i + 1, 0] = H.values[i, 0]
A1['Condefficiency'] = A8
A1['Condefficiency(t-1)'] = H
A1['Condefficiency(t-2)'] = Au8
X1=A1
Y1 = pd.DataFrame(datas, columns=['Condefficiency'])
Y2 = pd.DataFrame(datas, columns=['Condefficiency'])
for i in range(2904):
           Y1.values[i,0]=Y2.values[i+1,0]
#Y1.values[2904,0]=0
#Y1=Y1[Y1!=0]
#Y1=Y1.dropna()
I=[]
for i in range(2905):
       n=0
       if Y1.values[i,0]<0.3 and i<2902 and Y1.values[i,0]!=0:
           Y1.values[i, 0] = 0
           I.append(i)
           n=1
           Y1.values[i+1,0]=0
           I.append(i+1)
           Y1.values[i+2,0]=0
           I.append(i+2)
           Y1.values[i + 3, 0] = 0
           I.append(i + 3)
       elif Y1.values[i,0]>0.575 and (i<2902) and Y1.values[i,0]!=0:
           Y1.values[i, 0] = 0
           I.append(i)
           n=1
           Y1.values[i+1,0]=0
           I.append(i+1)
           Y1.values[i+2,0]=0
           I.append(i+2)
           Y1.values[i + 3, 0] = 0
           I.append(i + 3)
       if n == 0 and i < 2902 and Y1.values[i,0]!=0:
         if ((X2.values[i, 37] * X2.values[i, 38]) / 3600) != 0 and (X2.values[i, 45] / 3600) != 0:
           if abs((((X2.values[i, 44]) / ((X2.values[i, 37] * X2.values[i, 38]) / 3600)) - (1 / (X2.values[i, 45] / 3600)))) >= 0.05:
                       Y1.values[i, 0] = 0
                       I.append(i)
                       Y1.values[i + 1, 0] = 0
                       I.append(i + 1)
                       Y1.values[i + 2, 0] = 0
                       I.append(i + 2)
                       Y1.values[i + 3, 0] = 0
                       I.append(i + 3)
Y1=Y1[Y1!=0]
Y1=Y1.dropna()
a=np.shape(I)[0]
for i in range(2905):
     for j in range(24):
         for k in range(a):
             if i == I[k]:
                 X1.values[i,j]=0
X1=X1[X1!=0]
X1=X1.dropna()
X = X1.loc[:, :].values
Y = Y1.loc[:, :].values
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
#Predict response with another way
#d = model.coef_.dot(np.transpose(X_test))
#y_pred1 = model.intercept_ + np.sum(d.astype('float64'), axis=1)
#print('predicted response:', y_pred1, sep='\n')

#Mean Squared Error
MSE = np.square(np.subtract(Y_test.astype('float64'), y_pred)).mean()