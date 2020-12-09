#1 Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 Importing the dataset:
#dataset = pd.read_excel(r'C:\Users\Micha\Desktop\Diplwmatikh ELPE\Σχέσεις\python kai MATLAB arxeia\Levhtasapodwsh\Dataslevhtas-nontotal.xlsx', sheet_name='Fullo1')
dataset = pd.read_excel(r'Dataslevhtas-nontotal.xlsx', sheet_name='Fullo1')
data = dataset[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38']]
data.columns = ['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'HRSGefficiency', 'Condenserefficiency']
datas=data.loc[3:2908, :]
datase=dataset.values[3:2908, :]
Xdatas=datas[['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine']]
Ydatas=datas[['HRSGefficiency', 'Condenserefficiency']]
X1 = pd.DataFrame(datas, columns=['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'HRSGefficiency', 'Condenserefficiency'])
#2 Importing a second dataset in order to test data's validity:
dat = pd.read_excel(r'C:\Users\Micha\Desktop\Diplwmatikh ELPE\Σχέσεις\Dataslevhtas.xlsx', sheet_name='Fullo1')
da = dat[['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47']]
da.columns = ['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'Gasfuel', 'LHV', 'GasTurbine', 'SelfConsumption', 'Gasflow', 'Gastempinputcondenser', 'Gastempoutputcondenser', 'PlantMW', 'HeatRate', 'HRSGefficiency', 'Condenserefficiency']
datasets=da.loc[3:2908, :]
datasetse=dat.values[3:2908, :]
X2 = pd.DataFrame(datasets, columns=['Condtemp', 'wattempafsump', 'watpresafsump', 'watflowafsump', 'wattempbefpreheat', 'wattempaftpreheat', 'wattempfwt', 'watpresfwt', 'watflowaftprehLP', 'wattempLP', 'watpresLP', 'tempaftsupheatLP', 'presaftsupheatLP', 'flowaftsupheatLP', 'presaftpumpMP', 'tempaftpumpMP', 'tempafteconMP', 'flowafteconMP', 'flowtogasthem', 'presbotMP', 'tempbotMP', 'tempgasaftsupheat', 'presgasaftsupheat', 'tempaftuni(IP+CRH)', 'tempaftuni(IP+CRH)andaftreheat', 'flowaftuni(IP+CRH)andaftreheat', 'flowaftpumpHP', 'presaftpumpHP', 'tempaftpumpHP', 'wattempafteconHP', 'tempbotHP', 'presbotHP', 'tempHPaftNo1supheatHP', 'tempHPaftNo1supheatHPandaftDesupheat', 'tempgasHPtoturbine', 'presgasHPtoturbine', 'flowgasHPtoturbine', 'Gasfuel', 'LHV', 'GasTurbine', 'SelfConsumption', 'Gasflow', 'Gastempinputcondenser', 'Gastempoutputcondenser', 'PlantMW', 'HeatRate', 'HRSGefficiency', 'Condenserefficiency'])

A = pd.DataFrame(datas, columns=['Condtemp'])
A1 = pd.DataFrame(datas, columns=['Condtemp'])
for i in range(2904):
           A.values[i+1,0]=A1.values[i,0]
A1['Condtemp(t-1)'] = A
B = pd.DataFrame(datas, columns=['wattempafsump'])
A2 = pd.DataFrame(datas, columns=['wattempafsump'])
for i in range(2904):
           B.values[i+1,0]=A2.values[i,0]
A1['wattempafsump'] = A2
A1['wattempafsump(t-1)'] = B
C = pd.DataFrame(datas, columns=['watpresafsump'])
A3 = pd.DataFrame(datas, columns=['watpresafsump'])
for i in range(2904):
           C.values[i+1,0]=A3.values[i,0]
A1['watpresafsump'] = A3
A1['watpresafsump(t-1)'] = C
D = pd.DataFrame(datas, columns=['watflowafsump'])
A4 = pd.DataFrame(datas, columns=['watflowafsump'])
for i in range(2904):
           D.values[i+1,0]=A4.values[i,0]
A1['watflowafsump'] = A4
A1['watflowafsump(t-1)'] = D
E = pd.DataFrame(datas, columns=['wattempbefpreheat'])
A5 = pd.DataFrame(datas, columns=['wattempbefpreheat'])
for i in range(2904):
           E.values[i+1,0]=A5.values[i,0]
A1['wattempbefpreheat'] = A5
A1['wattempbefpreheat(t-1)'] = E
F = pd.DataFrame(datas, columns=['wattempaftpreheat'])
A6 = pd.DataFrame(datas, columns=['wattempaftpreheat'])
for i in range(2904):
           F.values[i+1,0]=A6.values[i,0]
A1['wattempaftpreheat'] = A6
A1['wattempaftpreheat(t-1)'] = F
G = pd.DataFrame(datas, columns=['wattempfwt'])
A7 = pd.DataFrame(datas, columns=['wattempfwt'])
for i in range(2904):
           G.values[i+1,0]=A7.values[i,0]
A1['wattempfwt'] = A7
A1['wattempfwt(t-1)'] = G
H1 = pd.DataFrame(datas, columns=['watpresfwt'])
A8 = pd.DataFrame(datas, columns=['watpresfwt'])
for i in range(2904):
           H1.values[i+1,0]=A8.values[i,0]
A1['watpresfwt'] = A8
A1['watpresfwt(t-1)'] = H1
H2 = pd.DataFrame(datas, columns=['watflowaftprehLP'])
A9 = pd.DataFrame(datas, columns=['watflowaftprehLP'])
for i in range(2904):
           H2.values[i+1,0]=A9.values[i,0]
A1['watflowaftprehLP'] = A9
A1['watflowaftprehLP(t-1)'] = H2
H3 = pd.DataFrame(datas, columns=['wattempLP'])
A10 = pd.DataFrame(datas, columns=['wattempLP'])
for i in range(2904):
           H3.values[i+1,0]=A10.values[i,0]
A1['wattempLP'] = A10
A1['wattempLP(t-1)'] = H3
H4 = pd.DataFrame(datas, columns=['watpresLP'])
A11 = pd.DataFrame(datas, columns=['watpresLP'])
for i in range(2904):
           H4.values[i+1,0]=A11.values[i,0]
A1['watpresLP'] = A11
A1['watpresLP(t-1)'] = H4
H5 = pd.DataFrame(datas, columns=['tempaftsupheatLP'])
A12 = pd.DataFrame(datas, columns=['tempaftsupheatLP'])
for i in range(2904):
           H5.values[i+1,0]=A12.values[i,0]
A1['tempaftsupheatLP'] = A12
A1['tempaftsupheatLP(t-1)'] = H5
H6 = pd.DataFrame(datas, columns=['presaftsupheatLP'])
A13 = pd.DataFrame(datas, columns=['presaftsupheatLP'])
for i in range(2904):
           H6.values[i+1,0]=A13.values[i,0]
A1['presaftsupheatLP'] = A13
A1['presaftsupheatLP(t-1)'] = H6
H7 = pd.DataFrame(datas, columns=['flowaftsupheatLP'])
A14 = pd.DataFrame(datas, columns=['flowaftsupheatLP'])
for i in range(2904):
           H7.values[i+1,0]=A14.values[i,0]
A1['flowaftsupheatLP'] = A14
A1['flowaftsupheatLP(t-1)'] = H7
H8 = pd.DataFrame(datas, columns=['presaftpumpMP'])
A15 = pd.DataFrame(datas, columns=['presaftpumpMP'])
for i in range(2904):
           H8.values[i+1,0]=A15.values[i,0]
A1['presaftpumpMP'] = A15
A1['presaftpumpMP(t-1)'] = H8
H9 = pd.DataFrame(datas, columns=['tempaftpumpMP'])
A16 = pd.DataFrame(datas, columns=['tempaftpumpMP'])
for i in range(2904):
           H9.values[i+1,0]=A16.values[i,0]
A1['tempaftpumpMP'] = A16
A1['tempaftpumpMP(t-1)'] = H9
H10 = pd.DataFrame(datas, columns=['tempafteconMP'])
A17 = pd.DataFrame(datas, columns=['tempafteconMP'])
for i in range(2904):
           H10.values[i+1,0]=A17.values[i,0]
A1['tempafteconMP'] = A17
A1['tempafteconMP(t-1)'] = H10
H11 = pd.DataFrame(datas, columns=['flowafteconMP'])
A18 = pd.DataFrame(datas, columns=['flowafteconMP'])
for i in range(2904):
           H11.values[i+1,0]=A18.values[i,0]
A1['flowafteconMP'] = A18
A1['flowafteconMP(t-1)'] = H11
H12 = pd.DataFrame(datas, columns=['flowtogasthem'])
A19 = pd.DataFrame(datas, columns=['flowtogasthem'])
for i in range(2904):
           H12.values[i+1,0]=A19.values[i,0]
A1['flowtogasthem'] = A19
A1['flowtogasthem(t-1)'] = H12
H13 = pd.DataFrame(datas, columns=['presbotMP'])
A20 = pd.DataFrame(datas, columns=['presbotMP'])
for i in range(2904):
           H13.values[i+1,0]=A20.values[i,0]
A1['presbotMP'] = A20
A1['presbotMP(t-1)'] = H13
H14 = pd.DataFrame(datas, columns=['tempbotMP'])
A21 = pd.DataFrame(datas, columns=['tempbotMP'])
for i in range(2904):
           H14.values[i+1,0]=A21.values[i,0]
A1['tempbotMP'] = A21
A1['tempbotMP(t-1)'] = H14
H15 = pd.DataFrame(datas, columns=['tempgasaftsupheat'])
A22 = pd.DataFrame(datas, columns=['tempgasaftsupheat'])
for i in range(2904):
           H15.values[i+1,0]=A22.values[i,0]
A1['tempgasaftsupheat'] = A22
A1['tempgasaftsupheat(t-1)'] = H15
H16 = pd.DataFrame(datas, columns=['presgasaftsupheat'])
A23 = pd.DataFrame(datas, columns=['presgasaftsupheat'])
for i in range(2904):
           H16.values[i+1,0]=A23.values[i,0]
A1['presgasaftsupheat'] = A23
A1['presgasaftsupheat(t-1)'] = H16
H17 = pd.DataFrame(datas, columns=['tempaftuni(IP+CRH)'])
A24 = pd.DataFrame(datas, columns=['tempaftuni(IP+CRH)'])
for i in range(2904):
           H17.values[i+1,0]=A24.values[i,0]
A1['tempaftuni(IP+CRH)'] = A24
A1['tempaftuni(IP+CRH)(t-1)'] = H17
H18 = pd.DataFrame(datas, columns=['tempaftuni(IP+CRH)andaftreheat'])
A25 = pd.DataFrame(datas, columns=['tempaftuni(IP+CRH)andaftreheat'])
for i in range(2904):
           H18.values[i+1,0]=A25.values[i,0]
A1['tempaftuni(IP+CRH)andaftreheat'] = A25
A1['tempaftuni(IP+CRH)andaftreheat(t-1)'] = H18
H19 = pd.DataFrame(datas, columns=['flowaftuni(IP+CRH)andaftreheat'])
A26 = pd.DataFrame(datas, columns=['flowaftuni(IP+CRH)andaftreheat'])
for i in range(2904):
           H19.values[i+1,0]=A26.values[i,0]
A1['flowaftuni(IP+CRH)andaftreheat'] = A26
A1['flowaftuni(IP+CRH)andaftreheat(t-1)'] = H19
H20 = pd.DataFrame(datas, columns=['flowaftpumpHP'])
A27 = pd.DataFrame(datas, columns=['flowaftpumpHP'])
for i in range(2904):
           H20.values[i+1,0]=A27.values[i,0]
A1['flowaftpumpHP'] = A27
A1['flowaftpumpHP(t-1)'] = H20
H21 = pd.DataFrame(datas, columns=['presaftpumpHP'])
A28 = pd.DataFrame(datas, columns=['presaftpumpHP'])
for i in range(2904):
           H21.values[i+1,0]=A28.values[i,0]
A1['presaftpumpHP'] = A28
A1['presaftpumpHP(t-1)'] = H21
H22 = pd.DataFrame(datas, columns=['tempaftpumpHP'])
A29 = pd.DataFrame(datas, columns=['tempaftpumpHP'])
for i in range(2904):
           H22.values[i+1,0]=A29.values[i,0]
A1['tempaftpumpHP'] = A29
A1['tempaftpumpHP(t-1)'] = H22
H23 = pd.DataFrame(datas, columns=['wattempafteconHP'])
A30 = pd.DataFrame(datas, columns=['wattempafteconHP'])
for i in range(2904):
           H23.values[i+1,0]=A30.values[i,0]
A1['wattempafteconHP'] = A30
A1['wattempafteconHP(t-1)'] = H23
H24 = pd.DataFrame(datas, columns=['tempbotHP'])
A31 = pd.DataFrame(datas, columns=['tempbotHP'])
for i in range(2904):
           H24.values[i+1,0]=A31.values[i,0]
A1['tempbotHP'] = A31
A1['tempbotHP(t-1)'] = H24
H25 = pd.DataFrame(datas, columns=['presbotHP'])
A32 = pd.DataFrame(datas, columns=['presbotHP'])
for i in range(2904):
           H25.values[i+1,0]=A32.values[i,0]
A1['presbotHP'] = A32
A1['presbotHP(t-1)'] = H25
H26 = pd.DataFrame(datas, columns=['tempHPaftNo1supheatHP'])
A33 = pd.DataFrame(datas, columns=['tempHPaftNo1supheatHP'])
for i in range(2904):
           H26.values[i+1,0]=A33.values[i,0]
A1['tempHPaftNo1supheatHP'] = A33
A1['tempHPaftNo1supheatHP(t-1)'] = H26
H27 = pd.DataFrame(datas, columns=['tempHPaftNo1supheatHPandaftDesupheat'])
A34 = pd.DataFrame(datas, columns=['tempHPaftNo1supheatHPandaftDesupheat'])
for i in range(2904):
           H27.values[i+1,0]=A34.values[i,0]
A1['tempHPaftNo1supheatHPandaftDesupheat'] = A34
A1['tempHPaftNo1supheatHPandaftDesupheat(t-1)'] = H27
H28 = pd.DataFrame(datas, columns=['tempgasHPtoturbine'])
A35 = pd.DataFrame(datas, columns=['tempgasHPtoturbine'])
for i in range(2904):
           H28.values[i+1,0]=A35.values[i,0]
A1['tempgasHPtoturbine'] = A35
A1['tempgasHPtoturbine(t-1)'] = H28
H29 = pd.DataFrame(datas, columns=['presgasHPtoturbine'])
A36 = pd.DataFrame(datas, columns=['presgasHPtoturbine'])
for i in range(2904):
           H29.values[i+1,0]=A36.values[i,0]
A1['presgasHPtoturbine'] = A36
A1['presgasHPtoturbine(t-1)'] = H29
H30 = pd.DataFrame(datas, columns=['flowgasHPtoturbine'])
A37 = pd.DataFrame(datas, columns=['flowgasHPtoturbine'])
for i in range(2904):
           H30.values[i+1,0]=A37.values[i,0]
A1['flowgasHPtoturbine'] = A37
A1['flowgasHPtoturbine(t-1)'] = H30
H31 = pd.DataFrame(datas, columns=['HRSGefficiency'])
A38 = pd.DataFrame(datas, columns=['HRSGefficiency'])
for i in range(2904):
           H31.values[i+1,0]=A38.values[i,0]
A1['HRSGefficiency'] = A38
A1['HRSGefficiency(t-1)'] = H31
H32 = pd.DataFrame(datas, columns=['Condenserefficiency'])
A39 = pd.DataFrame(datas, columns=['Condenserefficiency'])
for i in range(2904):
           H32.values[i+1,0]=A39.values[i,0]
A1['Condenserefficiency'] = A39
A1['Condenserefficiency(t-1)'] = H32
X1=A1
Y1 = pd.DataFrame(datas, columns=['HRSGefficiency', 'Condenserefficiency'])
Y2 = pd.DataFrame(datas, columns=['HRSGefficiency', 'Condenserefficiency'])
for i in range(2904):
    for j in range(2):
           Y1.values[i,j]=Y2.values[i+1,j]
#Y1.values[2904,0]=0
#Y1=Y1[Y1!=0]
#Y1=Y1.dropna()
I=[]
for i in range(2905):
       n=0
       if Y1.values[i,1]<0.3 and i<2903 and Y1.values[i,1]!=0:
           Y1.values[i, 0] = 0
           Y1.values[i, 1] = 0
           I.append(i)
           n=1
           Y1.values[i+1,0]=0
           Y1.values[i+1, 1] = 0
           I.append(i+1)
           Y1.values[i+2,0]=0
           Y1.values[i+2, 1] = 0
           I.append(i+2)
       elif Y1.values[i,1]>0.575 and (i<2903) and Y1.values[i,1]!=0:
           Y1.values[i, 0] = 0
           Y1.values[i, 1] = 0
           I.append(i)
           n=1
           Y1.values[i+1,0]=0
           Y1.values[i+1, 1] = 0
           I.append(i+1)
           Y1.values[i+2,0]=0
           Y1.values[i+2, 1] = 0
           I.append(i+2)
       if n == 0 and i < 2903 and Y1.values[i,1]!=0:
         if ((X2.values[i, 37] * X2.values[i, 38]) / 3600) != 0 and (X2.values[i, 45] / 3600) != 0:
           if abs((((X2.values[i, 44]) / ((X2.values[i, 37] * X2.values[i, 38]) / 3600)) - (1 / (X2.values[i, 45] / 3600)))) >= 0.05:
                       Y1.values[i, 0] = 0
                       Y1.values[i, 1] = 0
                       I.append(i)
                       Y1.values[i + 1, 0] = 0
                       Y1.values[i + 1, 1] = 0
                       I.append(i + 1)
                       Y1.values[i + 2, 0] = 0
                       Y1.values[i + 2, 1] = 0
                       I.append(i + 2)

Y1=Y1[Y1!=0]
Y1=Y1.dropna()
a=np.shape(I)[0]

for i in range(np.shape(I)[0]):
     X1=X1.drop(X1.index[I[np.shape(I)[0]-1-i]])

for i in range(np.shape(X1.columns)[0]):
     X1[X1.columns[i]]=X1[X1.columns[i]].fillna(np.mean(X1.values[0, i]))

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