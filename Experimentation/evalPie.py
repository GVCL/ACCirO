import pandas as pd
import numpy as np

_SQRT2 = np.sqrt(2)

for imgno in range(1,16):
    data1 = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/LINE_PIE/SYNTHETIC_DATA/PIE/orig_spie"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/LINE_PIE/SYNTHETIC_DATA/PIE_RESULTS/data_spie"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()

    t1 = (data1[:,1]/sum(data1[:,1]))*100
    t2 = (data2[:,1]/sum(data2[:,1]))*100

    TPc = 0
    FPc = 0 #comission errors, wrongly predicting exsisting val
    FNc = 0 # omission errors, missing data/prediction
    for i in range(len(data1)):
        if (abs(t1[i]-t2[i])/100 <= 0.02):
            if t1[i]!=0 and t2[i]==0:
                FNc += 1
            else:
                TPc += 1
        else :
            FPc +=1
    prec = 0
    recall = 0
    F1src = 0
    if (TPc+FPc) != 0:
        prec = TPc/(TPc+FPc)
    if (TPc+FNc) != 0:
        recall = TPc/(TPc+FNc)
    if (prec+recall) != 0:
        F1src = 2*prec*recall/(prec+recall)

    MAPE = 0
    for i in range(len(t1)):
        MAPE += abs(t1[i]-t2[i])/t1[i]
    MAPE = MAPE/len(t1)
    nMAE = np.sum(np.abs(t1-t2))/np.sum(t1)

    prec = TPc/(TPc+FPc)
    recall = TPc/(TPc+FNc)
    F1src = 2*prec*recall/(prec+recall)

    print(prec,recall,F1src,MAPE,nMAE)
