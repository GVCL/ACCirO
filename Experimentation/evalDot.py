import pandas as pd
import numpy as np

_SQRT2 = np.sqrt(2)

for imgno in range(1,16):
    Dy = [8,8,4,7,6,6,6,13,12,7,12,10,14,7,11] # vGB
    path = "SYNTHETIC_DATA/DOT"

    data1 = pd.read_csv(path+"/data_dp"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = pd.read_csv(path+"_RESULTS/data_dp"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data1 = data1[:,1:]
    data2 = data2[:,1:-4]
    dy = Dy[imgno-1]

    TPc = 0
    FPc = 0 #comission errors, wrongly predicting exsisting val
    FNc = 0 # omission errors, missing data/prediction
    m = []
    for k in range(len(data1[2])):
        for i in range(len(data1)):
            if (abs(data1[i][k]-data2[i][k])/dy <= 0.02):
                if data1[i][k]!=0 and data2[i][k]==0:
                    FNc += 1
                else:
                    TPc += 1
            else :
                FPc +=1
        m += [np.abs(data1[i][k]-data2[i][k])/data1[i][k] for i in range(len(data1)) if data1[i][k]!=0]
    MAPE = np.sum(m)/len(m)
    nMAE = np.sum(np.abs(data1[:,1:len(data1[2])]-data2[:,1:len(data1[2])]))/max(1,np.sum(data1[:,1:len(data1[2])]))
    prec = 0
    recall = 0
    F1src = 0
    if (TPc+FPc) != 0:
        prec = TPc/(TPc+FPc)
    if (TPc+FNc) != 0:
        recall = TPc/(TPc+FNc)
    if (prec+recall) != 0:
        F1src = 2*prec*recall/(prec+recall)

    print(prec,recall,F1src,MAPE,nMAE)



