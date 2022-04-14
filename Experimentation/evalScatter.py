import pandas as pd
import numpy as np
from scipy.stats import pearsonr

_SQRT2 = np.sqrt(2)

for imgno in range(1,16):
    data1 = pd.read_csv("SYNTHETIC_DATA/SCATTER/sc"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = pd.read_csv("SYNTHETIC_DATA/SCATTER_RESULTS/data_sc"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()

    dx = max(data1[:,0])-min(data1[:,0])
    t=np.concatenate(data1[:,1:])
    t = t[~np.isnan(t)]
    dy = max(t)-min(t)
    TPc = 0
    FPc = 0
    FNc = 0
    M = []
    for k in range(len(data1[0])-1):
        ids = list(range(len(data2)))
        t = 0
        for i in range(len(data1)):
            flag = True
            Cflag = True
            for j in ids:
                (a,b,c,d)=(data1[i][0],data1[i][k+1],data2[j][2*k],data2[j][2*k+1])
                if not np.isnan((a,b,c,d)).any():
                    if (abs(a-c)/dx <= 0.02):
                        M += [[b,d]]
                        Cflag = False
                        if (abs(b-d)/dy <= 0.02):
                            flag = False
                            ids.remove(j)
                            TPc += 1
                            break
            if flag:
                if Cflag:
                    FNc += 1
                else:
                    FPc += 1


    # print(TPc,FPc,FNc)

    M=np.array(M)
    m = [np.abs((i[1]-i[0]))/i[0] for i in M if i[0] > 0]
    MAPE = np.sum(m)/len(m)
    nMAE = np.sum(np.abs(M[:,0]-M[:,1]))/np.sum(M[:,0])

    prec = 0
    recall = 0
    F1src = 0
    if (TPc+FPc) != 0:
        prec = TPc/(TPc+FPc)
    if (TPc+FNc) != 0:
        recall = TPc/(TPc+FNc)
    if (prec+recall) != 0:
        F1src = 2*prec*recall/(prec+recall)
    #
    # corra, _ = pearsonr(data1[:,0],data1[:,1])
    # corrb, _ = pearsonr(data2[:,0],data2[:,1])

    print(prec,recall,F1src,MAPE,nMAE)#, corra,corrb)
