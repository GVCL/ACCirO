import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import pearsonr

_SQRT2 = np.sqrt(2)

for imgno in range(1,16):
    Dy = [1000,100,250,200,7,30,65,50,25,25000,30,50,16,45,50] # vGB
    DX = [140,50,80,70,12,70,60,50,100,700000,35,25,50,450,70] # vGB
    path = "SYNTHETIC_DATA/BUBBLE"


    # im = Image.open(path+"/bb"+str(imgno)+".png")
    # print(im.info['dpi'])
    data1 = pd.read_csv(path+"/data_bb"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = pd.read_csv(path+"_RESULTS/data_bb"+str(imgno)+".csv",  sep=",", index_col=False).to_numpy()
    data2 = data2[:,:-4]
    data1[:,-1] = np.sqrt(data1[:,-1].astype(np.double)) * 100 / 72.0
    omx = max(data1[:,-1])
    omn = min(data1[:,-1])
    data1[:,-1] = (data1[:,-1]-omn)/(omx-omn)
    nmx = max(data2[:,-1])
    nmn = min(data2[:,-1])
    data2[:,-1] = (data2[:,-1]-nmn)/(nmx-nmn)
    data1 = np.sort(data1, axis=0)
    data2 = np.sort(data2, axis=0)
    dx = DX[imgno-1]
    dy = Dy[imgno-1]

    TPc = 0
    FPc = 0 #comission errors, wrongly predicting exsisting val
    FNc = 0 # omission errors, missing data/prediction
    M = []
    # print(len(data1),len(data2))
    ids = list(range(len(data2)))
    for i in range(len(data1)):
        flag = True
        Cflag = True
        for j in ids:
            (a,b,c,d,s1,s2,r1,r2)=(data1[i][0],data1[i][1],data2[j][0],data2[j][1],data1[i][2],data2[j][2],data1[i][3],data2[j][3])
            if (abs(a-c)/dx <= 0.02):
                M += [[b,d]]
                Cflag = False
                if (abs(b-d)/dy <= 0.02):
                    Cflag = False
                    # if (s1.lower() != s2.lower()):
                    #     print(a,b,c,d,s1,s2)
                    if s1.lower() == s2.lower():
                        if abs(r1-r2) > 0.5:
                            print(r1,r2,abs(r1-r2))
                        flag = False
                        ids.remove(j)
                        TPc += 1
                        break
    if flag:
        if Cflag:
            FNc += 1
        else:
            FPc += 1

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

    corra, _ = pearsonr(data1[:,0],data1[:,1])
    corrb, _ = pearsonr(data2[:,0],data2[:,1])

    print(prec,recall,F1src,MAPE,nMAE,corra,corrb)

