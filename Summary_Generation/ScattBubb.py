import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import scipy.stats as st
from gingerit.gingerit import GingerIt
import pysbd, re # pip install pysbd

# To get local maxima ans minima
def predictTrend(hgt,ylabels,xlabel,chart_type,x_title,y_title,  inter):
    trend_str = ''
    xlbls = xlabel
    local_max = argrelextrema(hgt, np.greater)
    local_min = argrelextrema(hgt, np.less)
    order = np.array([0] * len(xlbls))
    order[local_min] = -1
    order[local_max] = 1

    if  inter == False or np.count_nonzero(order)<2:
        # In the case of interclass trends
        if inter == True:
            neg_count = len([num for num in hgt if num <= 0])
            if neg_count>(len(hgt)-neg_count):
                hgt = np.array([num*-1 for num in hgt])
            trend_str=". The bar height differnce between "+ylabels[0]+" and "+ylabels[1]
            if y_title != '_':
                trend_str=". The "+y_title+" differnce between "+ylabels[0]+" and "+ylabels[1]
        elif ylabels =='Y':
            trend_str=". The Y axis value"
            if y_title != '_':
                trend_str=". The "+y_title
        elif ylabels =='freq':
            trend_str=". The frequency"
        else:
            trend_str=". The "+ylabels
            if y_title != '_':
                trend_str=". The "+y_title+" of "+ylabels

        if list(order)==[0]*len(xlbls):
            if(int(hgt[0])<int(hgt[1])):
                trend_str += " has an overall increasing trend"
                if( int(hgt[len(hgt)-2])>int(hgt[len(hgt)-1]) ):
                    trend_str += " till "+str(xlbls[len(xlbls)-2])+" and ends with a drop in "+str(xlbls[len(xlbls)-1])
                else:
                    trend_str += " from "+str(xlbls[0])+" to "+str(xlbls[len(xlbls)-1])
            elif(int(hgt[0])>int(hgt[1])):
                trend_str += " has an overall decreasing trend"
                if( int(hgt[len(hgt)-2])<int(hgt[len(hgt)-1]) ):
                    trend_str += " till "+str(xlbls[len(xlbls)-2])+" and ends with a peak in "+str(xlbls[len(xlbls)-1])
                else:
                    trend_str += " from "+str(xlbls[0])+" to "+str(xlbls[len(xlbls)-1])
            else:
                if(int(hgt[0])!=int(hgt[len(hgt)-1])):
                    trend_str += " is uniform with "+str(int(hgt[0]))+" till "+str(xlbls[len(xlbls)-2])+" and finally ends with "+str(int(hgt[len(hgt)-1]))+" in "+str(xlbls[len(xlbls)-1])
                else:
                    trend_str += " is uniform with "+str(int(hgt[0]))+" throughout the entire period"

        elif np.count_nonzero(order)<4:
            # speak about global maximum and min
            ht= hgt.tolist()
            xlbls = xlbls.tolist()
            xlbls2 = xlbls
            xlbls2[ht.index(max(ht))] = str(xlbls[ht.index(max(ht))]) + ' the maximum value'
            xlbls2[ht.index(min(ht))] = str(xlbls[ht.index(min(ht))]) + ' the minimum value'

            trend_str += " starts with "+str(int(hgt[0]))+" at "+x_title+' '+str(xlbls2[0])+" then "
            j=1
            while j<len(order):
                if order[j]==-1:
                    if list(order[:j])==[0]*j:
                        trend_str += "declines till "+str(xlbls2[j])
                    else:
                        trend_str += ", followed by a decreasing trend till "+str(xlbls2[j])
                elif order[j]==1:
                    if list(order[:j])==[0]*j:
                        trend_str += "increases till "+str(xlbls2[j])
                    else :
                        trend_str += ", followed by an increasing trend till "+str(xlbls2[j])
                j+=1
            if(order[j-2]!=0):
                trend_str += ", and finally ends with "+str(int(hgt[j-1]))+" in "+str(xlbls2[j-1])
            else :
                if(hgt[j-1]<hgt[j-2]):
                    trend_str += ", and ends with a decreasing trend till "+str(xlbls2[j-1])
                else:
                    trend_str += ", and ends with a decreasing trend till "+str(xlbls2[j-1])
            xlbls2[ht.index(max(ht))] = xlbls2[ht.index(max(ht))].replace(' the maximum value','')
            xlbls2[ht.index(min(ht))] = xlbls2[ht.index(min(ht))].replace(' the minimum value','')
        else:
            #Just discuss the maximum and minmium value
            ht= hgt.tolist()
            trend_str += ' has it maximum and minmum values '+str(int(max(ht)))+' and '+str(int(min(ht)))+' at '+str(x_title)+' '+str(xlbls[ht.index(max(ht))])+', and '+str(xlbls[ht.index(min(ht))])+" respectively"

    return trend_str

def simplebarsumm(yvals,ylabels,slabs,chart_type,x_title,y_title, inter):
    if len(slabs)>6 or inter == True:
        # Speak about maxima and minima
        Summ = predictTrend(yvals,ylabels,slabs,chart_type,x_title,y_title, inter)
    else :
        if x_title=='_':
            x_title = 'labels'
        if str(slabs[0]).isnumeric():
            if len(slabs) == 1:
                Summ = '. For the '+str(x_title)+' with '+str(slabs[0])+' the '+str(y_title)
                if not (ylabels == 'Y' or  ylabels == 'freq'):
                    Summ += " of "+ylabels
                Summ += ' is '+str(round(yvals[0],2))
            else:
                Summ = '. For the '+str(x_title)+' ranging form '+str(slabs[0])+' - '+str(slabs[-1])+' at the interval '+str(abs(slabs[0]-slabs[1]))
                Summ += ', the '+str(y_title)
                if not (ylabels == 'Y' or  ylabels == 'freq'):
                    Summ += " of "+ylabels
                Summ += ' are '
                for i in yvals[:-1]:
                    Summ += str(round(i,2))+', '
                Summ += 'and '+str(round(yvals[-1],2))+' respectively'
        else :
            if len(slabs) == 1:
                Summ = '. The '+str(y_title)
                if not (ylabels == 'Y' or  ylabels == 'freq'):
                    Summ += " of "+ylabels
                Summ += ' is '+str(round(yvals[0],2))+' for the '+str(x_title)+" "+str(slabs[0])
            else:
                Summ = '. The '+str(y_title)
                if not (ylabels == 'Y' or  ylabels == 'freq'):
                    Summ += " of "+ylabels
                Summ += ' are '
                for i in yvals[:-1]:
                    Summ += str(round(i,2))+', '
                Summ += 'and '+str(round(yvals[-1],2))
                Summ += ' for the '+str(x_title)+" "
                for i in slabs[:-1]:
                    Summ += str(i)+', '
                Summ += 'and '+str(slabs[-1])+' respectively'

    return Summ

segmentor = pysbd.Segmenter(language="en", clean=False)
subsegment_re = r'[^;:\n•]+[;,:\n•]?\s*'

def GrammerCorrect(par):
    fixed = []
    for sentence in segmentor.segment(par):
        if len(sentence) < 300:
            fixed.append(GingerIt().parse(sentence)['result'])
        else:
            subsegments = re.findall(subsegment_re, sentence)
            if len(subsegments) == 1 or any(len(v) < 300 for v in subsegments):
                # print(f'Skipped: {sentence}') // No grammar check possible
                fixed.append(sentence)
            else:
                res = []
                for s in subsegments:
                    res.append(GingerIt().parse(s)['result'])
                fixed.append("".join(res))
    return " ".join(fixed)

def genrateSumm_SB(file):
    imgno = file.split('/')[-1].split(".")[0].split("_")[-1]
    path = os.path.dirname(file)+'/'
    df = pd.read_csv(file)
    title = df['title'][0]
    chart_type = df['chart_type'][0]
    x_title = df['x-title'][0]
    y_title = df['y-title'][0]
    if chart_type == 'Bubble Plot':
        df = df[df.Hue != '_']
    ylabels = list(df)[:len(list(df))-4]
    data = (df.loc[ : , ylabels]).values
     
    ### Visual Summary
    #Speak about starting line with titles
    Summ = 'The plot depicts a '+chart_type+' Graph'
    if title !='_':
        Summ += ' illustrating '+title
    if x_title != '_' and y_title != '_':
        Summ +='. The plot is between '+y_title+' on y-axis over '+x_title+' on the x-axis'
    elif y_title != '_':
        Summ +='. The plot is having '+y_title+' on y-axis'
    elif x_title != '_':
        Summ +='. The plot is having '+x_title+' on x-axis'
        
    
    
    if 'Simple' not in chart_type:
        # speaking about legend
        legname = np.unique(data[:,2])
        Summ +=' for '
        for i in range(len(legname)-1):
            Summ += str(legname[i])+", "
        Summ += "and "+str(legname[i+1])+" varients"

        # intra class differrences
        for i in range(len(legname)): 
            if chart_type == 'Bubble Plot':
                data2 = (df.loc[df['Hue'] == legname[i]]).values[:,:2]
            elif 'Simple' not in chart_type:
                data2 = (df.loc[df['Catogery'] == legname[i]]).values[:,:2]
            data2 = data2.astype(float).round(0).astype(int)
            Summ += simplebarsumm(data2[:,1],legname[i],data2[:,0],chart_type,x_title,y_title,False)
        Summ = Summ.replace("_ ", "")
        Summ = Summ.replace("\n", " ").replace('\r', '')
    else :
        data2 = df.values[:,:2].astype(float).round(0).astype(int)
        Summ += simplebarsumm(data2[:,1],'Y',data2[:,0],chart_type,x_title,y_title,False)
        Summ = Summ.replace("_ ", "")
        Summ = Summ.replace("\n", " ").replace('\r', '')



        

    ### Statistical Summary
    # For Catogeorical Graphs
    if x_title == '_' :
        x_title = 'x axis values'
    elif y_title == '_' :
        y_title = 'y axis values'
    pos = []
    neg = []
    rest = []
    corrval = []
    if 'Simple' in chart_type:
        data2 = df.values[:,:2].astype(float).round(0).astype(int)
        corrval, _ = st.spearmanr(data2[:,0], data2[:,1])
        if corrval >= 0.5:
            Summ += '. The '+str(x_title)+' and '+str(y_title)+' are positively correlated with spearman correlation value '+str(round(corrval,2))  
        elif corrval <= -0.5:
            Summ += '. The '+str(x_title)+' and '+str(y_title)+' are negatively correlated with spearman correlation value '+str(round(corrval,2))  
        else : 
            Summ += '. The '+str(x_title)+' and '+str(y_title)+' doesn\'t exhibit any correlation with spearman correlation value '+str(round(corrval,2))  
    else :
        for i in range(len(legname)):   
            if chart_type == 'Bubble Plot':
                data2 = (df.loc[df['Hue'] == legname[i]]).values[:,:2]
            else :
                data2 = (df.loc[df['Catogery'] == legname[i]]).values[:,:2]
            data2 = data2.astype(float).round(0).astype(int) 

            corr, _ = st.spearmanr(data2[:,0], data2[:,1])
            corrval += [corr]
            if corr >= 0.5:
                pos += [i]
            elif corr <= -0.5:
                neg += [i]
            else : 
                rest += [i]
    
        if len(pos)!=0:
            if len(pos) == 1:
                Summ += '. The '+str(x_title)+' and '+str(y_title)+' are positively correlated for the catogery \''+str(legname[pos[0]])+'\' with spearman correlation value '+str(round(corrval[pos[0]],2))  
            else : 
                Summ += '. The '+str(x_title)+' and '+str(y_title)+' are positively correlated for catogeries \''
                for i in pos[:-1]:         
                    Summ += str(legname[i])+'\', ' 
                Summ += 'and \''+str(legname[pos[-1]])+'\' with spearman correlation values '
                for i in pos[:-1]: 
                    Summ += str(round(corrval[i],2))+', '       
                Summ += 'and '+str(round(corrval[i],2))+' respectively '
        elif len(neg)!=0 :
            if len(neg) == 1:
                Summ += '. The '+str(x_title)+' and '+str(y_title)+' are negatively correlated for the catogery \''+str(legname[neg[0]])+'\' with spearman correlation value '+str(round(corrval[neg[0]],2))  
            else :
                Summ += '. The '+str(x_title)+' and '+str(y_title)+' are negatively correlated for catogeries \''
                for i in neg[:-1]:         
                    Summ += str(legname[i])+'\', ' 
                Summ += 'and \''+str(legname[neg[-1]])+'\' with spearman correlation values '
                for i in neg[:-1]: 
                    Summ += str(round(corrval[i],2))+', '       
                Summ += 'and '+str(round(corrval[i],2))+' respectively '
        else:
            if len(rest) == 1:
                Summ += '. The '+str(x_title)+' and '+str(y_title)+' doesn\'t exhibit any correlation for the catogery \''+str(legname[rest[0]])+'\' with spearman correlation value '+str(round(corrval[rest[0]],2))  
            else:
                Summ += '. The '+str(x_title)+' and '+str(y_title)+' doesn\'t exhibit any correlation for catogeries \''
                for i in rest[:-1]:         
                    Summ += str(legname[i])+'\', ' 
                Summ += 'and \''+str(legname[rest[-1]])+'\' '

    Summ = GrammerCorrect(Summ+'.')
    # Summ = Summ+'.'
    text_file = open(path+"summ_"+str(imgno)+".txt", "w")
    n = text_file.write(Summ)
    text_file.close()
