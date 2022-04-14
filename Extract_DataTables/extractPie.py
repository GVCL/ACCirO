import os
import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from Extract_DataTables.utils import findPie,viewSegChart
from Summary_Generation.PieLineDot import genrateSumm_PLD
import textwrap

def extPie(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    graph_img = cv2.imread(filename)

    if(graph_img.shape[2]==4):
        graph_img[graph_img[:,:,3]==0] = [255,255,255]
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2RGB)
    kernel = np.ones((3,3), np.uint8)

    '''Canvas and Legend Extraction'''
    fig=plt.figure()
    img = graph_img.copy()
    data, legend_names, legend_colors, Chart_Title,= findPie(img)
    print("Chart Components and Data Extracted Sucessfully .. !")

    my_wrap = textwrap.TextWrapper(width = 20)
    plt.pie(data, colors = legend_colors, counterclock=False, startangle=90, autopct='%1.1f%%', pctdistance=1.1)
    plt.legend(labels = ['\n'.join(my_wrap.wrap(text=i)) for i in legend_names], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.title(Chart_Title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path+"Reconstructed_"+str(image_name)+".png")

    # Writing data to CSV file
    L = [['X','Y','chart_type','title']]
    L = L + [[legend_names[0], round(data[0]*100/sum(data),1), 'Pie', Chart_Title]]
    L = L + [[legend_names[i], round(data[i]*100/sum(data),1)] for i in range(1,len(legend_names))]
    with open(path+'data_'+str(image_name)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)
    print("Chart Reconstruction Done .. !")

    genrateSumm_PLD(path+'data_'+str(image_name)+'.csv')
    print("Chart Summary Generated .. !")
