import math
import numpy as np
import cv2
import os
import sys
import csv
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Extract_DataTables.utilsScatter import *
from Extract_DataTables.utils import *
from Summary_Generation.PieLineDot import genrateSumm_PLD
from Summary_Generation.ScattBubb import genrateSumm_SB


def findSubtype(im, notbubble = False):
    # imgplot = plt.imshow(im)
    # plt.show()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    bg_fill = np.zeros_like(thresh)

    areas = []
    xcoords = []
    ycoords = []
    for c in contours:
        # compute the center of the contours
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, pts = [c], color=255)
            #to check if the contours are going to overlap on already recorded contours
            intersect_img = cv2.bitwise_and(bg_fill, mask, mask = None)
            if not cv2.countNonZero(intersect_img):
                bg_fill = cv2.bitwise_or(bg_fill, mask, mask = None)
                areas = areas + [cv2.contourArea(c)]
                xcoords = xcoords + [cX]
                ycoords = ycoords + [cY]
    radius = np.sqrt(np.array(areas)/math.pi).astype(int)
    if len(areas)!=0:
        _, counts = np.unique(np.array(areas), return_counts=True)
        if np.amax(counts)<3 and (not notbubble):
            return 'bubble'
        # no overlapping contours
        elif len([i for i in list(radius) if abs(i-np.mean(radius))>2]) == 0:
            # print(np.diff(np.diff(np.unique(np.array(xcoords))[1:-1])),np.diff(np.diff(np.unique(np.array(ycoords))[1:-1])))
            if len([i for i in np.diff(np.diff(np.unique(np.array(xcoords))[1:-1])) if abs(i)>3]) == 0:
                if len([i for i in np.diff(np.diff(np.unique(np.array(ycoords))[1:-1])) if abs(i)>3]) == 0:
                    return 'dot'
        return 'scatter'
    else :
        sys.exit("\u001b[31m FAILED: Data extraction \nCan't find graphical objects in canvas")


def extScatterTypes(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    graph_img = cv2.imread(filename)
    if(graph_img.shape[2]==4):
        graph_img[graph_img[:,:,3]==0] = [255,255,255,255]
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2RGB)
    h, w, _= np.shape(graph_img)

    '''Canvas and Legend Extraction'''
    img = graph_img.copy()
    # chart_dict, IS_MULTI_CHART, legend_colors, legend_names, canvas_img, pix_centers = extractCanvaLeg(img,'scatter')
    chart_dict, IS_MULTI_CHART, legend_colors, legend_names, bg_color, canvas_img, pix_centers = extractCanvaLeg(img,'scatter')
    print("Canvas & Legend Extracted Sucessfully .. !")

    d = chart_dict['canvas']
    ecanvasimg = np.ones(img.shape,dtype=np.uint8)*255
    ecanvasimg[d['y']:d['h']+d['y'],d['x']:d['w']+d['x'],:] = canvas_img[d['y']:d['h']+d['y'],d['x']:d['w']+d['x'],:]
    kernel = np.ones((3,3), np.uint8)
    ecanvasimg = cv2.erode(ecanvasimg, kernel, iterations=1)
    sub_type = findSubtype(ecanvasimg,notbubble = True)

    if sub_type == 'bubble':
        print("The sub type of scatter plot is : ",sub_type)
        '''Bubble Parameters Extraction - [center, color, radius] of each bubble '''
        bubble_params = np.array(Compute_BubbPixData(ecanvasimg,list(legend_colors)))

        # Indentifying bubble color and assigning its groups .. !
        ovrlap_clrs = []
        for id, i in enumerate(bubble_params[:,2]):
            if i in list(legend_colors) :
                bubble_params[id][2] = legend_names[legend_colors.index(i)]
            elif i not in ovrlap_clrs:
                ovrlap_clrs += [i]

        if len(ovrlap_clrs) != 0:
            ovrlap_sets = findOverlapComb(list(legend_colors), bg_color, ovrlap_clrs)
            extra = []
            for id, i in enumerate(bubble_params[:,2]) :
                if not isinstance(i, str):
                    if len(ovrlap_sets)!=0 and (i in np.array(ovrlap_sets)[:,0].tolist()) :
                        k = ovrlap_sets[np.array(ovrlap_sets)[:,0].tolist().index(i)][1]
                        bubble_params[id][2] = legend_names[k[0]]
                        for t in k[1:] :
                            x = list(bubble_params[id])
                            x[2] = legend_names[t]
                            extra += [x]
                    else :
                        bubble_params[id][2] = '_'
            if len(ovrlap_sets)!=0:
                bubble_params = np.append(bubble_params, np.array(extra), axis=0)
        # print(bubble_params)
        print(" Data Extracted Sucessfully in Pixel Space ")

        img = graph_img.copy()
        title, y_title, ybox_centers, Ylabel, x_title, xbox_centers, Xlabel = extractLablTitl(img,chart_dict,IS_MULTI_CHART)
        print("Chart Labels & Titles Extracted Sucessfully .. !")
        # print( title, y_title, x_title)
        bubble_params[:,:2] = Scale_XYCoords(Ylabel,ybox_centers,Xlabel,xbox_centers,bubble_params[:,:2])

        data = pd.DataFrame(bubble_params,columns =["X", "Y", "Hue", "Size"])
        fig, ax = plt.subplots()
        axes = sns.scatterplot(data=data, x="X", y="Y", hue="Hue", size="Size", alpha=0.5, sizes=(100, 1000))
        # Legend split and place outside
        num_of_colors   = len(data["Hue"].unique()) + 1
        handles, labels = axes.get_legend_handles_labels()
        color_hl = handles[:num_of_colors], labels[:num_of_colors]
        color_leg = axes.legend(*color_hl,
                                bbox_to_anchor = (1.05, 1),
                                loc            = 'upper left',
                                borderaxespad  = 0.,
                                labelspacing = 1.2)
        for lh in color_leg.legendHandles:
            lh.set_alpha(0.5)
            lh._sizes = [100]
        axes.add_artist(color_leg)
        plt.subplots_adjust(right=0.75)
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        # plt.show()

        # # # Writing data to CSV file
        plt.savefig(path+"Reconstructed_"+str(image_name)+".png")
        bubble_params = bubble_params.tolist()
        bubble_params = [["X", "Y", "Hue", "Size", 'chart_type','title','x-title','y-title']]+bubble_params
        bubble_params[1] = bubble_params[1]+['Bubble Plot',title, x_title, y_title]
        with open(path+'data_'+str(image_name)+'.csv', 'w' ) as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(bubble_params)
        print("Chart Reconstruction Done .. !")
        genrateSumm_SB(path+'data_'+str(image_name)+'.csv')
        print("Chart Summary Generated .. !")

    else:
        '''Canvas and Legend Extraction'''
        img = graph_img.copy()
        chart_dict, IS_MULTI_CHART, legend_colors, legend_names, bg_color, canvas_img2, pix_centers = extractCanvaLeg(img,'scatter',smplescatt = True)
        if sub_type == 'scatter':
            d = chart_dict['canvas']
            ecanvasimg2 = np.ones(img.shape,dtype=np.uint8)*255
            ecanvasimg2[d['y']-10:d['h']+d['y']+20,d['x']-10:d['w']+d['x']+20,:] = canvas_img2[d['y']-10:d['h']+d['y']+20,d['x']-10:d['w']+d['x']+20,:]
            kernel = np.ones((3,3), np.uint8)
            ecanvasimg2 = cv2.erode(ecanvasimg2, kernel, iterations=1)
            sub_type = findSubtype(ecanvasimg2)
        print("The sub type of scatter plot is : ",sub_type)

        '''Components extraction'''
        img = graph_img.copy()
        title, y_title, ybox_centers, Ylabel, x_title, xbox_centers, Xlabel = extractLablTitl(img,chart_dict,IS_MULTI_CHART)
        print("Chart Labels & Titles Extracted Sucessfully .. !")

        # img = graph_img.copy()
        # segchart = viewSegChart(img,chart_dict)

        ''' Map pixels to original coordinates'''
        if(isinstance(Ylabel[0], str)):
            ybox_centers = np.array([ybox_centers[i] for i in range(len(Ylabel)) if Ylabel[i].isnumeric()])
            Ylabel = [int(i) for i in Ylabel if i.isnumeric()]
        for i in np.unique(Ylabel):
            id=[j for j, val in enumerate(Ylabel) if i==val]
            if(len(id)==2 and Ylabel[id[0]]!=0):
                if(ybox_centers[id[0]][1]<ybox_centers[id[1]][1]):
                    Ylabel[id[1]]*=-1
                    neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[1]][1])[0]
                else:
                    Ylabel[id[0]]*=-1
                    neg_ids=np.where(ybox_centers[:,1] > ybox_centers[id[0]][1])[0]
                for i in neg_ids:
                    Ylabel[i]*=-1

        if(isinstance(Xlabel[0], str)):
            xbox_centers = np.array([xbox_centers[i] for i in range(len(Xlabel)) if Xlabel[i].isnumeric()])
            Xlabel = [int(i) for i in Xlabel if i.isnumeric()]
        xbox_center =xbox_centers[:,0]
        for i in np.unique(Xlabel):
            id=[j for j, val in enumerate(Xlabel) if i==val]
            if(len(id)==2 and Ylabel[id[0]]!=0):
                if(xbox_center[id[0]]>xbox_center[id[1]]):
                    Xlabel[id[1]]*=-1
                    neg_ids=np.where(xbox_center < xbox_center[id[1]])[0]
                else:
                    Xlabel[id[0]]*=-1
                    neg_ids=np.where(xbox_center < xbox_center[id[0]])[0]
                for i in neg_ids:
                    Xlabel[i]*=-1

        '''normalize center to obtained labels'''
        # centers = np.array(center).astype(float)
        xbox_centers, Xlabel = (list(xbox_centers),list(Xlabel))
        normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
        t = np.array(sorted(np.concatenate((ybox_centers, np.array([Ylabel]).T), axis=1), key=lambda x: x[1]))#, reverse= True))
        ybox_centers,Ylabel = (t[:,0:2],list(t[:,2]))
        normalize_scaley = (Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1])
        if sub_type == 'dot':
            centers = []
            for i in range(len(pix_centers)):
                c = np.array(pix_centers[i]).astype(float)
                typ = [legend_names[i]]*len(pix_centers[i])
                c[:, 0] = (((c[:, 0] - xbox_centers[0][0]) * normalize_scalex) + Xlabel[0]).round(0)
                c[:, 1] = (((c[:, 1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]).round(0)
                c = c.astype(int).tolist()
                centers += (np.append(np.array(c), np.transpose(np.array([typ])), axis=1)).tolist()
            df = pd.DataFrame(np.array(centers))
            df.iloc[:, [0,1]] = df.iloc[:, [0,1]].apply(pd.to_numeric)
            df = df.groupby([0,2]).agg({1 : [np.min, np.max]}).reset_index()
            dot_params = df.to_numpy()
            if list(dot_params[:,1]).count('_')==len(dot_params) :
                # single dot plot
                dot_params[:,1] = dot_params[:,3]- dot_params[:,2] + 1
                dot_params = dot_params[:,:2]
                fig, ax = plt.subplots()
                for val in dot_params:
                    ax.plot([val[0]]*(val[1]), list(range(1,val[1]+1)), 'ko', ms=10, linestyle='')
                plt.title(title)
                plt.xlabel(x_title)
                plt.ylabel(y_title)
                # # # Writing data to CSV file
                plt.savefig(path+"Reconstructed_"+str(image_name)+".png")
                dot_params = dot_params.tolist()
                dot_params = [["X", "Y", 'chart_type','title','x-title','y-title']]+dot_params
                dot_params[1] = dot_params[1]+['Simple Dot Plot', title, x_title, y_title]
            else:
                # multi dot plot
                df = pd.DataFrame(dot_params)
                ids = np.unique(dot_params[:,0])
                dot_params = ids
                for j in legend_names:
                    t = df.loc[df[1] == j].to_numpy()
                    t[:,2] = t[:,3] - t[:,2] + 1
                    t2 = []
                    for i in ids:
                        if i in list(t[:,0]):
                            t2 += [t[list(t[:,0]).index(i)][2]]
                        else:
                            t2 += [0]
                    dot_params = np.column_stack((dot_params, np.array(t2)))
                # Reconstruction
                dot_params = dot_params.transpose()
                cmap = matplotlib.cm.get_cmap('tab10')
                colors=[cmap(i) for i in range(len(legend_names))][::-1]
                fig, ax = plt.subplots(figsize=(6,4.8))
                for i in range(1,len(dot_params)) :
                    for j in range(len(dot_params[0])):
                        k = np.sum(dot_params[1:i,j])+1
                        ax.plot([dot_params[0][j]]*dot_params[i][j], list(range(k,dot_params[i][j]+k)), 'o', ms=9, color=colors[i-1], linestyle='')
                    k = np.sum(dot_params[1:i,0])+1
                    ax.plot([dot_params[0][0]]*dot_params[i][0], list(range(k,dot_params[i][0]+k)),'o', ms=9, color=colors[i-1], linestyle='', label=legend_names[i-1])
                plt.subplots_adjust(right=0.75)
                plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
                plt.title(title)
                plt.xlabel(x_title)
                plt.ylabel(y_title)
                # # # Writing data to CSV file
                plt.savefig(path+"Reconstructed_"+str(image_name)+".png")
                dot_params = [["X"]+legend_names+['chart_type','title','x-title','y-title']]+dot_params.transpose().tolist()
                dot_params[1] = dot_params[1]+['Dot Plot', title, x_title, y_title]
            with open(path+'data_'+str(image_name)+'.csv', 'w' ) as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(dot_params)
            print("Chart Reconstruction Done .. !")
            genrateSumm_PLD(path+'data_'+str(image_name)+'.csv')
            print("Chart Summary Generated .. !")
        else :
            centers = []
            for i in range(len(pix_centers)):
                c = np.array(pix_centers[i]).astype(float)
                typ = [legend_names[i]]*len(pix_centers[i])
                c = c[np.argsort(c[:, 0])]
                c[:, 0] = (((c[:,0] - xbox_centers[0][0]) * normalize_scalex) + Xlabel[0]).round(2)
                c[:, 1] = (((c[:, 1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]).round(2)
                centers += [np.append(np.array(c,dtype=np.object), np.transpose(np.array([typ],dtype=np.object)), axis=1).tolist()]

            # for i,clr in enumerate(legend_colors):
            #
            #     # if np.any( np.all(graph_img==clr,axis=2)):
            #     #     mas = np.all(graph_img==clr,axis=2).astype(np.uint8)*255
            #     #     if clr == [0,0,0] :
            #     #         mas = np.all(ecanvasimg==clr,axis=2).astype(np.uint8)*255
            #     #     c = np.array(CmpuCntr(mas)).astype(float)
            #     #     typ = [legend_names[i]]*len(c)
            #     #     c = c[np.argsort(c[:, 0])]
            #     #     c[:, 0] = (((c[:,0] - xbox_centers[0][0]) * normalize_scalex) + Xlabel[0]).round(2)
            #     #     c[:, 1] = (((c[:, 1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]).round(2)
            #     #     centers += [np.append(np.array(c,dtype=np.object), np.transpose(np.array([typ],dtype=np.object)), axis=1).tolist()]


            '''Reconstruct Chart'''
            fig, ax = plt.subplots()
            for i in range(len(centers)):
                plt.scatter(np.array(centers[i])[:,0].astype(float), np.array(centers[i])[:,1].astype(float), label=legend_names[i], color=(np.array(legend_colors[i][::-1])/255))
            plt.xlabel(x_title)
            plt.ylabel(y_title)
            plt.title(title)
            if IS_MULTI_CHART:
                plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(path+"reconstructed_"+str(image_name)+".png")

            # Writing data to CSV file
            L=[]
            L = L + [['X', 'Y', 'Catogery']] + np.concatenate(centers).tolist()
            # L = L + [sum([['X', legend_names[j]] for j in range(len(centers))],[])]
            # max_len = max([len(i) for i in centers])
            # for j in range(max_len):
            #     L = L+[(np.array([centers[i][j] for i in range(len(centers)) if j<len(centers[i])]).flatten()).tolist()]

            L[0] = L[0]+['chart_type','title','x-title','y-title']
            if IS_MULTI_CHART:
                L[1] = L[1] + ['Scatter',title, x_title, y_title]
            else:
                L[1] = L[1] + ['Simple Scatter',title, x_title, y_title]
            with open(path+'data_'+str(image_name)+'.csv', 'w' ) as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerows(L)
            print("Chart Reconstruction Done .. !")
            genrateSumm_SB(path+'data_'+str(image_name)+'.csv')
            print("Chart Summary Generated .. !")


