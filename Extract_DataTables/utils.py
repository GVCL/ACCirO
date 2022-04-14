import cv2
import numpy as np
from statistics import mode
from sklearn.cluster import *
from CRAFT_TextDetector import detect_text
from Deep_TextRecognition import text_recog
from keras.models import model_from_json
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from deskew import determine_skew
from skimage.transform import rotate
import glob
from Extract_DataTables.graphObjSeg import segment

def flwchnz(l):
    dff = np.diff(l)
    pos = list(np.where(dff > 0)[0])
    neg = list(np.where(dff < 0)[0])
    id = []
    if (len(pos)<len(neg)) or ((len(pos)==len(neg)!=0) and pos[0]>neg[0]):
        if len(pos) ==0:
            return []
        id += [pos[0]]
        flag = 1
    else:
        if len(neg)==0:
            return []
        id += [neg[0]]
        flag = -1

    while len(pos)!=0 or len(neg)!=0:
        if flag == -1:
            for i in range(len(pos)):
                if pos[i] > id[-1]:
                    id += [pos[i]]
                    pos = pos[i+1:]
                    flag = 1
                    break
            if flag != 1:
                break
        if flag == 1:
            for i in range(len(neg)):
                if neg[i] > id[-1]:
                    id += [neg[i]]
                    neg = neg[i+1:]
                    flag = -1
                    break
            if flag != -1:
                break
    # print(l,dff,pos,neg,id)
    return id


def findRanges(nums):
    ranges = sum((list(t) for t in zip(nums, nums[1:]) if t[0]+1 != t[1]), [])
    iranges = iter(nums[0:1] + ranges + nums[-1:])
    return [[n,next(iranges)] for n in iranges]


def extractText(img,dict,islabel=False):
    if len(dict)!=0:
        img = img[dict['y']:dict['y']+dict['h'], dict['x']:dict['x']+dict['w']]
    im = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    im = cv2.erode(im, np.ones((1,1), np.uint8), iterations=1)
    im  = cv2.dilate(im, np.ones((1,1), np.uint8), iterations=1)
    if len(dict)!=0:
        detected_label,boxes,box_centers = detect_text(im, check_slope = 1)
        text_labels = text_recog()
        _,boxes2,box_centers2 = detect_text(im, check_slope = -1)
        text_labels2 = text_recog()
        if(len([int(i) for i in text_labels2 if i.isnumeric()])>len([int(i) for i in text_labels2 if i.isnumeric()])):
            text_labels = text_labels2
            box_centers = box_centers2
            boxes = boxes2
    else:
        detected_label,boxes,box_centers = detect_text(im)
        text_labels = text_recog()
    if islabel:
        text_labels = text_recog()
    box_centers, boxes, text_labels = zip( *sorted(zip(box_centers, boxes, text_labels)) )
    box_centers = np.array(box_centers)/3
    boxes = np.array(boxes)/ 3
    if len(dict)!=0:
        box_centers = box_centers + [dict['x'], dict['y']]
        boxes = boxes + [dict['x'], dict['y']]
    return box_centers, boxes, text_labels


def extractCanvaLeg(graph_img,chart_type,smplescatt = False):
    # Intialization
    IS_MULTI_CHART = False
    img = graph_img.copy()
    box_centers, boxes, text_labels = extractText(graph_img,{})

    '''Canvas Preperation'''
    if chart_type == 'bar':
        imh,imw,_ = img.shape
        chart_dict = {'canvas': {'x': 0, 'y': 0, 'w': imw,'h': imh}}
        canvas_img = segment(img, chart_type)
    else:
        # GET BACKGROUND COLOR AND MAKE IT WHITE
        unique_clr, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
        bg_color = sorted([[cnt,list(unique_clr[i])] for i,cnt in enumerate(counts/sum(counts)) if cnt>=0.01],reverse = True)[0][1]
        if bg_color != [255,255,255]:
            img[(img[:,:,0] == bg_color[0]) & (img[:,:,1] == bg_color[1]) & (img[:,:,2] == bg_color[2])] = [255,255,255]
        # Remove all text regions
        for i in boxes:
            img = cv2.rectangle(img, tuple(i[0]), tuple(i[2]), (255, 255, 255), -1)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        imh,imw = thresh.shape
        # vertical lines removal
        yline=[]
        for i in range(imw):
            rang = findRanges([id for id,it in enumerate(thresh[:,i] == 255) if it ])
            rangdiff = [i[1]-i[0] for i in rang]
            if len(rang)!=0 and max(rangdiff) > 300:
                yline+=[[i,rang[rangdiff.index(max(rangdiff))]]]
        # horizontal lines removal
        xline=[]
        for i in range(imh):
            rang = findRanges([id for id,it in enumerate(thresh[i] == 255) if it ])
            rangdiff = [i[1]-i[0] for i in rang]
            if len(rang)!=0 and max(rangdiff) > 300:
                xline+=[[i,rang[rangdiff.index(max(rangdiff))]]]
        # no restriction on removal of vertical line
        if len(yline)>0:
            l1 = findRanges((np.array(yline)[:,0]).tolist())
            for i in yline:
                 thresh[i[1][0]:i[1][1]+1,i[0]] = [0]*(i[1][1]-i[1][0]+1)
        # restriction on removal of horizontal line
        if len(xline)>0:
            l2 = findRanges((np.array(xline)[:,0]).tolist())
            '''If horizontal lines we have are seperated by equal interval,
            then we consider it only contains gridlines, apart from horizontal line graph
            so list of distances between lines have equal mean and mode, it proves if'''
            t = np.mean(np.array(l2),axis=1)
            t =[int(t[i+1]-t[i]) for i in range(len(t)-1)]
            # moDe = max(set(t), key = t.count)
            if chart_type != 'line' or (len(t)>0 and abs(np.mean(t)-max(set(t), key = t.count))<5):
                for i in xline:
                    thresh[i[0],i[1][0]:i[1][1]+1] = [0]*(i[1][1]-i[1][0]+1)
        # TO get only horizontal line graph so that we can get color of lines
        chart_dict = {'canvas': {'x': 0, 'y': 0, 'w': imw,'h': imh}}
        # print(xline,"\n",yline,"\n",l1,"\n",l2)
        if len(yline)>0 and (len(l1)>0 and l1[0][1]<imw//2):
             # a single intial axis line
            chart_dict['canvas']['x'] = l1[0][1]+3
            if len(l1)>1:# has many vertical boundary lines
                chart_dict['canvas']['w'] = l1[len(l1)-1][0]-l1[0][1]-6
        if len(xline)>0 and (len(l2)>0 and l2[0][1]>imh//2):
            # a single intial axis line
            chart_dict['canvas']['h'] = l2[0][1]-3
            if len(l2)>1:# has horizontal axis line
                chart_dict['canvas']['y'] = l2[0][1]-3
                chart_dict['canvas']['h'] = l2[len(l2)-1][0]-l2[0][1]-6

        # Start
        img[~thresh.astype(bool)] = [255,255,255]
        d = chart_dict['canvas']
        canvas_img = np.ones(img.shape,dtype=np.uint8)*255
        canvas_img[d['y']:d['h']+d['y'],d['x']:d['w']+d['x'],:] = img[d['y']:d['h']+d['y'],d['x']:d['w']+d['x'],:]


    '''Legends and color extraction'''
    # Intialization
    legend_colors = []
    legend_names = []
    graph_pts = []
    scatt_centers = []
    kernel = np.ones((3,3), np.uint8)
    # Thicken objects and extract all unique colors
    unique_clr, counts = np.unique(cv2.erode(canvas_img, kernel, iterations=3).reshape(-1, 3), axis=0, return_counts=True)
    unique_clr = unique_clr[:-1]
    counts = counts[:-1]
    L = sorted([[cnt,list(unique_clr[i])] for i,cnt in enumerate(counts/sum(counts)) if cnt>=0.005],reverse = True)
    if smplescatt:
        L = sorted([[cnt,list(unique_clr[i])] for i,cnt in enumerate(counts/sum(counts)) if cnt>=0.1],reverse = True)
    # If a color occupies more that 70% in the image we realize that chart doesn't have multiple labels of types
    if L[0][0] < 0.7:
        IS_MULTI_CHART = True
    if IS_MULTI_CHART :
        legend_centers = []
        legend_text_id = []
        leg_pts = []
        unique_clr = np.array(L)[:,1].tolist()
        if chart_type == 'line' or chart_type == 'bar':
            for clr in unique_clr:
                if np.any( np.all(graph_img==clr,axis=2) ):
                    mas = np.all(graph_img==clr,axis=2).astype(np.uint8)*255
                    if clr == [0,0,0] or chart_type == 'bar':
                        mas = np.all(canvas_img==clr,axis=2).astype(np.uint8)*255
                    mas = cv2.dilate(mas, kernel, iterations=3)
                    pts = np.argwhere(mas==255)
                    pts = pts[:,::-1]
                    db = DBSCAN(eps=15, min_samples=5).fit(pts)
                    labels = db.labels_
                    if (len(np.unique(labels))>=2 and len(np.unique(labels))<7):
                        lab = min(set(labels), key = list(labels).count)
                        t = np.array([pts[id] for id in range(len(labels)) if labels[id] == lab])
                        text_loc = [id for id,i in enumerate(box_centers) if i[0]>max(t[:,0]) and abs(i[1]-np.mean(t[:,1]))<7]
                        if(len(text_loc)!=0):
                            legend_centers += [np.mean(np.array([pts[id] for id in range(len(labels)) if labels[id] == lab]),axis=0)]
                            legend_text_id += text_loc
                            legend_colors += [clr]
                            leg_pts += [np.concatenate([np.array([pts[id] for id in range(len(labels)) if labels[id] == lab]),np.concatenate([boxes[i] for i in text_loc])])]
                            graph_pts += [np.array([pts[id] for id in range(len(labels)) if labels[id] != lab])]
            if len(leg_pts) == 0:
                IS_MULTI_CHART = False
            else:
                leg_pts = np.concatenate(leg_pts).astype(int)
                graph_pts = np.concatenate(graph_pts).astype(int)
        elif chart_type == 'scatter':
            for clr in unique_clr:
                if np.any( np.all(graph_img==clr,axis=2)):
                    mas = np.all(graph_img==clr,axis=2).astype(np.uint8)*255
                    if clr == [0,0,0] :
                        mas = np.all(canvas_img==clr,axis=2).astype(np.uint8)*255
                    pts = CmpuCntr(mas)
                    if len(pts)!=0:
                        text_loc = [[id,t] for id,i in enumerate(box_centers) for t in pts if ((i[0]-t[0])>0 and (i[0]-t[0])<100) and abs(i[1]-t[1])<7]
                        if(len(text_loc)!=0):
                            for i in np.unique(np.array(text_loc)[:,1]).tolist():
                                pts.remove(i)
                            if(len(pts)!=0):
                                legend_centers += [np.mean(np.array(text_loc)[:,1].tolist(),axis=0).tolist()]
                                legend_text_id += np.unique((np.array(text_loc)[:,0])).tolist()
                                legend_colors += [clr]
                                graph_pts += [pts]
            if len(legend_centers) == 0:
                IS_MULTI_CHART =False
            else:
                leg_pts = np.concatenate([[[j[0]-7,j[1]-7],[j[0]+7,j[1]-7],[j[0]+7,j[1]+7],[j[0]-7,j[1]+7]] for j in legend_centers]+[boxes[i].tolist() for i in legend_text_id] ).astype(int)
                scatt_centers = graph_pts
                graph_pts = np.concatenate(graph_pts).astype(int)

            # kernel = np.ones((3,3), np.uint8)
            # skel_image = np.zeros((imh,imw),dtype = np.uint8)
            # for j in unique_clr[1:2]:
            #     maskimg  = np.all(graph_img== j ,axis=2).astype(np.uint8)*255
            #     maskimg  = cv2.dilate(maskimg, kernel, iterations=3)
            #     skel_image = cv2.bitwise_or(skel_image, maskimg, mask = None)
            # img = graph_img.copy()
            # graph_img[skel_image!=255] = [255,255,255]
            # cv2.imwrite("/Users/daggubatisirichandana/Desktop/3.png",graph_img)


        if IS_MULTI_CHART:
            chart_dict['legend'] = {'x': min(leg_pts[:,0])-2, 'y': min(leg_pts[:,1])-2, 'w': max(leg_pts[:,0]) - min(leg_pts[:,0])+4,'h': max(leg_pts[:,1]) - min(leg_pts[:,1])+4}
            legend_text_id = sorted(np.unique(legend_text_id))
            if chart_type == 'scatter':
                legend_centers, legend_colors, scatt_centers = zip( *sorted(zip(np.array(legend_centers).tolist(), legend_colors, scatt_centers)) )
            else:
                legend_centers, legend_colors= zip(*sorted(zip(np.array(legend_centers).tolist(), legend_colors)))
            for lc in legend_centers[::-1]:
                l0 = [i for i in legend_text_id if lc[0]<box_centers[i][0] and abs(lc[1]-box_centers[i][1])<7]
                legend_names += [' '.join([text_labels[i] for i in l0])]
                legend_text_id = list(set(legend_text_id) - set(l0))
            legend_names = legend_names[::-1]

    if not IS_MULTI_CHART :
        legend_colors = [L[0][1]]
        legend_names = ['_']
        mas = np.all(graph_img==legend_colors[0],axis=2).astype(np.uint8)*255
        if legend_colors[0] == [0,0,0] or chart_type == 'bar' or chart_type == 'scatter':
            mas = np.all(canvas_img==legend_colors[0],axis=2).astype(np.uint8)*255
        if chart_type == 'scatter':
            scatt_centers = [CmpuCntr(mas)]
        mas = cv2.dilate(mas, kernel, iterations=3)
        graph_pts = np.array(np.argwhere(mas==255))[:,::-1]
    chart_dict['canvas'] = {'x': min(graph_pts[:,0])-2, 'y': min(graph_pts[:,1])-2, 'w': max(graph_pts[:,0]) - min(graph_pts[:,0])+4,'h': max(graph_pts[:,1]) - min(graph_pts[:,1])+4}
    if chart_type == 'scatter' :
        if IS_MULTI_CHART :
            chart_dict['canvas'] = {'x': min(graph_pts[:,0])-5, 'y': min(graph_pts[:,1])-5, 'w': max(graph_pts[:,0]) - min(graph_pts[:,0])+10,'h': max(graph_pts[:,1]) - min(graph_pts[:,1])+10}
        return  chart_dict, IS_MULTI_CHART, legend_colors, legend_names, bg_color, canvas_img, list(scatt_centers)
    return  chart_dict, IS_MULTI_CHART, legend_colors, legend_names, bg_color, canvas_img


def viewSegChart(img,chart_dict,):
    # # Display segmented components of chart
    for i in chart_dict:
        d = chart_dict[i]
        img = cv2.rectangle(img, (d['x'],d['y']), (d['w']+d['x'],d['h']+d['y']), (255, 0, 255), 2)
        img = cv2.putText(img, i, (d['x'],d['y']), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return img


def extractLablTitl(graph_img, chart_dict, IS_MULTI_CHART):
    '''Components extraction'''
    img = graph_img.copy()
    d = chart_dict['canvas']
    img = cv2.rectangle(img, (d['x'],d['y']), (d['w']+d['x'],d['h']+d['y']), (255, 255, 255), -1)
    if IS_MULTI_CHART:
        d = chart_dict['legend']
        img = cv2.rectangle(img, (d['x'],d['y']), (d['w']+d['x'],d['h']+d['y']), (255, 255, 255), -1)
    box_centers, boxes, text_labels = extractText(img,{})
    y_title = '_'
    l0 = [i for i in range(len(box_centers)) if box_centers[i][0]<chart_dict['canvas']['x']]
    l1 = max([[j for j in l0 if abs(box_centers[i][0]-box_centers[j][0]) <=7 ] for i in l0], key = len)
    l0 = list(set(l0)-set(l1))
    x =  np.concatenate(np.array([boxes[j] for j in l1]))[:,0]
    l2 = [[j for j in l0 if abs(box_centers[i][0]-box_centers[j][0]) <=7 and (box_centers[i][0]>max(x) or box_centers[i][0]<min(x))] for i in l0]
    if(len(l2)!=0 and len(l2[0])!=0):
        l2 = max(l2, key = len)
        if(box_centers[l1[0]][0]<box_centers[l2[0]][0]):
            l0 = l1
            l1 = l2
            l2 = l0
        l2 = np.array([boxes[j] for j in l2])
        chart_dict['y-title'] = {'x': int(np.amin(l2[:,:,0]))-2, 'y': int(np.amin(l2[:,:,1]))-2, 'w': int(np.amax(l2[:,:,0]))-int(np.amin(l2[:,:,0]))+4, 'h':  int(np.amax(l2[:,:,1]))-int(np.amin(l2[:,:,1]))+4}
        cnte, _, text_labels = extractText(img,chart_dict['y-title'])
        _, text_labels = zip(*sorted(zip(cnte[:,1],text_labels),reverse=True))
        y_title= ' '.join([j for j in text_labels if j!=' ' and j!=''])
        # y_title= ' '.join([text_labels[j] for j in l2 if text_labels[j]!=' ' and text_labels[j]!=''])
        img = cv2.rectangle(img, (chart_dict['y-title']['x'],chart_dict['y-title']['y']), (chart_dict['y-title']['x']+chart_dict['y-title']['w'],chart_dict['y-title']['y']+chart_dict['y-title']['h']), (255, 255, 255), -1)
    # Ylabel = [text_labels[j] for j in l1]
    # ybox_centers = np.array([box_centers[j] for j in l1])
    l1 = np.array([boxes[j] for j in l1])
    chart_dict['y-labels'] = {'x': int(np.amin(l1[:,:,0]))-2, 'y': int(np.amin(l1[:,:,1]))-2, 'w': int(np.amax(l1[:,:,0]))-int(np.amin(l1[:,:,0]))+4, 'h':  int(np.amax(l1[:,:,1]))-int(np.amin(l1[:,:,1]))+4}
    ybox_centers, _, Ylabel  = extractText(img,chart_dict['y-labels'],islabel=True)
    img = cv2.rectangle(img, (chart_dict['y-labels']['x'],chart_dict['y-labels']['y']), (chart_dict['y-labels']['x']+chart_dict['y-labels']['w'],chart_dict['y-labels']['y']+chart_dict['y-labels']['h']), (255, 255, 255), -1)


    '''To get title, X-labels, X-title'''
    box_centers, boxes, text_labels = extractText(img,{})

    title = '_'
    l1 = [i for i in range(len(box_centers)) if box_centers[i][1]<chart_dict['canvas']['y']]
    l1 = [i for i in l1 if abs(box_centers[i][1]-box_centers[l1[-1]][1])<5]
    if(len(l1)!=0):
        title= ' '.join([text_labels[j] for j in l1 if text_labels[j]!=' ' and text_labels[j]!=''])
        l1 = np.array([boxes[j] for j in l1])
        chart_dict['title'] = {'x': int(np.amin(l1[:,:,0]))-2, 'y': int(np.amin(l1[:,:,1]))-2, 'w': int(np.amax(l1[:,:,0]))-int(np.amin(l1[:,:,0]))+4, 'h':  int(np.amax(l1[:,:,1]))-int(np.amin(l1[:,:,1]))+4}
    x_title = '_'
    k = chart_dict['canvas']['y']+chart_dict['canvas']['h']
    l0 = [i for i in range(len(box_centers)) if box_centers[i][1]>k]
    l1 = [i for i in l0 if abs(box_centers[i][1]-box_centers[l0[-1]][1])<5]
    l2 = list(set(l0)-set(l1))
    l2 = [i for i in l2 if abs(box_centers[i][1]-box_centers[l2[-1]][1])<5]
    l1 = np.array([boxes[j] for j in l1])
    chart_dict['x-labels'] = {'x': int(np.amin(l1[:,:,0]))-2, 'y': int(np.amin(l1[:,:,1]))-2, 'w': int(np.amax(l1[:,:,0]))-int(np.amin(l1[:,:,0]))+4, 'h':  int(np.amax(l1[:,:,1]))-int(np.amin(l1[:,:,1]))+4}
    xbox_centers, _, Xlabel  = extractText(img,chart_dict['x-labels'],islabel=True)
    if(len(l2)!=0):
        l2 = np.array([boxes[j] for j in l2])
        chart_dict['x-title'] = {'x': int(np.amin(l2[:,:,0]))-2, 'y': int(np.amin(l2[:,:,1]))-2, 'w': int(np.amax(l2[:,:,0]))-int(np.amin(l2[:,:,0]))+4, 'h':  int(np.amax(l2[:,:,1]))-int(np.amin(l2[:,:,1]))+4}
        _, _, text_labels = extractText(img,chart_dict['x-title'])
        x_title= ' '.join([j for j in text_labels if j!=' ' and j!=''])

    return title, y_title, np.array(ybox_centers), list(Ylabel), x_title, np.array(xbox_centers), list(Xlabel)


def findPie(img):
    box_centers, boxes, text_labels = extractText(img,{})
    txt_clust = DBSCAN(eps=32, min_samples=2).fit(box_centers).labels_
    txt_clust_centers = []
    for i in box_centers.astype(int):
        cv2.circle(img, tuple(i), 2, [255,0,0], -1)
    for i in set(txt_clust)-{-1}:
        txt_clust_centers+=[np.mean([box_centers[j] for j in range(len(txt_clust)) if txt_clust[j]==i],axis=0).astype(int).tolist()]
        # cv2.circle(img, tuple(txt_clust_centers[i]), 5, [0,0,0], -1)
    # edges = cv2.Canny(img,100,200)
    # Remove all text regions
    img2 = img.copy()
    for i in boxes:
        img2 = cv2.rectangle(img2, tuple(i[0]), tuple(i[2]), (255, 255, 255), -1)
        # edges = cv2.rectangle(edges, tuple(i[0]), tuple(i[2]), (0), -1)
    # cv2.imshow("orig",img)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    canvas_img = np.zeros(gray.shape,dtype=np.uint8)
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(cv2.blur(gray, (3, 3)),cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                   param2 = 30, minRadius = 1, maxRadius = 5000)
    pt = np.uint16(np.around(detected_circles[0][0]))
    a, b, r = pt[0], pt[1], pt[2]
    # Draw the circumference of the circle.
    gray = np.zeros(gray.shape,dtype=np.uint8)
    cv2.circle(gray, (a, b), r-5, 255, -1)
    # edges[gray==0] = 0
    cv2.circle(gray, (a, b), (2*r)//3, 0, -1)
    canvas_img = img2.copy()
    canvas_img[gray==0] = [255,255,255]
    kernel = np.ones((3,3), np.uint8)
    unique_clr, counts = np.unique(canvas_img.reshape(-1, 3), axis=0, return_counts=True)
    legend_centers = []
    legend_text_id = []
    used_clst_id= []
    legend_names = []
    legend_colors1 = []
    legend_colors2 = []
    data1 = []
    data2 = []
    unique_clr = [[cnt,list(unique_clr[i])] for i,cnt in enumerate(counts) if cnt>=10]
    for pix_cnt,clr in sorted(unique_clr)[::-1][1:]:
        # print("The pix count of color ",clr," is: ",pix_cnt)
        if np.any( np.all(img==clr,axis=2) ):
            mas = np.all(img==clr,axis=2).astype(np.uint8)*255
            #If text present in sector so we dilate , But this doesn't come to pix count
            mas = cv2.dilate(mas, kernel, iterations=3)
            pts = np.argwhere(mas==255)[:,::-1]
            db = DBSCAN(eps=15, min_samples=5).fit(pts)
            labels = db.labels_

            if( set(labels)=={0,1} or set(labels)=={0, 1, -1} ):
                lab = 1
                t = np.array([pts[id] for id in range(len(labels)) if labels[id] == lab])
                pt = np.mean(t, axis = 0)
                # If the cluster is sector of pie, but we need a legend cluster
                if np.sqrt((a-pt[0])**2+(b-pt[1])**2) < r:
                    lab = 0
                    t = np.array([pts[id] for id in range(len(labels)) if labels[id] == lab])
                    pt = np.mean(t, axis = 0)
                text_loc = [id for id,i in enumerate(box_centers) if i[0]>pt[0] and abs(i[1]-pt[1])<7]
                if(len(text_loc)!=0):
                    if text_loc in legend_text_id:
                        data1[legend_text_id.index(text_loc)] += pix_cnt
                    else :
                        legend_centers += [np.mean(np.array([pts[id] for id in range(len(labels)) if labels[id] == lab]),axis=0)]
                        legend_colors1 += [clr]
                        data1 += [pix_cnt]
                        legend_text_id += [text_loc]

            elif( set(labels)=={0} or set(labels)=={0, -1} ):
                pts = np.array([pts[id] for id in range(len(labels)) if labels[id] == 0]).tolist()
                l0 = []
                for id,tc in enumerate(txt_clust_centers):
                    if tc in pts:
                        l0 += [j for j in range(len(txt_clust)) if id not in used_clst_id and txt_clust[j]==id]
                        used_clst_id += [id]
                if len(l0) != 0:
                    legend_colors2 += [clr]
                    data2 += [pix_cnt]
                    legend_names += [' '.join(np.array(sorted([[box_centers[i][1],box_centers[i][0],text_labels[i]] for i in l0]))[:,2])]
                elif set(labels)=={0}:
                    cv2.circle(mas, (a, b), int(0.8*r), [0,0,0], -1)
                    pt = np.mean(np.argwhere(mas==255)[:,::-1], axis = 0)
                    # cv2.circle(mas, (int(pt[0]),int(pt[1])), 10, 127, -1)
                    # cv2.imshow("t",mas)
                    # cv2.waitKey(0)
                    t = [[np.sqrt((tc[0]-pt[0])**2+(tc[1]-pt[1])**2),id] for id,tc in enumerate(txt_clust_centers) if id not in used_clst_id]
                    t.sort()
                    if len(t)!=0:
                        id = int(t[0][1])
                        l0 = [j for j in range(len(txt_clust)) if id not in used_clst_id and txt_clust[j]==id]
                        used_clst_id += [id]
                        legend_names += [' '.join(np.array(sorted([[box_centers[i][1],box_centers[i][0],text_labels[i]] for i in l0]))[:,2])]
                        legend_colors2 += [clr]
                        data2 += [pix_cnt]

    if len(data1)>len(data2):
        legend_names = []
        legend_text_id = list(set(sum(legend_text_id, [])))
        Chart_Title = ' '.join([text_labels[i] for i in range(len(box_centers)) if abs(b-r)>box_centers[i][1] and i not in legend_text_id])
        del_id = []
        legend_text_id = sorted(np.unique(legend_text_id))
        legend_centers, legend_colors1, data1 = zip( *sorted(zip(np.array(legend_centers).tolist(), legend_colors1, data1)) )
        for j,lc in enumerate(legend_centers[::-1]):
            l0 = [i for i in legend_text_id if lc[0]<box_centers[i][0] and abs(lc[1]-box_centers[i][1])<7]
            if len(l0)!=0:
                l0.sort()
                legend_names += [' '.join([text_labels[i] for i in l0])]
            else:
                del_id += [len(data1)-j-1]
            legend_text_id = list(set(legend_text_id) - set(l0))
        legend_names = legend_names[::-1]
        data1= np.delete(np.array(data1), del_id)
        legend_colors1 = np.delete(np.array(legend_colors1), del_id, axis=0)
        return np.array(data1), legend_names, np.array(legend_colors1)[:,::-1]/255, Chart_Title
    else :
        Chart_Title = ' '.join([text_labels[i] for i in range(len(box_centers)) if (abs(b-r)>box_centers[i][1] or abs(b+r)<box_centers[i][1]) and txt_clust[i] not in used_clst_id])
        return np.array(data2), legend_names, np.array(legend_colors2)[:,::-1]/255, Chart_Title


def CmpuCntr(thresh) :
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    bg_fill = np.zeros_like(thresh)
    area_centers = []
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
                area_centers = area_centers + [[cv2.contourArea(c),[cX,cY],c]]
    area_centers = np.array(area_centers)
    # print(area_centers)

    # To get the contour area of single scatter point(CASS)
    if len(area_centers)!=0:
        hist=np.histogram(area_centers[:,0])
        id = np.where(hist[0] == np.amax(hist[0]))[0][0]
        cass = np.mean([i for i in area_centers[:,0] if i>=hist[1][id] and i<=hist[1][id+1]])
        # print(cass)
        # if area of scatter point is greater than cass then points are overlapping
        centers = []
        for i in range(len(area_centers)) :
            nopts = int((area_centers[i][0]/cass)+0.85)
            # print(area_centers[i][0],cass,area_centers[i][0]/cass,nopts,"_____________________________________________________")
            # round numbers to it's next value if it's decimal val is greater than 0.25
            if nopts>1:
                mask = np.zeros_like(thresh)
                cv2.fillPoly(mask, pts = [area_centers[i][2]], color=255)
                pts = np.where(mask == 255)
                kmeans = KMeans(n_clusters=nopts, random_state=0).fit(np.array(pts).T)
                centers = centers + kmeans.cluster_centers_.astype(int)[:,::-1].tolist()
            else :
                centers = centers + [list(area_centers[i][1])]
        return centers
    else:
        return []
