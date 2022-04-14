import cv2
import numpy as np
from sklearn.cluster import SpectralClustering
import circle_fit as cf

def fitCircle(X):
    xc,yc,r,s = cf.hyper_fit(X)
    # cf.plot_data_circle(X[:,0], X[:,1], xc, yc, r)
    # plt.show()
    return [int(round(xc)), int(round(yc)), int(round(r))]

def Color_Seg(image, clustno):
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, clustno, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    return segmented_image

def eachClrCntr(img,clr):
    thresh = np.all(img==clr,axis=2).astype(np.uint8)*255
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
                area_centers = area_centers + [[cX,cY]]
    return area_centers

def Compute_BubbPixData(im,obj_clrs):
    clr_centers = []
    colorsegImg = Color_Seg(im, len(obj_clrs)+1)
    for id,i in enumerate(obj_clrs):
        clr_centers += [eachClrCntr(im,i)]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    bg_fill = np.zeros_like(thresh)
    # the bubble params contains center, color, and radius of bubble
    bubble_params = []
    print("Started Bubble detection")
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            mask = np.zeros_like(thresh)
            cv2.fillPoly(mask, pts = [c], color=255)
            # to check if the contours are going to overlap on already VISITED contours
            intersect_img = cv2.bitwise_and(bg_fill, mask, mask = None)
            if not cv2.countNonZero(intersect_img):
                bg_fill = cv2.bitwise_or(bg_fill, mask, mask = None)
                A = set(map(tuple,np.concatenate(clr_centers)))
                B = set(map(tuple,np.argwhere(mask == 255)[:,::-1]))
                intersexnpts = [list(x) for x in A & B]
                Flg = False
                if len(intersexnpts)<=1 :
                    '''NON-OVERLAPPING CLUSTERS - CONTOUR PARAMETERS TO DECTECT CIRCLE'''
                    extLeft = (c[c[:, :, 0].argmin()][0])
                    extRight = (c[c[:, :, 0].argmax()][0])
                    rad = int(np.sqrt(np.sum(np.square(extLeft - extRight)))/2)
                    # THIS CASE ARISES IF ONE OF THE OVERLAPPING POINT IN CLUSTER
                    # IS COMBINATION OF TWO EQUAL SIZE POINTS - GIVING A NEW COLORED POINT
                    if len(intersexnpts) == 1 and np.sqrt((intersexnpts[0][0]-cX)**2+(intersexnpts[0][1]-cY)**2)>10:
                        intersexnpts = intersexnpts+[[cX,cY]]
                        Flg = True
                    else :
                        # ADD COMPUTED CIRCLE PARAMETERS TO CIRCLE LIST
                        bubble_params = bubble_params + [[cX,cY,list(im[cY,cX]),rad]]
                        # # POINT IS COMBINATION OF TWO EQUAL SIZE POINTS - GIVING A NEW COLORED POINT
                        if len(intersexnpts) == 0 :
                            print("FOUND A BUBBLE SCATTER POINT WITH UNDETECTED COLOR")
                if len(intersexnpts) > 1 :
                    '''OVERLAPPING CLUSTERS - CIRCLE REGRESSION TO DECTECT CIRCLE, 
                    PROCESS CIRCLE DETECTION SEPERATELY, ON EACH COLOR CLUSTER'''
                    A = np.argwhere(mask == 255)
                    # print(np.unique(colorsegImg[mask == 255], axis=0))
                    for eachclr in np.unique(colorsegImg[mask == 255], axis=0) :
                        B = np.argwhere(colorsegImg == eachclr)[:,:2]
                        X = np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])[:,::-1]
                        clstrpnt = [x for x in set(tuple(x) for x in X) & set(tuple(x) for x in np.array(intersexnpts))]
                        if len(clstrpnt) == 1 :
                            c1, c2, r = fitCircle(X)
                            # CORRECT RADIUS PRESICION TO CLUSTER BOUNDARY
                            extLeft = X[X[:, 0].argmin()]
                            extRight = X[X[:, 0].argmax()]
                            r1 = int(np.sqrt(np.sum(np.square(extLeft - np.array([c1,c2]) ))))
                            r2 = int(np.sqrt(np.sum(np.square(extRight  - np.array([c1,c2]) ))))
                            r = max(min(r1,r2),r)
                            # ADD COMPUTED CIRCLE PARAMETERS TO CIRCLE LIST
                            unique_clr, counts = np.unique([im[i[1],i[0]] for i in X], axis=0, return_counts=True)
                            L = [list(unique_clr[i]) for i,cnt in enumerate(counts/sum(counts)) if cnt>=0.01 and (list(unique_clr[i]) in obj_clrs)]
                            if len(L) == 1:
                                bubble_params = bubble_params + [[c1, c2, L[0], r]]
                            else:
                                bubble_params = bubble_params + [[c1, c2, list(im[c2,c1]), r]]
                            # if Flg :
                            #     bubble_params = bubble_params + [[c1, c2, list(im[c2,c1]), r]]
                            # else :
                            #     bubble_params = bubble_params + [[c1, c2, list(im[clstrpnt[0][1],clstrpnt[0][0]]), r]]
                        elif len(clstrpnt) > 1:
                            ''' PERFORM LOCATION BASED SPECTRAL CLUSTERING IN CASE OF MULTIPLE CLUSTERS OF SAME COLOR'''
                            clustering = SpectralClustering(n_clusters=len(clstrpnt), assign_labels='discretize', random_state=0).fit(X)
                            y_pred = clustering.labels_
                            UsedSet = set()
                            # ITERATE THRU' SORTED LIST OF ITEMS BASED ON FREQUENCY
                            for i in np.unique(y_pred,return_counts=True)[0][::-1]:
                                # THE REMAINING POINTS LEFT AFTER REMOVING IDENTIFIED CIRCLE CLUSTERS
                                cstrpts = np.array(list( set(map(tuple, X[y_pred == i])).difference(UsedSet) ))
                                c1, c2, r = fitCircle(cstrpts)
                                # CORRECT RADIUS PRESICION TO CLUSTER BOUNDARY
                                extLeft = cstrpts[cstrpts[:, 0].argmin()]
                                extRight = cstrpts[cstrpts[:, 0].argmax()]
                                r1 = int(np.sqrt(np.sum(np.square(extLeft - np.array([c1,c2]) ))))
                                r2 = int(np.sqrt(np.sum(np.square(extRight  - np.array([c1,c2]) ))))
                                r = max(min(r1,r2),r)
                                # ADD COMPUTED CIRCLE PARAMETERS TO CIRCLE LIST
                                unique_clr, counts = np.unique([im[i[1],i[0]] for i in cstrpts], axis=0, return_counts=True)
                                L = [list(unique_clr[i]) for i,cnt in enumerate(counts/sum(counts)) if cnt>=0.01 and (list(unique_clr[i]) in obj_clrs)]
                                if len(L) == 1:
                                    bubble_params = bubble_params + [[c1, c2, L[0], r]]
                                else:
                                    bubble_params = bubble_params + [[c1, c2, list(im[c2,c1]), r]]
                                # if Flg :
                                #     bubble_params = bubble_params + [[c1, c2, list(im[c2,c1]), r]]
                                # else :
                                #     bubble_params = bubble_params + [[c1, c2, list(im[clstrpnt[0][1],clstrpnt[0][0]]), r]]

                                # ADD IDENTIFIED CIRCLE CLUSTERS PTS TO 'UsedSet' LIST
                                mask2 = np.zeros_like(thresh)
                                mask2 = cv2.circle(mask2, (c1, c2), r, 255, -1)
                                UsedSet = UsedSet.union(set(map(tuple, np.argwhere(mask2 == 255))))
    return bubble_params

def findOverlapComb(legend_colors,  bg_color, ovrlap_clrs):
    dupovrlap_clrs = ovrlap_clrs
    ovrlap_sets = []
    x = []
    expctSRC_clr = []
    for id,lc in enumerate(legend_colors):
        # Testing all possible Alpha values in 0.1-0.95
        for a in [i/100 for i in range(10,100,5)]:
            clr = np.divide(np.subtract(lc,np.multiply(1 - a,bg_color)),a).astype(int)
            # To check non negativity
            if min(clr)>=0 :
                expctSRC_clr += [[id, clr.tolist(), a]]
                x += [[[id], lc]]
    SetComb = [x]
    Finalalp = 0
    cnt = 0
    # while len(SetComb)<len(legend_colors) and len(ovrlap_clrs)!=0 :
    while len(SetComb)<3 and len(ovrlap_clrs)!=0 :
        x = []
        for id, ex_src, alp in expctSRC_clr:
            if Finalalp == alp or Finalalp == 0:
                for Oleg, lc in SetComb[cnt]:
                    clr = np.add(np.multiply(alp,ex_src),np.multiply((1 - alp),lc)).astype(int)
                    # remove non neg colors, and round up the floating point values
                    if min(clr)>=0 :
                        x  += [[Oleg+[id], clr]]
                        for indx,i in enumerate(ovrlap_clrs):
                            if (np.array(clr) == np.array(i)).all() :
                                ovrlap_sets += [[ovrlap_clrs.pop(indx),Oleg+[id]]]
                                Finalalp = alp
                                break
        cnt+= 1
        SetComb += [x]

    return ovrlap_sets

def Scale_XYCoords(Ylabel,ybox_centers,Xlabel,xbox_centers,c):
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
    xbox_centers, Xlabel = (list(xbox_centers),list(Xlabel))
    normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
    t = np.array(sorted(np.concatenate((ybox_centers, np.array([Ylabel]).T), axis=1), key=lambda x: x[1]))#, reverse= True))
    ybox_centers,Ylabel = (t[:,0:2],list(t[:,2]))
    normalize_scaley = (Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1])

    c = c.astype(float)
    c[:,0] = ((c[:,0] - xbox_centers[0][0]) * normalize_scalex) + Xlabel[0]
    c[:,1] = ((c[:, 1] - ybox_centers[0][1]) * normalize_scaley) + Ylabel[0]

    return c.round(0).astype(int)



