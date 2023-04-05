import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi,distance
import seaborn as sns
import argparse

def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0-y1
    b = x1-x0
    c = x0*y1-x1*y0
    return a, b, c 

# Line intersection
def get_line_cross_point(line1, line2): 
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0*b1-a1*b0
    if D==0:
        return None
    x = (b0*c1-b1*c0)/D
    y = (a1*c0-a0*c1)/D

    l1xmax,l1xmin,l2xmax,l2xmin=max(line1[::2]),min(line1[::2]),max(line2[::2]),min(line2[::2])
    l1ymax,l1ymin,l2ymax,l2ymin=max(line1[1::2]),min(line1[1::2]),max(line2[1::2]),min(line2[1::2])

    if(x<=l1xmax and x>=l1xmin and x>=l2xmin and x<=l2xmax and y<=l1ymax and y>=l1ymin and y>=l2ymin and y<=l2ymax):
        return True
    else:
        return None

# Resursion # Find suitable threshold
def goodmatch(imgl,imgr,thresh,sig):
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=thresh,sigma=sig)
    kp1, des1 = sift.detectAndCompute(imgl,None)
    kp2, des2 = sift.detectAndCompute(imgr,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    if(len(good)<=20 and len(good)>=16):
        return len(good),kp1,kp2,good,thresh,sig
    elif(len(good)==0):
        return goodmatch(imgl,imgr,0.0001,sig)
    elif(len(good)<16):
        return goodmatch(imgl,imgr,thresh*0.995,sig)
    elif(len(good)>20):
        return goodmatch(imgl,imgr,thresh+0.001,sig)

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))

        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

def sift_ransac(imgl,imgr,height,width):
    points=[]
    disp=[]
    cut_w=3
    cut_h=6
    square_h=h//cut_h
    square_w=w//cut_w
    num_saver=[]
    Sigma_save=1.0
    for i in range(cut_h):
        for j in range(cut_w):
            img1=imgl[i*square_h:(i+1)*square_h,j*square_w:(j+1)*square_w]
            img2=imgr[i*square_h:(i+1)*square_h,j*square_w:(j+1)*square_w]
            num,kp1,kp2,good,Thres_save,Sigma_save=goodmatch(img1,img2,0.001,Sigma_save)
            num_saver.append(num)
            for m in good:
                readpt1 = [j*square_w+int(kp1[m.queryIdx].pt[0]),i*square_h+int(kp1[m.queryIdx].pt[1])]
                readpt2 = [j*square_w+int(kp2[m.trainIdx].pt[0]),i*square_h+int(kp2[m.trainIdx].pt[1])]
                if not (any(elem==readpt1 for elem in points)):
                    points.append(readpt1)
                    disp.append(readpt1[0]-readpt2[0])
    return points,disp,num_saver

if __name__ == '__main__':

    image1 = cv2.imread('./Midd(Eval3)/trainingF/Jadeplant/im0.png',0)
    image2 = cv2.imread('./Midd(Eval3)/trainingF/Jadeplant/im1.png',0)
    h = image1.shape[0]
    w = image1.shape[1]
    interval=0.00116
    points,disp,numofpt=sift_ransac(image1,image2,h,w)
    for i in range(len(points)):
        cv2.circle(image1,(points[i][0],points[i][1]),10,(255, 0, 0),-1)
    cv2.imwrite('temp.png', image1)

    points=np.array(points)
    vor = Voronoi(points)

    ridge_vertices_list=[]
    for pt1,pt2 in vor.ridge_vertices:
        if(pt1!=-1 and pt2!=-1):
            ridge_vertices_list.append([pt1,pt2])

    # Connect Neighbor point and Judge consistency
    region_consist=np.zeros(len(points))
    DDL_saver=np.zeros(len(points))

    for idx_1, idx_2 in vor.ridge_points:
        times=0
        line1=[points[idx_1][0], points[idx_1][1],points[idx_2][0], points[idx_2][1]]
        xs = [points[idx_1][0], points[idx_2][0]]
        ys = [points[idx_1][1], points[idx_2][1]]
        for v_idx_1 , v_idx_2 in ridge_vertices_list:
            line2=[vor.vertices[v_idx_1][0],vor.vertices[v_idx_1][1],vor.vertices[v_idx_2][0],vor.vertices[v_idx_2][1]]
            delta_disp=abs(disp[idx_1]-disp[idx_2])
            if (get_line_cross_point(line1,line2)):
                times=times+1
        if(times<=1):
            plt.plot(xs, ys, color="black")
            if(delta_disp<1): ##Consist
                region_consist[idx_1]=1
                region_consist[idx_2]=1
        if(times==2):
            plt.plot(xs, ys,color="black",linestyle="--")

    regions, vertices = voronoi_finite_polygons_2d(vor)

    area_saver=np.zeros(len(points))
    pts = MultiPoint([Point(i) for i in points])
    mask = pts.convex_hull
    new_vertices = []

    # Compute DDL

    for idx_1, idx_2 in vor.ridge_points:
        times=0
        line1=[points[idx_1][0], points[idx_1][1],points[idx_2][0], points[idx_2][1]]
        xs = [points[idx_1][0], points[idx_2][0]]
        ys = [points[idx_1][1], points[idx_2][1]]
        for v_idx_1 , v_idx_2 in ridge_vertices_list:
            line2=[vor.vertices[v_idx_1][0],vor.vertices[v_idx_1][1],vor.vertices[v_idx_2][0],vor.vertices[v_idx_2][1]]
            if (get_line_cross_point(line1,line2)):
                times=times+1
        if(times<=1):
            delta_disp=abs(disp[idx_1]-disp[idx_2])
            delta_dist=distance.euclidean(points[idx_1], points[idx_2])
            if(region_consist[idx_1]==1 or region_consist[idx_2]==1):
                DDL_saver[idx_1]+=delta_disp/delta_dist
                DDL_saver[idx_2]+=delta_disp/delta_dist

    # Color Voronoi diagram # Compute region area # Sum the DDL

    DDL_val=0
    cur_DDL_saver=np.zeros(len(points))
    sum_area=0
    count_area=0
    sum_dispperpix=0
    count_dispperpix=0
    max_dispperpix=0
    for i in range(len(regions)):
        polygon = vertices[regions[i]]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
        new_vertices.append(poly)
        area_saver[i]=Polygon(poly).area
        if(region_consist[i]==1): ## Consist 
            count_area+=1
            sum_area+=area_saver[i]
            count_dispperpix+=1
            sum_dispperpix+=DDL_saver[i]
            DDL_val+=area_saver[i]*DDL_saver[i]
            cur_DDL_saver[i]=( area_saver[i]*DDL_saver[i] ) / (w*h)
            plt.fill(*zip(*poly), alpha=0.1)
        else:
            plt.fill(*zip(*poly), alpha=0.2)


    hist_DDL=[]
    for val in cur_DDL_saver:
        if(val!=0):
            hist_DDL.append(val)

    hist_DDL=np.array(hist_DDL)
    all=np.zeros(8)
    msk=[(el//interval<8) or (el//interval==8 and el%interval==0) for el in hist_DDL]
    sum_DDL=np.sum(hist_DDL[msk])
    for i in range(len(all)):
        msk=[(el//interval<i) or (el//interval==i and el%interval==0) for el in hist_DDL]
        all[i] = sum_DDL- np.sum(hist_DDL[msk])
    
    for all_el in all:
        print(all_el)

    # Information For Statistics
    
    # print('num of points :',len(hist_DDL))
    # print('DDL_Avg_val : ',np.average(hist_DDL))
    # print('DDL_Std',np.std(hist_DDL))
    # print('DDL_Max',np.max(hist_DDL))
    # print('DDL_min',np.min(hist_DDL))
    # print('Avg_consist_area',(sum_area/count_area)/(w*h))
    # print('Avg_dispperpixel',(sum_dispperpix/count_dispperpix))
    # print('Sum_dispperpixel',(sum_dispperpix))
    # print('Max_dispperpixel',max_dispperpix)

# plt.xlim(0, w) 
# plt.ylim(0, h)
# plt.show()
