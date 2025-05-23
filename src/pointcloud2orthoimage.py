#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang, Ph.D.(http://www.yuhanjiang.com)
# Date:         6/24/2021
# Discriptions : pointcloud to orthoimage
# Major updata : Automatically align point cloud to a plane surface
import copy
import gc
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import laspy,os # laspy 2.1.2
import numpy as np
from mpl_toolkits import mplot3d
import cv2 as cv # using 3.4.2
from scipy.interpolate import griddata
import pandas as pd # using pandas 1.15.5
import matplotlib.pyplot as plt # 3.3.4
#%matplotlib inline
from itertools import repeat
from multiprocessing import get_context, Process as BaseProcess
from multiprocessing.pool import Pool
import open3d as o3d
import os

class NoDaemonProcess(BaseProcess):
    def __init__(self, *args, **kwargs):
        # Force the first arg (group) to be None
        args = (None,) + args[1:] if len(args) > 0 else args
        kwargs.pop('group', None)
        super().__init__(*args, **kwargs)

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)

class MyPool(Pool):
    def __init__(self, *args, **kwargs):
        ctx = get_context("spawn")
        super().__init__(*args, context=ctx, **kwargs)

    Process = NoDaemonProcess
def rotate(img,angle):
    h, w =img.shape[0:2]
    center = (w/2, h/2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img,M,(w,h))
    return rotated
def newdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        #print(path+'   Successful')
        return True
    else:
        #print(path+'   Exists')
        return False
def generateGridImageUisngMultiCPU(X,Y,Z):
    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))

    grid_x,grid_y=np.mgrid[X.min():X.max():(x_range*1j),Y.min():Y.max():(y_range*1j)]  # create the grid size

    grid_z = griddata((X,Y), Z, (grid_x, grid_y), method='linear')#{‘linear’, ‘nearest’, ‘cubic’}, optional
    try:
        return grid_z
    finally:
        del X,Y,Z,grid_z,grid_x,grid_y
        gc.collect()
def PointCloud2Orthoimage(PCD,downsample=10,GSDmm2px=5):
    print('[PointCloudShape X,Y,Z]',PCD.x.shape,PCD.y.shape,PCD.z.shape)

    if downsample>0:
        X=PCD.x[::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=PCD.z[::downsample]*1000/GSDmm2px  # [::10] downsample 1/10
        Z=PCD.y[::downsample]*1000/GSDmm2px
        #R=(PCD.red[::downsample]/(2**16-1)*255).astype('uint8')  # covert 16-bit color to 8-bit
        #G=(PCD.green[::downsample]/(2**16-1)*255).astype('uint8')
        #B=(PCD.blue[::downsample]/(2**16-1)*255).astype('uint8')
        R=(PCD.red[::downsample])#.astype('uint8')  # keep 16-bit
        G=(PCD.green[::downsample])#.astype('uint8')
        B=(PCD.blue[::downsample])#.astype('uint8')
        print('[DownSamplePCShape]',X.shape,Y.shape,Z.shape)
    else:
        X=PCD.x*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=PCD.z*1000/GSDmm2px  # [::10] downsample 1/10
        Z=PCD.y*1000/GSDmm2px
        #R=(PCD.red[::downsample]/(2**16-1)*255).astype('uint8')  # covert 16-bit color to 8-bit
        #G=(PCD.green[::downsample]/(2**16-1)*255).astype('uint8')
        #B=(PCD.blue[::downsample]/(2**16-1)*255).astype('uint8')
        R=PCD.red  #.astype('uint8')  # keep 16-bit
        G=PCD.green  #.astype('uint8')
        B=PCD.blue#.astype('uint8')

    print("[RGBColorRange]",R,G,B)

    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))
    print("[ImageFrameSize]",x_range,y_range)

    ele_max=Z.max()
    ele_min=Z.min()
    z_range=[ele_min,ele_max]

    print('[x,y,z range in mm]',x_range*GSDmm2px,y_range*GSDmm2px,z_range)

    #print(X.shape,Y.shape,Z.shape)

    #grid_ele = griddata((X,Y), Z, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_R = griddata((X,Y), R, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_G = griddata((X,Y), G, (grid_x, grid_y), method='cubic').astype(np.float)
    #grid_B = griddata((X,Y), B, (grid_x, grid_y), method='cubic').astype(np.float)
    EleRGB=[Z,R,G,B]
    pool=MyPool(4)  #//5+1)
    grid_Mutiple=pool.starmap(generateGridImageUisngMultiCPU,zip(repeat(X),repeat(Y),EleRGB))
    pool.close()

    grid_ele=grid_Mutiple[0].astype('float')
    grid_R=grid_Mutiple[1].astype('uint16')
    grid_G=grid_Mutiple[2].astype('uint16')
    grid_B=grid_Mutiple[3].astype('uint16')

    grid_RGB=np.zeros((grid_ele.shape[0],grid_ele.shape[1],3)).astype('uint16')
    grid_RGB[:,:,0]=grid_R
    grid_RGB[:,:,1]=grid_G
    grid_RGB[:,:,2]=grid_B

    print('[RGB,Ele imageShape]',grid_RGB.shape,grid_ele.shape)
    return grid_RGB,grid_ele,[ele_min,ele_max]
def PointCloud2Orthoimage2(points,colors,downsample=10,GSDmm2px=5):
    print('[PointCloudShape XYZ RGB]',points.shape,colors.shape)
    if downsample>0:
        X=points[:,0][::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=points[:,1][::downsample]*-1000/GSDmm2px  # [::10] downsample 1/10
        if X.max()>Y.max():
            print('[Rotated]')
            X=points[:,1][::downsample]*1000/GSDmm2px  # 1000 means:1mm to 1 px
            Y=points[:,0][::downsample]*-1000/GSDmm2px  # [::10] downsample 1/10
        Z=points[:,2][::downsample]*1000# elevation in mm
        R=(colors[:,0][::downsample])# keep 16-bit
        G=(colors[:,1][::downsample])
        B=(colors[:,2][::downsample])
        print('[DownSamplePCShape]',X.shape,Y.shape,Z.shape)
    else:
        X=points[:,0]*1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=points[:,1]*-1000/GSDmm2px  # [::10] downsample 1/10
        if X.max()>Y.max():
            print('[Rotated]')
            X=points[:,1]*1000/GSDmm2px  # 1000 means:1mm to 1 px
            Y=points[:,0]*-1000/GSDmm2px  # [::10] downsample 1/10
        Z=points[:,2]*1000#elevation in mm
        R=colors[:,0]# keep 16-bit
        G=colors[:,1]
        B=colors[:,2]
    print("[RGBColorRange]",R,G,B)
    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))
    print("[ImageFrameSize]",x_range,y_range)
    ele_max=Z.max()
    ele_min=Z.min()
    z_range=[ele_min,ele_max]
    print('[x,y,z range in mm]',x_range*GSDmm2px,y_range*GSDmm2px,z_range)

    EleRGB=[Z,R,G,B]
    pool=MyPool(4)
    grid_Mutiple=pool.starmap(generateGridImageUisngMultiCPU,zip(repeat(X),repeat(Y),EleRGB))
    pool.close()

    grid_ele=grid_Mutiple[0].astype('float')
    grid_R=grid_Mutiple[1].astype('uint16')
    grid_G=grid_Mutiple[2].astype('uint16')
    grid_B=grid_Mutiple[3].astype('uint16')

    grid_RGB=np.zeros((grid_ele.shape[0],grid_ele.shape[1],3)).astype('uint16')
    grid_RGB[:,:,0]=grid_R
    grid_RGB[:,:,1]=grid_G
    grid_RGB[:,:,2]=grid_B

    print('[RGB,Ele imageShape]',grid_RGB.shape,grid_ele.shape)
    print('[GSD: mm/px]',GSDmm2px)
    try:
        return grid_RGB,grid_ele,[ele_min,ele_max]
    finally:
        del grid_B,grid_G,grid_R,grid_RGB,grid_ele,grid_Mutiple,EleRGB,X,Y,Z,R,G,B,pool,points,colors

def preparedata(point_cloud):
    #import pptk
    points = np.vstack((point_cloud.x, point_cloud.z,point_cloud.y)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green,point_cloud.blue)).transpose()
    normals=0#pptk.estimate_normals(points,k=6,r=np.inf)
    #normals = np.vstack((point_cloud.normalx, point_cloud.normaly,point_cloud.normalz)).transpose()
    return point_cloud,points,colors,normals
# def pptkviz(points,colors):
#     #import pptk
#     v = pptk.viewer(points)
#     v.attributes(colors/65535)
#     v.set(point_size=0.001,bg_color= [0,0,0,0],show_axis=0,
#     show_grid=0)
#     return v
def cameraSelector(v):
    camera=[]
    camera.append(v.get('eye'))
    camera.append(v.get('phi'))
    camera.append(v.get('theta'))
    camera.append(v.get('r'))
    return np.concatenate(camera).tolist()
# def computePCFeatures(points, colors, knn=10, radius=np.inf):
#     #import pptk
#     normals=pptk.estimate_normals(points,knn,radius)
#     idx_ground=np.where(points[...,2]>np.min(points[...,2]+0.3))
#     idx_normals=np.where(abs(normals[...,2])<0.9)
#     idx_wronglyfiltered=np.setdiff1d(idx_ground, idx_normals)
#     common_filtering=np.append(idx_normals, idx_wronglyfiltered)
#     return points[common_filtering],colors[common_filtering]
def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def get_floor_plane(pcd, dist_threshold=0.02, num_iterations=2000,bool_visualize=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=3,
                                             num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    inlier_cloud=pcd.select_by_index(inliers)
    if bool_visualize:
        inlier_cloud.paint_uniform_color([1.0,0,0])
        outlier_cloud=pcd.select_by_index(inliers,invert=True)
        o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud],window_name='FloorPlane_Sidewalk@Red ',width=1920//3*2,height=1080//3*2)
    il_points=np.array(inlier_cloud.points)
    plane_ele_mean=il_points[:,2].mean()
    print('[A SidewalkPlaneRange@center]',il_points[:,2].min(),il_points[:,2].max(),plane_ele_mean)
    try:
        return plane_model,plane_ele_mean
    finally:
        del pcd,inliers,inlier_cloud,il_points
def align_sidewalk_surface(pcd,bool_visualize=False,bool_repeate=True,dist_threshold=0.02, num_iterations=2000):
    downpcd=copy.deepcopy(pcd).voxel_down_sample(voxel_size=0.05)
    floor,plane_ele_mean=get_floor_plane(downpcd,bool_visualize=bool_visualize,dist_threshold=dist_threshold, num_iterations=num_iterations)
    a,b,c,d=floor
    # Translate plane to z-coordinate = 0
    pcd.translate((0,0,-plane_ele_mean))
    # Calculate rotation angle between plane normal & z-axis
    plane_normal=tuple(floor[:3])
    z_axis=(0,0,1)
    rotation_angle=vector_angle(plane_normal,z_axis)
    # Calculate rotation axis
    plane_normal_length=math.sqrt(a**2+b**2+c**2)
    u1=b/plane_normal_length
    u2=-a/plane_normal_length
    rotation_axis=(u1,u2,0)
    # Generate axis-angle representation
    optimization_factor=1  #1.4
    axis_angle=tuple([x*rotation_angle*optimization_factor for x in rotation_axis])
    # Rotate point cloud
    R=pcd.get_rotation_matrix_from_axis_angle(axis_angle)
    pcd.rotate(R,center=(0,0,0))
    if bool_repeate:
        bool_repeate=False
        return align_sidewalk_surface(pcd,bool_visualize=False,bool_repeate=bool_repeate,dist_threshold=0.02*2, num_iterations=2000)
    else:
        try:
            return pcd
        finally:
            del pcd,downpcd

def main2(glb_file_path,pointName='5mm_18_34_56',downsample=10,GSDmm2px=5,bool_alignOnly=False,b='win',bool_generate=False):
    print('$',pointName)
    bool_confirm=False
    if b=='win' or b =='mac':
        #import pptk
        import open3d as o3d
        axis_mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()  #o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=5.0,origin=np.array([0.,0.,0.]))
        PCD=laspy.read(glb_file_path+pointName+".las")
        #region get_the_min_rotated_boundingbox
        point_cloud,points,colors,normals=preparedata(PCD)
        #viewer1=pptkviz(points,colors)#,normals)
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(points)
        pcd.colors=o3d.utility.Vector3dVector(colors/65535)
        #pcd.normals=o3d.utility.Vector3dVector(normals)
        #o3d.visualization.draw_geometries([pcd,axis_mesh],window_name='OrginalPCD_Sidewalk '+pointName,width=1920//3*2,height=1080//3*2)
        #region get floor plane# Get the plane equation of the floor → ax+by+cz+d = 0
        pcd=align_sidewalk_surface(pcd)
        #get num of point > 0, and num of point <0
        zzz=np.array(pcd.points)[:,2]#[::2000]
        print(zzz.max(),zzz.min())
        plane_ele_mean=0.035#.3
        print(np.sum(zzz>plane_ele_mean),np.sum(zzz<-plane_ele_mean))
        if abs(zzz.max())>abs(zzz.min()) and np.sum(zzz>plane_ele_mean)>np.sum(zzz<-plane_ele_mean):
            upmodel=1
        else:
            upmodel=-1
        #o3d.visualization.draw_geometries([pcd,axis_mesh],window_name='FloorPlane_Sidewalk '+pointName,width=1920//3*2,height=1080//3*2)
        #endregion
        #
        #obb=pcd.get_oriented_bounding_box()
        #obb.color=(0,1,0)  #obbBounding box is green
        #center=obb.center
        #extent=obb.extent
        #print('#[BoundingBox]',obb)
        #
        obb=pcd.get_oriented_bounding_box()
        obb.color=(0,1,0)  #obbBounding box is green
        center=obb.center
        extent=obb.extent
        R=np.matrix(obb.R)
        #print(center,extent,R,R.I)
        R=R.I
        #R[:,2]=1
        pcd_r=copy.deepcopy(pcd).rotate(R,center=center)#rotation
        pcd_r=align_sidewalk_surface(pcd_r)
        #get num of point > 0, and num of point <0
        points_r=np.array(pcd_r.points)
        zzz_r=points_r[:,2]#[::2000]
        print(zzz_r.max(),zzz_r.min())
        print(np.sum(zzz_r>plane_ele_mean),np.sum(zzz_r<-plane_ele_mean))
        if abs(zzz_r.max())>abs(zzz_r.min()) and np.sum(zzz_r>plane_ele_mean)>np.sum(zzz_r<-plane_ele_mean):
            upmodel_r=1
        else:
            upmodel_r=-1
        if upmodel+upmodel_r==0:
            back_pcd_r=copy.deepcopy(pcd_r)
            print('[Flipped Z-axis]')
            points_r[:,2]=points_r[:,2]*-1
            points_r[:,1]=points_r[:,1]*-1
            pcd_r.points=o3d.utility.Vector3dVector(points_r)
            # THIS SHOWED THE WINDOW OF A POINTCLOUD IMAGE
            # o3d.visualization.draw_geometries([pcd_r,axis_mesh],window_name='Flipped_Sidewalk '+pointName,width=1920//3*2,height=1080//3*2)  # if only show red box, then the green box is been covered. then the results is correct.
            # bool_confirm=input('Confirm the filp? y/n:')
            # if  not bool_confirm in ['y',"Y"]:
            #     print('Discard filp')
            #     pcd_r=copy.deepcopy(back_pcd_r)
            #     bool_confirm=True
        obb_r=pcd_r.get_oriented_bounding_box()
        #obb_r.color=(0,1,0)  #obbBounding box is green
        center_xy=np.array(obb_r.center)
        center_xy[2]=0
        pcd_t=copy.deepcopy(pcd_r).translate(-center_xy)
        aabb=pcd_t.get_axis_aligned_bounding_box()
        aabb.color=(1,0,0)  #aabb bounding box is red
        obb_t=pcd_t.get_oriented_bounding_box()
        obb_t.color=(0,1,0)  #obbBounding box is green
        print(aabb)
        print(obb_t)
        # if bool_confirm==False:
        #     o3d.visualization.draw_geometries([pcd_t,aabb,obb_t,axis_mesh],window_name='Tanslated_Sidewalk '+pointName,width=1920//3*2,height=1080//3*2)  # if only show red box, then the green box is been covered. then the results is correct.

        if bool_alignOnly and bool_generate:
            print('[Start]---/...')
            #o3d.io.write_point_cloud(glb_file_path+pointName+"aligned.pcd",pcd_t)
            df=np.hstack([np.array(pcd_t.points),np.asarray(pcd_t.colors)])
            df=pd.DataFrame(df)
            df.to_csv(glb_file_path+pointName+"aligned.csv",index=False,header=False)
            print('[Saved]',glb_file_path+pointName+"aligned.csv")

    if b=='server':

        pc=pd.read_csv(glb_file_path+pointName+"aligned.csv",index_col=False,header=None)
        pc=np.array(pc)
        print(['CSV pointcloud formate'],pc.shape)
        points=pc[:,0:3]#*-1
        colors=pc[:,3:6]
        #endregion
    if bool_alignOnly:
        print('[Piontcloud aligment only]')
        return False
    #region ouput
    if b=='win' or b =='mac':
        grid_RGB,grid_ele,(ele_min,ele_max)=PointCloud2Orthoimage2(np.array(pcd_t.points),np.asarray(pcd_t.colors)*65535,downsample=downsample,GSDmm2px=GSDmm2px)  #PointCloud2Orthoimage(PCD,downsample=0,GSDmm2px=5)
    if b=='server':
        grid_RGB,grid_ele,(ele_min,ele_max)=PointCloud2Orthoimage2(np.array(points),np.asarray(colors)*65535,downsample=downsample,GSDmm2px=GSDmm2px)  #PointCloud2Orthoimage(PCD,downsample=0,GSDmm2px=5)
    grid_RGB=(grid_RGB/(2**16-1)*255).astype('uint8')
    grid_map=((grid_ele-ele_min)/(ele_max-ele_min)*255).astype('uint8')
    #if grid_ele.shape[0]>grid_ele.shape[1]:# alway keep width larger than height
    #    grid_ele=rotate(grid_ele,90)
    #    grid_RGB=rotate(grid_RGB,90)
    try:
        return grid_RGB,grid_ele,grid_map,(ele_min,ele_max),GSDmm2px
    finally:
        newdir(glb_file_path+'/Demo/'+pointName+'/')
        cv.imwrite(glb_file_path+'/Demo/'+pointName+'/'+pointName+'RGB.jpg',cv.cvtColor(grid_RGB,cv.COLOR_RGB2BGR),[int(cv.IMWRITE_JPEG_QUALITY),100])
        cv.imwrite(glb_file_path+'/Demo/'+pointName+'/'+pointName+'DEM.jpg',grid_map,[int(cv.IMWRITE_JPEG_QUALITY),100])
        print('[Done]',glb_file_path+'/Demo/'+pointName+'/'+pointName+'RGB/DEM.jpg')
        try:
            del PCD,point_cloud,points,colors,pcd_t,pcd_r
        except:
            del pc,points,colors
        gc.collect()

#-------
import os

def p2o_main(pointcloud_file_path):

    pcFolderPath = pointcloud_file_path  # this file is passed by the function in main.py. It's the path of sidewalk folders
    # The set up the folder should be you have a pointcloud folder that holds .las files inside. 
    # It will also hold another folder that has the elevation, orthoimages
    # Finally, once processed for vertical displacement, the folder will be moved to measured_sidewalks folder

    las_files = [f for f in os.listdir(pcFolderPath) if f.endswith('.las') and os.path.isfile(os.path.join(pcFolderPath, f))]

    # Environment/platform setup
    if os.path.exists('C:/'):
        b = 'win'
        cpu = 3
        import open3d as o3d
    elif os.path.exists('/Users/'):
        b = 'mac'
        cpu = 4
        import open3d as o3d
    elif os.path.exists('/data/'):
        b = 'server'
        cpu = 4

    for las_file in las_files:
        fileName = os.path.splitext(las_file)[0]  
        
        # Skip if it's already processed
        if os.path.exists(os.path.join(pcFolderPath, 'Demo', fileName)) or \
           os.path.exists(os.path.join(pcFolderPath, 'measured_sidewalks', fileName)):
            continue

        glb_file_path = pcFolderPath  # screenshot saving path

        if b == 'win':
            main2(pointName=fileName, glb_file_path=glb_file_path, GSDmm2px=5, bool_alignOnly=0, b=b, bool_generate=0)
        else:
            main2(pointName=fileName, glb_file_path=glb_file_path, GSDmm2px=5, bool_alignOnly=False, b=b)
