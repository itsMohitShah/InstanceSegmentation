import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Utils.FileWalker import FileWalker
from Utils.PointClouderPlotter import *
from Utils.SegmentationImagePlotter import *
from Utils.Open3DPlotter import *


colors = [
    (255, 0, 0),        # Red
    (0, 255, 0),        # Green
    (0, 0, 255),        # Blue
    (255, 255, 0),      # Yellow
    (246, 109, 155),    # Magenta
    (0, 255, 255),      # Cyan
    (128, 0, 0),        # Maroon
    (0, 128, 0),        # Olive
    (149, 97, 226),     # Purple
    (229, 199, 218),    # Pink
    (229, 117, 4),      # Orange
    (175, 52, 45),      # Brown
    (255, 255, 255)     # White
]



def inspect_bbox3d_trial(bbox3d_FW,rgb_FW,pc_FW):
    path_imgwithbbox = 'ImageswithBbox3D--Temp'
    os.makedirs(path_imgwithbbox, exist_ok=True)
    
    for idx, _ in tqdm(enumerate(bbox3d_FW.id), total=len(bbox3d_FW.id)):
        bbox3d = np.load(bbox3d_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])
        pc = np.load(pc_FW.roots[idx])
        numofinstances = pc.shape[0]


        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(rgb)
        img_h, img_w, _ = np.shape(rgb)
        for bbox in bbox3d:
            x, y, z = bbox[:, 0], bbox[:, 1], bbox[:, 2]
            z = np.clip(z, a_min=1e-5, a_max=None)  # Avoid division by zero
            x = (x/z)*(img_w) + (img_w/2)
            y = (y/z)*(img_h) + (img_h/2)
            

            # Define the vertices that compose the bounding box
            vertices = [
                [x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]],
                [x[4], y[4]], [x[5], y[5]], [x[6], y[6]], [x[7], y[7]]
            ]
           
            # Define the 12 edges of the bounding box
            edges = [
                [vertices[0], vertices[1]], [vertices[1], vertices[2]], [vertices[2], vertices[3]], [vertices[3], vertices[0]],
                [vertices[4], vertices[5]], [vertices[5], vertices[6]], [vertices[6], vertices[7]], [vertices[7], vertices[4]],
                [vertices[0], vertices[4]], [vertices[1], vertices[5]], [vertices[2], vertices[6]], [vertices[3], vertices[7]]
            ]
            # Plot the edges
            for edge in edges:
                ax.plot(*zip(*edge), color='r', alpha=1)

        ax.set_title("2D Projection of 3D Bounding Boxes")
        plt.savefig(os.path.join(path_imgwithbbox,f'ID{str(idx).zfill(3)}--3DBox--{numofinstances}instances.png'), bbox_inches='tight')
        # plt.show()
        # print("Displaying 3D Bounding Boxes")
        plt.close()
    print("Execution complete")



# def project_3d_to_2d(points_3d,fx,fy,cx,cy):
#     x,y,z = points_3d[:,0],points_3d[:,1],points_3d[:,2]
#     u = fx * x / z + cx
#     v = fy * y / z + cy
#     return np.stack([u,v],axis=-1)



# def inspect_bbox3d_trial(bbox3d_FW,rgb_FW,pc_FW):
#     path_imgwithbbox = 'ImageswithBbox3D--Temp'
#     os.makedirs(path_imgwithbbox, exist_ok=True)
    
#     for idx, _ in tqdm(enumerate(bbox3d_FW.id), total=len(bbox3d_FW.id)):
#         bbox3d = np.load(bbox3d_FW.roots[idx])
#         rgb = plt.imread(rgb_FW.roots[idx])
#         pc = np.load(pc_FW.roots[idx])
#         numofinstances = pc.shape[0]


#         fig, ax = plt.subplots(figsize=(15, 5))
#         ax.imshow(rgb)
#         img_h, img_w, _ = np.shape(rgb)
#         fx, fy = 1000, 1000
#         cx, cy = img_w/2, img_h/2

#         for bbox in bbox3d:
#             bbox_2d = project_3d_to_2d(bbox,fx,fy,cx,cy)
#             x,y = bbox_2d[:,0],bbox_2d[:,1]
#             ax.plot(np.append(x,x[0]),np.append(y,y[0]), color='r', alpha=1)

#         ax.set_title("2D Projection of 3D Bounding Boxes")
#         # plt.savefig(os.path.join(path_imgwithbbox,f'ID{str(idx).zfill(3)}--3DBox--{numofinstances}instances.png'), bbox_inches='tight')
#         plt.show()
#         print("Displaying 3D Bounding Boxes")
#         plt.close()
#     print("Execution complete")





        
    print("Execution complete")

if __name__ == "__main__":
    mask_FW = FileWalker('dl_challenge','mask.npy') # Segmentation Mask (number of instances, H, W)
    bbox3d_FW = FileWalker('dl_challenge','bbox3d.npy')
    pc_FW = FileWalker('dl_challenge','pc.npy')
    rgb_FW = FileWalker('dl_challenge','rgb.jpg')

    # inspect_segmentationmask(mask_FW, rgb_FW,colors)
    # inspect_bbox3d(bbox3d_FW,rgb_FW,pc_FW)
    # inspect_pc(pc_FW, mask_FW, rgb_FW)

    # inspect_bbox3d_trial(bbox3d_FW,rgb_FW,pc_FW)
    generate_Open3D_pointcloud(pc_FW, mask_FW, bbox3d_FW)