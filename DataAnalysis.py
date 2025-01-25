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








def inspect_bbox3d():
    path_imgwithbbox = 'ImageswithBbox3D'
    os.makedirs(path_imgwithbbox,exist_ok=True)
    for idx, _ in tqdm(enumerate(bbox3d_FW.id),total=len(bbox3d_FW.id)):
        bbox3d = np.load(bbox3d_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])

        print(bbox3d.shape)


        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.imshow(rgb)

        for bbox in bbox3d:
            x, y, z = bbox[:, 0], bbox[:, 1], bbox[:, 2]
            print("Bounding Box:", x)
            print("Bounding Box:", y)
            print("Bounding Box:", z)
            
            # Define the vertices that compose the bounding box
            vertices = [
                [x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]], [x[3], y[3], z[3]],
                [x[4], y[4], z[4]], [x[5], y[5], z[5]], [x[6], y[6], z[6]], [x[7], y[7], z[7]]
            ]
            
            # Define the 12 edges of the bounding box
            edges = [
                [vertices[0], vertices[1]], [vertices[1], vertices[2]], [vertices[2], vertices[3]], [vertices[3], vertices[0]],
                [vertices[4], vertices[5]], [vertices[5], vertices[6]], [vertices[6], vertices[7]], [vertices[7], vertices[4]],
                [vertices[0], vertices[4]], [vertices[1], vertices[5]], [vertices[2], vertices[6]], [vertices[3], vertices[7]]
            ]
            
            # Plot the edges
            for edge in edges:
                ax.plot3D(*zip(*edge), color='r', alpha=0.8)

        ax.set_title("3D Bounding Boxes")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        break
    print("Execution complete")

def inspect_bbox3d_trial():
    path_imgwithbbox = 'ImageswithBbox3D--Temp'
    os.makedirs(path_imgwithbbox, exist_ok=True)
    for idx, _ in tqdm(enumerate(bbox3d_FW.id), total=len(bbox3d_FW.id)):
        bbox3d = np.load(bbox3d_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])
        mask = np.load(mask_FW.roots[idx])
        numofinstances = mask.shape[0]


        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15, 5))
        ax1.imshow(rgb)
        img_h, img_w, _ = np.shape(rgb)
        # plt.axis('off')
        for bbox in bbox3d:
            y, x, z = -bbox[:, 1], bbox[:, 0], bbox[:, 2]
            # print("Bounding Box:", x)
            # print("Bounding Box:", y)
            # print("Bounding Box:", z)
            x = img_w/2 + img_w * x
            y = img_h/2 + img_h * y
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
            print(x[0],y[0])
            # Plot the edges
            for edge in edges:
                ax2.plot(*zip(*edge), color='r', alpha=1)

        ax2.set_title("2D Projection of 3D Bounding Boxes")
        # plt.savefig(os.path.join(path_imgwithbbox,f'ID{str(idx).zfill(3)}--3DBox--{numofinstances}instances.png'), bbox_inches='tight')
        # plt.close()
        plt.show()
        break

    print("Execution complete")





        
    print("Execution complete")

if __name__ == "__main__":
    mask_FW = FileWalker('dl_challenge','mask.npy') # Segmentation Mask (number of instances, H, W)
    bbox3d_FW = FileWalker('dl_challenge','bbox3d.npy')
    pc_FW = FileWalker('dl_challenge','pc.npy')
    rgb_FW = FileWalker('dl_challenge','rgb.jpg')

    inspect_segmentationmask(mask_FW, rgb_FW,colors)
    # inspect_bbox3d()
    inspect_pc(pc_FW, mask_FW, rgb_FW)

    inspect_bbox3d_trial()
