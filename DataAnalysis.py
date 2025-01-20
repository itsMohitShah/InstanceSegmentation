import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.FileWalker import FileWalker
from tqdm import tqdm
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D



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




def inspect_segmentationmask():
    dict_numofinstances = {}
    path_imgwithmask = 'ImageswithSegmentationMask'
    os.makedirs(path_imgwithmask,exist_ok=True)
    for idx, _ in tqdm(enumerate(mask_FW.id),total=len(mask_FW.id)):
        mask = np.load(mask_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])

        # print(mask.shape)
        # print(bbox3d.shape)
        # print(pc.shape)
        # print(rgb.shape)
        numofinstances = mask.shape[0]
        try:
            dict_numofinstances[numofinstances] += 1
        except:
            dict_numofinstances[numofinstances] = 1
        mask_img = np.zeros_like(rgb)

        for dim in range(mask.shape[0]):
            x = np.where(mask[dim]==True)
            mask_img[x] = colors[dim%len(colors)]

        plt.figure(figsize=(15,15))
        plt.subplot(2,1,1)
        plt.imshow(rgb)
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.imshow(mask_img)
        plt.text(40,40,f'Number of instances: {numofinstances}',color='white',fontsize=15, backgroundcolor='gray')
        plt.axis('off')
        plt.savefig(os.path.join(path_imgwithmask,f'ID{str(idx).zfill(3)}--RGBMask--{numofinstances}instances.png'), bbox_inches='tight')
        plt.close()
        
    # print(dict_numofinstances)
    bars = plt.bar(dict_numofinstances.keys(),dict_numofinstances.values(), edgecolor='white')
    plt.title("Distribution of Number of Instances")
    plt.xlabel("Number of instances in an image")
    plt.ylabel("Number of occurences")
    plt.bar_label(bars, fontsize=10, color='black')
    plt.savefig('Distribution_of_Number_of_Instances.png')
    print("Execution complete")



def inspect_bbox3d():
    path_imgwithbbox = 'ImageswithBbox3D'
    os.makedirs(path_imgwithbbox,exist_ok=True)
    for idx, _ in tqdm(enumerate(bbox3d_FW.id),total=len(bbox3d_FW.id)):
        bbox3d = np.load(bbox3d_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])

        print(bbox3d.shape)

        # Set up 3D plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot each bounding box
        # for bbox in bbox3d:
        #     x, y, z = bbox[:, 0], bbox[:, 1], bbox[:, 2]
        #     ax.scatter(x, y, z, label='Bounding Box Corners', alpha=0.8)
            
        #     # Connect corners to visualize the box
        #     for i in range(8):
        #         for j in range(i + 1, 8):
        #             ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'gray', alpha=0.5)

        # # Label axes
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title("3D Bounding Boxes")

        # plt.show()

        fig,ax = plt.subplots(figsize=(10,10))
        ax.imshow(rgb)
        for bbox in bbox3d:
            x,y = bbox[:,0],bbox[:,1]
            for i in range(8):
                for j in range(i+1,8):
                    ax.plot([x[i],x[j]],[y[i],y[j]],alpha=0.5)

        ax.set_title("3D Bounding Boxes")
        plt.axis('off')
        plt.show()

        break
    print("Execution complete")


def inspect_pc():
    path_imgwithpc = 'ImageswithPointCloud'
    os.makedirs(path_imgwithpc,exist_ok=True)
    for idx, _ in tqdm(enumerate(pc_FW.id),total=len(pc_FW.id)):
        pc = np.load(pc_FW.roots[idx])
        mask = np.load(mask_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])

        numofinstances = mask.shape[0]
        # pc_t = pc.transpose(1,2,0)
        plt.figure(figsize=(15,15))
        plt.tight_layout()
        plt.subplot(2,3,(1,3))
        plt.imshow(rgb)
        plt.text(40,40,f'Number of instances: {numofinstances}',color='white',fontsize=15, backgroundcolor='gray')
        plt.axis('off')        
        
        plt.subplot(2,3,4)
        plt.imshow(pc[0,:,:])
        plt.axis('off')

        plt.subplot(2,3,5)
        plt.imshow(pc[1,:,:])
        plt.axis('off')

        plt.subplot(2,3,6)
        plt.imshow(pc[2,:,:])
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join(path_imgwithpc,f'ID{str(idx).zfill(3)}--PC--{numofinstances}instances.png'), bbox_inches='tight')
        plt.close()




        
    print("Execution complete")

if __name__ == "__main__":
    mask_FW = FileWalker('dl_challenge','mask.npy') # Segmentation Mask (number of instances, H, W)
    bbox3d_FW = FileWalker('dl_challenge','bbox3d.npy')
    pc_FW = FileWalker('dl_challenge','pc.npy')
    rgb_FW = FileWalker('dl_challenge','rgb.jpg')

    # inspect_segmentationmask()
    inspect_bbox3d()
    # inspect_pc()

