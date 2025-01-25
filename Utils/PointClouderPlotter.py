import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np

def inspect_pc(pc_FW, mask_FW, rgb_FW):
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
