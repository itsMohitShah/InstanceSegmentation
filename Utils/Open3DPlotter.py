import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
def plot_3d_bboxes(pc_data, bbox_data, visualize=False):
    
    # Reshape point cloud (flatten spatial dimensions)
    x, y, z = pc_data.reshape(3, -1)
    points = np.vstack((x, y, z)).T
    
    # Filter out zero points (assuming zero means no depth info)
    valid_mask = ~np.all(points == 0, axis=1)
    points = points[valid_mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create bounding boxes
    bbox_lines = []
    for bbox in bbox_data:
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
        ]
        bbox_line_set = o3d.geometry.LineSet()
        bbox_line_set.points = o3d.utility.Vector3dVector(bbox)
        bbox_line_set.lines = o3d.utility.Vector2iVector(lines)
        bbox_lines.append(bbox_line_set)
    
    
    
    if visualize:
        o3d.visualization.draw_geometries([pcd] + bbox_lines)
    return pcd

def generate_Open3D_pointcloud(pc_FW, mask_FW, bbox3d_FW):
    path_pc = r'PointClouds'
    os.makedirs(path_pc, exist_ok=True)
    for idx, _ in tqdm(enumerate(pc_FW.id), total=len(pc_FW.id)):
        pc = np.load(pc_FW.roots[idx])
        mask = np.load(mask_FW.roots[idx])
        bbox_data = np.load(bbox3d_FW.roots[idx])
        numofinstances = mask.shape[0]

        pcd = plot_3d_bboxes(pc, bbox_data, visualize=True)
        result_path = os.path.join(path_pc, f'ID{str(idx).zfill(3)}--{numofinstances}instances.ply')
        # o3d.io.write_point_cloud(result_path, pcd, print_progress=True)
    print("Execution complete")

if __name__ == "__main__":
    # File paths

    pc_path = r"dl_challenge\8b061a8a-9915-11ee-9103-bbb8eae05561\pc.npy"
    bbox_path = r"dl_challenge\8b061a8a-9915-11ee-9103-bbb8eae05561\bbox3d.npy"
    plot_3d_bboxes(pc_path, bbox_path)
