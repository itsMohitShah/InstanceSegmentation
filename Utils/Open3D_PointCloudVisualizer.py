# Having an easy way to visualize point clouds. Open a file dialogue and navigate to the .ply file.
#TODO: Plot BBoxs as well

import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import os

while True:
    window = tk.Tk()
    window.withdraw()
    path = tk.filedialog.askopenfilename(filetypes = [("Point Cloud Files", "*.ply")])
    if path == "":
        break
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])