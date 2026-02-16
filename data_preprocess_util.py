import open3d as o3d
import numpy as np
import os

def compute_pcd_from_mesh(base_dir = "export/nersemble_v2/017_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine", keep_frames = {"00000", "00050"}):
    # Load OBJ as triangle mesh
    for frame in keep_frames:
        mesh_path = f"{base_dir}/meshes/{frame}.obj"
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        print("Loaded mesh:", len(mesh.vertices), "verts,", len(mesh.triangles), "tris")

        # Get triangle centroids: average of 3 vertices per face
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        centroids = vertices[triangles].mean(axis=1)   # (N_tris, 3)
        print("Computed centroids for", len(centroids), "triangles")
        # Optional: compute average vertex color per triangle
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            tri_colors = colors[triangles].mean(axis=1)  # average RGB
        else:
            tri_colors = None

        # Create point cloud from centroids
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centroids)
        if tri_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(tri_colors)

        # Save as PLY
        ply_path = f"{base_dir}/meshes/{frame}.ply"
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
        print(f"Saved {frame}.ply with", len(centroids), "points")

def delete_useless_data(base_dir = "export/nersemble_v2/017_EMO-1_v16_DS4_whiteBg_staticOffset_maskBelowLine", keep_frames = {"00000", "00050"}):

    # Folders with <frameno>_<camerano>.png
    frame_camera_dirs = [
        "fg_masks",
        "images",
    ]

    # Folders with <frameno>.<extension>
    frame_only_dirs = [
        "flame/exp",
        "flame_param",
        "meshes",
    ]

    # Handle <frameno>_<camerano>.png
    for subdir in frame_camera_dirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(dir_path):
            print(f"Directory not found, skipping: {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)

            if not os.path.isfile(fpath):
                continue

            # Expected format: <frameno>_<camerano>.png
            if "_" not in fname:
                os.remove(fpath)
                continue

            frame_no = fname.split("_")[0]

            if frame_no not in keep_frames:
                os.remove(fpath)

        print(f"Cleaned directory: {dir_path}")

    # Handle <frameno>.<extension>
    for subdir in frame_only_dirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(dir_path):
            print(f"Directory not found, skipping: {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)

            if not os.path.isfile(fpath):
                continue

            # Expected format: <frameno>.<extension>
            frame_no = os.path.splitext(fname)[0]

            if frame_no not in keep_frames:
                os.remove(fpath)

        print(f"Cleaned directory: {dir_path}")
    
if __name__ == "__main__":
    data_dir = "../VHAP/export/nersemble_v2/"
    sub_emo_dirs = os.listdir(data_dir)
    subject_range = (1,555) #inclusive
    emos = ["EMO-1", "EMO-2", "EMO-3", "EMO-4", "EXP-1", "EXP-2"]
    # emos = ["EMO-1"]
    for d in sub_emo_dirs:
        s = int(d.split("_")[0])
        e = d.split("_")[1]
        if s>= subject_range[0] and s<= subject_range[1] and e in emos:
            print(f"Processing subject {s} emotion {e}")
            base_dir = os.path.join(data_dir, d)
            keep_frames = {"00000", "00050"}
            delete_useless_data(base_dir, keep_frames)
            compute_pcd_from_mesh(base_dir, keep_frames)
            


