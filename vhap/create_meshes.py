
import math
import os
from typing import Optional, Literal, Dict, List
from glob import glob
import concurrent.futures
import multiprocessing
from copy import deepcopy
import yaml
import json
import tyro
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
# from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from vhap.config.base import DataConfig, ModelConfig, import_module
from vhap.data.nerf_dataset import NeRFDataset
from vhap.model.flame import FlameHead
from vhap.util.mesh import get_obj_content
from vhap.util.render_nvdiffrast import NVDiffRenderer

# to prevent "OSError: [Errno 24] Too many open files"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


max_threads = min(multiprocessing.cpu_count(), 8)

def load_flame_params(src_folder):
    """
    src_folder: base folder
    flame_param_paths: list of relative paths used during writing
    """
    files = sorted(Path(src_folder).glob("*"))
    flame_param_paths = [f.relative_to(src_folder) for f in files]

    collected = {}
    identity_keys = ['shape', 'static_offset']

    for rel_path in sorted(flame_param_paths):
        file_path = src_folder / rel_path

        data = np.load(file_path)  # same format used in write_data

        for key, value in data.items():

            # Identity parameters (saved fully each time)
            if key in identity_keys:
                if key not in collected:
                    collected[key] = value
                continue

            # Frame-varying parameters
            if key not in collected:
                collected[key] = []

            collected[key].append(value)  # each is shape (1, D)

    # Concatenate frame-wise params
    for key, value in collected.items():
        if isinstance(value, list):
            collected[key] = np.concatenate(value, axis=0)

    return collected

def infer_flame_params(flame_model: FlameHead, flame_params: Dict, indices:List):
    if 'static_offset' in flame_params:
        static_offset = flame_params['static_offset']
        if isinstance(static_offset, np.ndarray):
            static_offset = torch.tensor(static_offset)
    else:
        static_offset = None
    for k in flame_params:
        if isinstance(flame_params[k], np.ndarray):
            flame_params[k] = torch.tensor(flame_params[k])
    with torch.no_grad():
        ret = flame_model(
            flame_params['shape'][None, ...].expand(len(indices), -1),
            flame_params['expr'][indices],
            flame_params['rotation'][indices],
            flame_params['neck_pose'][indices],
            flame_params['jaw_pose'][indices],
            flame_params['eyes_pose'][indices],
            flame_params['translation'][indices],
            return_verts_cano=False,
            static_offset=static_offset,
        )
    verts = ret[0]
    return verts
def write_canonical_mesh(flame_params, flame_model, tgt_folder):
    print(f"Inferencing FLAME in the canonical space...")
    if 'static_offset' in flame_params:
        static_offset = torch.tensor(flame_params['static_offset'])
    else:
        static_offset = None
    with torch.no_grad():
        ret = flame_model(
            torch.tensor(flame_params['shape'])[None, ...],
            torch.zeros(*flame_params['expr'][:1].shape),
            torch.zeros(*flame_params['rotation'][:1].shape),
            torch.zeros(*flame_params['neck_pose'][:1].shape),
            torch.tensor([[0.3, 0, 0]]),
            torch.zeros(*flame_params['eyes_pose'][:1].shape),
            torch.zeros(*flame_params['translation'][:1].shape),
            return_verts_cano=False,
            static_offset=static_offset,
        )
    verts = ret[0]

    cano_mesh_path = tgt_folder / 'canonical.obj'
    print(f"Writing canonical mesh to: {cano_mesh_path}")
    obj_data = get_obj_content(verts[0], flame_model.faces)
    write_data({cano_mesh_path: obj_data})

    return flame_model

def write_expr_and_mesh(tgt_folder, exp_path, expr, mesh_path, verts, faces):
    path2data = {}

    expr_data = '\n'.join([str(n) for n in expr])
    path2data[tgt_folder / exp_path] = expr_data

    obj_data = get_obj_content(verts, faces)
    path2data[tgt_folder / mesh_path] = obj_data
    write_data(path2data)
def write_data(path2data):
    for path, data in path2data.items():
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")


if __name__ == "__main__":
    data_dir = Path('/raid/sanjeev_k/BTP-aayan-shree/VHAP/export/nersemble_v2')
    subjects = os.listdir(data_dir)
    for su in subjects:
        try:
            print(f"Processing subject: {su}")
            '''
            cfg_model

            add_teeth: true
            flame_params_path: null
            n_expr: 100
            n_shape: 300
            n_tex: 100
            occluded: !!python/tuple []
            remove_lip_inside: false
            residual_tex: true
            tex_clusters: !!python/tuple
            - skin
            - hair
            - boundary
            - lips_tight
            - teeth
            - sclerae
            - irises
            tex_extra: true
            tex_painted: true
            tex_resolution: 2048
            use_dynamic_offset: false
            use_static_offset: true
            '''
            tgt_dir = data_dir / su
            db_backup_path = tgt_dir / "transforms_backup.json"
            assert db_backup_path.exists(), f"Could not find {db_backup_path}"
            print(f"Loading database from: {db_backup_path}")
            db = json.load(open(db_backup_path, "r"))
            flame_params = load_flame_params(tgt_dir / "flame_param")
            flame_model = FlameHead(300, 100, add_teeth=True)

            #write canonical mesh
            flame_model = write_canonical_mesh(flame_params, flame_model, tgt_dir)
            indices = db['timestep_indices']
            verts = infer_flame_params(flame_model, flame_params, indices)

            saved = [False] * len(db['timestep_indices'])  # avoid writing the same mesh multiple times
            num_processes = multiprocessing.cpu_count()
            worker_args = []
            for i, frame in tqdm(enumerate(db['frames']), total=len(db['frames'])):
                ti_orig = frame['timestep_index_original']  # use ti_orig when loading FLAME parameters
                ti = frame['timestep_index']  # use ti when saving files
                frame['exp_path'] = f"flame/exp/{ti:05d}.txt"
                frame['mesh_path'] = f"meshes/{ti:05d}.obj"
                if not saved[ti]:
                    worker_args.append([tgt_dir, frame['exp_path'], flame_params['expr'][ti_orig], frame['mesh_path'], verts[ti_orig], flame_model.faces])
                    saved[ti] = True
                    func = write_expr_and_mesh

                if len(worker_args) == num_processes or i == len(db['frames'])-1:
                    pool = multiprocessing.Pool(processes=num_processes)
                    pool.starmap(func, worker_args)
                    pool.close()
                    pool.join()
                    worker_args= []

        except Exception as e:
            print(f"Error processing subject {su}: {e}")

