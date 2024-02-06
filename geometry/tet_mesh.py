# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import mesh
from render import render
import xatlas

from geometry.tet_geometry import (
    compute_matrix, compute_G_matrix, compute_energy, 
    get_surface_vf, 
    expand_sparse_tensor
)


###############################################################################
#  Geometry interface
###############################################################################

class TetMesh(torch.nn.Module):
    def __init__(self, scale, FLAGS):
        super(TetMesh, self).__init__()

        self.FLAGS         = FLAGS
        
        large_steps_data_path = "./"
        v = np.load(f"{large_steps_data_path}/sphere_tet/sphere_v.npy") * 0.4
        f = np.load(f"{large_steps_data_path}/sphere_tet/sphere_f.npy")
        
        surface_vid, surface_trimesh_f = get_surface_vf(f)
        
        surface_v = v[surface_vid]
        vmapping, indices, uvs = xatlas.parametrize(surface_v, surface_trimesh_f)
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        v_init = torch.tensor(v, dtype=torch.float32, device='cuda')
        tet_faces = torch.tensor(f, dtype=torch.long, device='cuda')
        
        load_L = True
        if load_L:
            L = torch.load(f"{large_steps_data_path}/sphere_tet/L.pt")
        else:
            L = compute_matrix(v_init, tet_faces)
            torch.save(L, f"{large_steps_data_path}/sphere_tetL.pt")
            
        G = compute_G_matrix(v_init, tet_faces)
        
        self.tet_v = torch.nn.Parameter(v_init.clone().detach(), requires_grad=True)
        self.register_parameter("tet_v", self.tet_v)
        
        self.tet_f = tet_faces
        self.surface_vid = torch.tensor(surface_vid, dtype=torch.long, device='cuda')
        self.surface_trimesh_f = torch.tensor(surface_trimesh_f, dtype=torch.long, device='cuda')
        
        self.L = L.to('cuda')
        self.G = G.to('cuda')
        
        self.uv_idx = torch.tensor(indices_int64, dtype=torch.long, device='cuda')
        self.uv = torch.tensor(uvs, dtype=torch.float32, device='cuda')

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.tet_v, dim=0).values, torch.max(self.tet_v, dim=0).values

    def getMesh(self, material):
        verts = self.tet_v[self.surface_vid]
        faces = self.surface_trimesh_f
        
        imesh = mesh.Mesh(verts, faces, v_tex=self.uv, t_tex_idx=self.uv_idx, material=material)
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        
        smooth_eng, barrier = compute_energy(self.tet_v, self.tet_f, self.G, self.L)
        
        # Compute regularizer. 
        reg_loss += 1e-4 * smooth_eng

        # Albedo (k_d) smoothnesss regularizer
        kd_smooth = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)
        reg_loss += kd_smooth

        # Visibility regularizer
        visibility_reg = torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)
        reg_loss += visibility_reg

        # Light white balance regularizer
        light_reg = lgt.regularizer() * 0.005
        reg_loss = reg_loss + light_reg
        
        log = {
            "img_loss": img_loss.item(),
            "smooth_eng": smooth_eng.item(),
            "barrier": barrier.item(),
            "kd_smooth": kd_smooth.item(),
            "visibility_reg": visibility_reg.item(),
            "light_reg": light_reg.item(),
            "total_reg_loss": reg_loss.item()
        }

        return img_loss * 100, reg_loss, log
    