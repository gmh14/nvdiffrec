import torch
import numpy as np


def laplacian_uniform(verts, faces):
    V = verts.shape[0]
    F = faces.shape[0]

    faces = faces.cpu().numpy()

    org_triangles = np.vstack(
        [
            faces[:, [1, 2, 3]],
            faces[:, [0, 3, 2]],
            faces[:, [0, 1, 3]],
            faces[:, [0, 2, 1]],
        ]
    )

    # Sort each triangle's vertices to avoid duplicates due to ordering
    triangles = np.sort(org_triangles, axis=1)

    unique_triangles, tri_idx, counts = np.unique(
        triangles, axis=0, return_index=True, return_counts=True
    )

    twice_tri_id = np.where(counts == 2)[0]
    twice_triangles = unique_triangles[twice_tri_id]

    # each twice_triangle is a shared face of two tet elements
    # find the ids of the two tet elements
    matches = np.all(triangles[:, None] == twice_triangles[None, :], axis=2)
    matching_indices = np.where(matches)

    # matching_indices[1] contains the row indices in B
    # Sort these indices and use them to reshape matching_indices[0] which contains the corresponding row indices in A
    sorted_indices = np.argsort(matching_indices[1])
    row_indices = matching_indices[0][sorted_indices].reshape(-1, 2)
    adj = row_indices % F
    adj = np.concatenate(
        [np.stack([adj[:, 1], adj[:, 0]], axis=1).transpose(), adj.transpose()], axis=1
    )

    adj_values = np.ones(adj.shape[1], dtype=np.float32)

    diag_idx = torch.tensor(adj[0])
    adj = torch.tensor(adj)
    adj_values = torch.tensor(adj_values)
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (F, F)).coalesce()


def compute_matrix(verts, faces):
    L = laplacian_uniform(verts, faces)
    return L  # T x T

def expand_sparse_tensor(sparse_tensor, n):
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    F = sparse_tensor.size(0)
    new_indices = torch.cat([indices + i * F for i in range(n)], dim=1)
    new_values = values.repeat(n)
    new_size = torch.Size([F * n, F * n])
    return torch.sparse_coo_tensor(new_indices, new_values, new_size).coalesce()
    

def compute_G_matrix(verts_init, faces):
    # compute gradient operator matrix
    V = verts_init.shape[0]
    T = faces.shape[0]

    Gd = torch.zeros([4, 3], device=verts_init.device)
    Gd[0, :] = -1.0
    Gd[1, 0] = 1.0
    Gd[2, 1] = 1.0
    Gd[3, 2] = 1.0  # 4 x 3

    X = verts_init[faces]  # T x 4 x 3
    X = X.transpose(1, 2)  # T x 3 x 4
    dX = torch.matmul(X, Gd.unsqueeze(0).expand(T, -1, -1))  # T x 3 x 3

    dX_inv = torch.inverse(dX)  # T x 3 x 3

    # G = torch.matmul(Gd.unsqueeze(0).expand(T, -1, -1), dX_inv) # T x 4 x 3
    # return G

    G = torch.zeros([T, 9, 12], device=verts_init.device)
    for dofi in range(12):
        E = torch.zeros([3, 4], device=verts_init.device)
        E[dofi % 3, dofi // 3] = 1.0

        Z = E @ Gd
        R = Z.unsqueeze(0).expand(T, -1, -1) @ dX_inv
        G[:, :, dofi] = R.view(T, -1)

    return G # T x 4 x 3


def compute_energy(verts, faces, G, L):
    v = verts[faces]  # T x 4 x 3
    # v = v.transpose(1, 2)  # T x 3 x 4
    # F = torch.matmul(v, G)  # T x 3 x 3
    T = faces.shape[0]
    F = G @ v.view(T, 12, 1)
    F = F.view(T, 3, 3)
    F_flatten = F.reshape(-1, 9)  # T x 9

    # smooth_eng = torch.trace(torch.matmul(torch.matmul(F_flatten.transpose(0, 1), L), F_flatten)) # 1
    FTL = torch.matmul(F_flatten.transpose(0, 1), L)
    smooth_eng = (FTL * FTL).sum() * 0.5

    # barrier = -1.0 * torch.sum(torch.log(torch.det(F))) # 1

    J = torch.relu(-torch.det(F)) - 1e-4
    barrier = (J * J).sum() * 0.5

    return smooth_eng, barrier


def get_surface_vf(faces):
    # get surface faces
    org_triangles = np.vstack(
        [
            faces[:, [1, 2, 3]],
            faces[:, [0, 3, 2]],
            faces[:, [0, 1, 3]],
            faces[:, [0, 2, 1]],
        ]
    )

    # Sort each triangle's vertices to avoid duplicates due to ordering
    triangles = np.sort(org_triangles, axis=1)

    unique_triangles, tri_idx, counts = np.unique(
        triangles, axis=0, return_index=True, return_counts=True
    )

    once_tri_id = counts == 1
    surface_triangles = unique_triangles[once_tri_id]

    surface_vertices = np.unique(surface_triangles)

    vertex_mapping = {vertex_id: i for i, vertex_id in enumerate(surface_vertices)}

    mapped_triangles = np.vectorize(vertex_mapping.get)(
        org_triangles[tri_idx][once_tri_id]
    )

    return surface_vertices, mapped_triangles

    