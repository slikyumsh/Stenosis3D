from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, convolve
from skimage import measure
from skimage.morphology import skeletonize


# ----------------------------- IO ----------------------------- #
def load_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    zooms = nii.header.get_zooms()
    pixdim = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, pixdim


# ----------------------------- Skeleton utils ----------------------------- #
def prune_skeleton_endpoints(skel_u8: np.ndarray, iters: int) -> np.ndarray:
    """
    Iteratively remove endpoints (voxels with <=1 neighbor) 'iters' times.
    Helps reduce tiny spurs.
    """
    if iters <= 0:
        return (skel_u8 > 0).astype(np.uint8)

    sk = (skel_u8 > 0).astype(np.uint8)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    for _ in range(iters):
        nbr = convolve(sk, kernel, mode="constant", cval=0)
        endpoints = (sk > 0) & (nbr <= 1)
        if not np.any(endpoints):
            break
        sk[endpoints] = 0

    return sk.astype(np.uint8)


def skeleton_points_and_edges_26(skel_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      pts_vox: [N,3] voxel coordinates (x,y,z)
      edges:   [M,2] undirected edges between 26-neighborhood neighbors
    """
    pts = np.argwhere(skel_u8 > 0)
    if pts.size == 0:
        return pts.astype(np.int32), np.zeros((0, 2), dtype=np.int32)

    pts_list = [tuple(p) for p in pts]
    idx: Dict[Tuple[int, int, int], int] = {p: i for i, p in enumerate(pts_list)}

    # half of offsets to avoid duplicating edges
    half_offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if (dx > 0) or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    half_offsets.append((dx, dy, dz))

    sx, sy, sz = skel_u8.shape
    edges = []
    for i, (x, y, z) in enumerate(pts_list):
        for dx, dy, dz in half_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz:
                j = idx.get((nx, ny, nz), -1)
                if j >= 0:
                    edges.append((i, j))

    if not edges:
        return pts.astype(np.int32), np.zeros((0, 2), dtype=np.int32)

    return pts.astype(np.int32), np.array(edges, dtype=np.int32)


def longest_path_on_skeleton_graph(pts_vox: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Extract an ordered "main branch" path on the skeleton graph using a classic trick:
      - pick an endpoint (or any node)
      - BFS -> farthest node A
      - BFS from A -> farthest node B, keep parents
      - reconstruct path A..B

    Returns: ordered voxel coordinates [L,3].
    """
    if pts_vox.size == 0:
        return pts_vox.astype(np.int32)
    n = pts_vox.shape[0]
    if edges.size == 0:
        return pts_vox[:1].astype(np.int32)

    # adjacency list
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    endpoints = np.where(degrees == 1)[0]
    start = int(endpoints[0]) if endpoints.size > 0 else 0

    def bfs(source: int):
        from collections import deque
        dist = np.full((n,), -1, dtype=np.int32)
        parent = np.full((n,), -1, dtype=np.int32)
        q = deque([source])
        dist[source] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        far = int(np.argmax(dist))
        return far, dist, parent

    a, _, _ = bfs(start)
    b, _, parent = bfs(a)

    # reconstruct b -> a
    path_idx = []
    cur = b
    while cur != -1:
        path_idx.append(cur)
        if cur == a:
            break
        cur = int(parent[cur])
    path_idx = path_idx[::-1]
    return pts_vox[np.array(path_idx, dtype=np.int32)].astype(np.int32)


# ----------------------------- Geometry / measurements ----------------------------- #
def marching_cubes_mesh(mask_bin: np.ndarray, pixdim: Tuple[float, float, float]):
    if mask_bin.max() == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    verts, faces, _, _ = measure.marching_cubes(mask_bin.astype(np.uint8), level=0.5, spacing=pixdim)
    return verts.astype(np.float32), faces.astype(np.int32)


def edt_radius_mm(mask_bin: np.ndarray, pixdim: Tuple[float, float, float]) -> np.ndarray:
    return distance_transform_edt(mask_bin.astype(bool), sampling=pixdim).astype(np.float32)


def vox_to_world(pts_vox: np.ndarray, pixdim: Tuple[float, float, float]) -> np.ndarray:
    return pts_vox.astype(np.float32) * np.array(pixdim, dtype=np.float32)[None, :]


def pick_chain_indices(length: int, center_k: int, n: int) -> np.ndarray:
    """
    Return n indices centered around center_k (clamped to [0, length-1]).
    If length is small, returns as many unique indices as possible.
    """
    if length <= 0:
        return np.array([], dtype=np.int32)

    n = max(1, int(n))
    half = n // 2
    start = max(0, center_k - half)
    end = min(length, start + n)
    start = max(0, end - n)
    idx = np.arange(start, end, dtype=np.int32)
    return idx


def choose_chain_on_main_branch(
    main_path_vox: np.ndarray,
    mask_bin: np.ndarray,
    pixdim: Tuple[float, float, float],
    mode: str,
    n_spheres: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decide where to place a chain of spheres on the MAIN branch.
    mode:
      - min_diam : chain centered at minimum diameter point on main branch
      - middle   : chain centered at middle of main branch

    Returns:
      centers_world: [K,3]
      radii_mm: [K]
    """
    if main_path_vox.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)

    r = edt_radius_mm(mask_bin, pixdim)
    rr = r[main_path_vox[:, 0], main_path_vox[:, 1], main_path_vox[:, 2]]
    rr = np.where(np.isfinite(rr), rr, 0.0)

    if mode == "min_diam":
        valid = rr > 0
        if np.any(valid):
            k0 = int(np.argmin(rr[valid]))
            k0 = int(np.where(valid)[0][k0])
        else:
            k0 = 0
    else:
        k0 = int(len(rr) // 2)

    chain_idx = pick_chain_indices(len(rr), k0, n_spheres)
    chain_vox = main_path_vox[chain_idx]
    centers_world = vox_to_world(chain_vox, pixdim)
    radii_mm = rr[chain_idx].astype(np.float32)

    # fallback radii if zeros
    max_r = float(np.max(rr)) if rr.size else 1.0
    radii_mm = np.where(radii_mm > 0, radii_mm, max_r).astype(np.float32)

    return centers_world, radii_mm


# ----------------------------- Rendering ----------------------------- #
def render_pyvista(
    out_png: Path,
    vessel_verts: np.ndarray,
    vessel_faces: np.ndarray,
    skel_pts_world: np.ndarray,
    edges: np.ndarray,
    sphere_centers_world: np.ndarray,
    sphere_radii_mm: np.ndarray,
    line_radius_mm: float,
    add_title: bool,
    title_text: str,
):
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=(1800, 1300))
    pl.set_background("white")

    # Vessel surface
    if vessel_verts.shape[0] > 0 and vessel_faces.shape[0] > 0:
        faces_pv = np.hstack([np.full((vessel_faces.shape[0], 1), 3, dtype=np.int32), vessel_faces]).ravel()
        surf = pv.PolyData(vessel_verts, faces_pv)
        pl.add_mesh(surf, opacity=0.25, smooth_shading=True)

    # Centerline as tubes (RED)
    if skel_pts_world.shape[0] > 0 and edges.shape[0] > 0:
        poly = pv.PolyData(skel_pts_world)
        lines = np.hstack([np.full((edges.shape[0], 1), 2, dtype=np.int32), edges]).ravel()
        poly.lines = lines
        tube = poly.tube(radius=float(line_radius_mm), n_sides=16)
        pl.add_mesh(tube, color="red", opacity=1.0)

    # Spheres chain (GREEN)
    if sphere_centers_world.shape[0] > 0:
        for c, rad in zip(sphere_centers_world, sphere_radii_mm):
            sph = pv.Sphere(
                radius=float(rad),
                center=tuple(map(float, c)),
                theta_resolution=64,
                phi_resolution=32,
            )
            pl.add_mesh(sph, color="green", opacity=0.95)

    if add_title and title_text:
        pl.add_text(title_text, font_size=14, color="black")

    pl.camera_position = "iso"
    pl.camera.zoom(1.2)

    pl.show(screenshot=str(out_png))
    pl.close()


def render_matplotlib(
    out_png: Path,
    vessel_verts: np.ndarray,
    vessel_faces: np.ndarray,
    skel_pts_world: np.ndarray,
    edges: np.ndarray,
    sphere_centers_world: np.ndarray,
    sphere_radii_mm: np.ndarray,
    add_title: bool,
    title_text: str,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    if add_title and title_text:
        ax.set_title(title_text)

    # Vessel
    if vessel_verts.shape[0] > 0 and vessel_faces.shape[0] > 0:
        tris = vessel_verts[vessel_faces]
        mesh = Poly3DCollection(tris, alpha=0.15)
        ax.add_collection3d(mesh)

    # Centerline edges (RED)
    if skel_pts_world.shape[0] > 0 and edges.shape[0] > 0:
        for i, j in edges:
            a = skel_pts_world[i]
            b = skel_pts_world[j]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=1.0, color="red")

    # Spheres chain (GREEN, wireframe)
    if sphere_centers_world.shape[0] > 0:
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        for c, rad in zip(sphere_centers_world, sphere_radii_mm):
            x = c[0] + rad * np.outer(np.cos(u), np.sin(v))
            y = c[1] + rad * np.outer(np.sin(u), np.sin(v))
            z = c[2] + rad * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.3, alpha=0.9, color="green")

    # autoscale
    all_pts = []
    if vessel_verts.shape[0] > 0:
        all_pts.append(vessel_verts)
    if skel_pts_world.shape[0] > 0:
        all_pts.append(skel_pts_world)
    if sphere_centers_world.shape[0] > 0:
        all_pts.append(sphere_centers_world)

    if all_pts:
        P = np.vstack(all_pts)
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        center = (mins + maxs) / 2
        extent = (maxs - mins).max()
        ax.set_xlim(center[0] - extent / 2, center[0] + extent / 2)
        ax.set_ylim(center[1] - extent / 2, center[1] + extent / 2)
        ax.set_zlim(center[2] - extent / 2, center[2] + extent / 2)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# ----------------------------- Main ----------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", type=str, required=True, help="Mask nii(.gz) to render as vessel surface.")
    ap.add_argument("--centerline", type=str, default="", help="Optional: centerline nii(.gz). If omitted -> computed from mask.")
    ap.add_argument("--out", type=str, required=True, help="Output PNG path.")
    ap.add_argument("--sphere-mode", type=str, default="min_diam", choices=["min_diam", "middle"])
    ap.add_argument("--n-spheres", type=int, default=10, help="How many spheres to draw in a row (default 10).")
    ap.add_argument("--line-radius-mm", type=float, default=0.25, help="Centerline tube radius (mm) for PyVista.")
    ap.add_argument("--prune-iters", type=int, default=0, help="Endpoint pruning iterations for skeleton (0 disables).")
    ap.add_argument("--no-title", action="store_true", help="Do not draw title (for paper figures).")
    args = ap.parse_args()

    mask_path = Path(args.mask)
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    mask_f, pixdim = load_nii(mask_path)
    mask_bin = (mask_f > 0).astype(np.uint8)

    # skeleton
    if args.centerline:
        cl_f, _ = load_nii(Path(args.centerline))
        skel = (cl_f > 0).astype(np.uint8)
    else:
        skel = skeletonize(mask_bin.astype(bool), method="lee").astype(np.uint8)

    if args.prune_iters > 0:
        skel = prune_skeleton_endpoints(skel, args.prune_iters)

    # full skeleton for visualization (all branches)
    skel_pts_vox, edges = skeleton_points_and_edges_26(skel)
    if skel_pts_vox.size == 0:
        raise RuntimeError("Skeleton is empty. Check mask / thresholding.")

    # main branch only for choosing a clean ordered chain of spheres
    main_path_vox = longest_path_on_skeleton_graph(skel_pts_vox, edges)
    if main_path_vox.size == 0:
        main_path_vox = skel_pts_vox[:1]

    sphere_centers_world, sphere_radii_mm = choose_chain_on_main_branch(
        main_path_vox=main_path_vox,
        mask_bin=mask_bin,
        pixdim=pixdim,
        mode=args.sphere_mode,
        n_spheres=args.n_spheres,
    )

    # surface + skeleton to world
    vessel_verts, vessel_faces = marching_cubes_mesh(mask_bin, pixdim)
    skel_pts_world = vox_to_world(skel_pts_vox, pixdim)

    title = f"mask={mask_path.name}  sphere_mode={args.sphere_mode}  n_spheres={args.n_spheres}"

    # render
    try:
        render_pyvista(
            out_png=out_png,
            vessel_verts=vessel_verts,
            vessel_faces=vessel_faces,
            skel_pts_world=skel_pts_world,
            edges=edges,
            sphere_centers_world=sphere_centers_world,
            sphere_radii_mm=sphere_radii_mm,
            line_radius_mm=args.line_radius_mm,
            add_title=(not args.no_title),
            title_text=title,
        )
        print(f"[OK] saved (pyvista): {out_png}")
    except Exception as e:
        print(f"[WARN] pyvista render failed: {repr(e)}")
        print("       Falling back to matplotlib...")
        render_matplotlib(
            out_png=out_png,
            vessel_verts=vessel_verts,
            vessel_faces=vessel_faces,
            skel_pts_world=skel_pts_world,
            edges=edges,
            sphere_centers_world=sphere_centers_world,
            sphere_radii_mm=sphere_radii_mm,
            add_title=(not args.no_title),
            title_text=title,
        )
        print(f"[OK] saved (matplotlib): {out_png}")


if __name__ == "__main__":
    main()

