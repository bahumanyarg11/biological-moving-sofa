"""molsurface.py -- molecular surface meshes + shaded 3D cutaway renders."""
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
import geom


def channel_atoms(coords, radii, axis, center, Lhalf_A, r_keep=22.0, z_pad=6.0):
    d = np.asarray(axis, float); d /= np.linalg.norm(d)
    rel = coords - center
    zc = rel @ d
    perp = np.linalg.norm(rel - np.outer(zc, d), axis=1)
    m = (perp < r_keep) & (np.abs(zc) < Lhalf_A + z_pad)
    return coords[m], radii[m]


def vdw_surface(coords, radii, pad=3.0, spacing=1.1, probe=0.0):
    """Marching-cubes vdW(+probe) isosurface. Returns (verts, faces)."""
    lo = coords.min(0) - (radii.max() + pad)
    hi = coords.max(0) + (radii.max() + pad)
    nx = np.arange(lo[0], hi[0], spacing)
    ny = np.arange(lo[1], hi[1], spacing)
    nz = np.arange(lo[2], hi[2], spacing)
    GX, GY, GZ = np.meshgrid(nx, ny, nz, indexing="ij")
    pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    tree = cKDTree(coords)
    dist, idx = tree.query(pts, k=1)
    field = (dist - radii[idx] - probe).reshape(GX.shape)
    verts, faces, normals, _ = measure.marching_cubes(field, level=0.0,
                                                      spacing=(spacing,) * 3)
    verts += lo
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=12)
        verts = np.asarray(m.vertices)
        normals = np.asarray(m.vertex_normals)
        faces = np.asarray(m.faces)
    except Exception:
        pass
    return verts, faces, normals


def shade_faces(verts, faces, normals, base_color, light=(0.4, 0.5, 0.85),
                ambient=0.5):
    ln = np.asarray(light, float); ln /= np.linalg.norm(ln)
    fn = normals[faces].mean(1)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
    lam = np.clip(fn @ ln, 0, 1)
    sh = ambient + (1 - ambient) * lam
    base = np.array(to_rgba(base_color)[:3])
    fc = np.clip(base[None, :] * sh[:, None], 0, 1)
    return fc


def cutaway(verts, faces, normals, cut_origin, cut_normal):
    """Keep faces whose centroid is on the negative side of the cut plane."""
    cn = np.asarray(cut_normal, float); cn /= np.linalg.norm(cn)
    cent = verts[faces].mean(1)
    keep = (cent - cut_origin) @ cn <= 0
    return faces[keep]


def render_channel(ax, pid, geo, atoms, color, rod=None, cut="back",
                   spacing=1.1, elev=8, azim=-88, alpha_surf=0.32):
    """Render a channel with the pore axis vertical and the optimal rod inside.

    cut="back": draw only the rear half of the surface (y>0) translucent so the
    opaque rod (drawn afterwards) reads as sitting inside the lumen with the
    rear wall visible behind it -- the cleanest 'cargo inside' look in
    matplotlib's painter-ordered 3D.
    """
    coords, radii = atoms
    g = geo[pid]
    axis = np.array(g["axis"]); axis /= np.linalg.norm(axis)
    cmid = np.array(g["centerline_A"]).mean(0)
    u, v, d = _basis(axis)
    Rmat = np.vstack([u, v, d])                 # rows map world->local
    cl = (coords - cmid) @ Rmat.T
    verts, faces, normals = vdw_surface(cl, radii, spacing=spacing)
    if cut == "back":
        faces = cutaway(verts, faces, normals, np.zeros(3), np.array([0, -1.0, 0]))
    fc = shade_faces(verts, faces, normals, color)
    pc = Poly3DCollection(verts[faces], facecolors=fc, edgecolors="none",
                          linewidths=0.0, antialiased=True, alpha=alpha_surf)
    pc.set_zsort("min")
    ax.add_collection3d(pc)

    if rod is not None:
        rx, c, e1, e2 = rod
        from figstyle import superquadric_mesh
        X, Y, Z = superquadric_mesh(rx, rx, c, e1, e2, nu=46, nv=92)
        from figstyle import shade_surface
        shade_surface(ax, X, Y, Z, "#E84855", alpha=1.0)

    _equal(ax, verts)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _basis(axis):
    d = np.asarray(axis, float); d /= np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v, d


def _equal(ax, V):
    mid = (V.max(0) + V.min(0)) / 2
    r = (V.max(0) - V.min(0)).max() / 2 * 1.0
    ax.set_xlim(mid[0]-r, mid[0]+r); ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


if __name__ == "__main__":
    import json, time, matplotlib.pyplot as plt
    import figstyle as F
    geo = json.load(open("results/channel_geometry.json"))
    val = json.load(open("results/validation.json"))
    opt = json.load(open("results/optimization.json"))
    FILES = {"7AHL": "7ahl.cif", "3EMN": "3emn.cif",
             "2ZW3": "2zw3.cif", "MspA": "1uun_bio.cif"}
    pid = "7AHL"
    t0 = time.time()
    coords, radii = geom.load_atoms("data/" + FILES[pid])
    g = geo[pid]; axis = np.array(g["axis"]); axis /= np.linalg.norm(axis)
    cmid = np.array(g["centerline_A"]).mean(0)
    atoms = channel_atoms(coords, radii, axis, cmid, g["length_nm"]*10/2)
    print("atoms", len(atoms[0]))
    fig = plt.figure(figsize=(3, 4))
    ax = fig.add_subplot(111, projection="3d")
    o = opt[pid]["opt_at_length"]
    rod = (o["rx"]*10*0.9, val[pid]["a_nav_nm"]*10, o["e1"], o["e2"])
    render_channel(ax, pid, geo, atoms, F.CH[pid], rod=rod)
    fig.savefig("figs/_test_mol.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    print("rendered in", round(time.time()-t0, 1), "s")
