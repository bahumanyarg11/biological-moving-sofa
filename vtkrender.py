"""vtkrender.py -- high-quality offscreen molecular renders (VTK).

Space-filling protein surface (clipped cutaway) with the optimal superquadric
nanorod inside the lumen, Phong-shaded with proper depth buffering. Outputs
transparent-background PNGs used by figures.py.
"""
import json
import numpy as np
import vtk
from vtk.util import numpy_support
import geom
from matplotlib.colors import to_rgb

DATA = "data/"
FILES = {"7AHL": "7ahl.cif", "3EMN": "3emn.cif",
         "2ZW3": "2zw3.cif", "MspA": "1uun_bio.cif"}
CHCOL = {"7AHL": "#0E9594", "3EMN": "#E07A1F", "2ZW3": "#8E44AD", "MspA": "#2E86DE"}
ROD = "#E84855"


def _basis(axis):
    d = np.asarray(axis, float); d /= np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v, d


def protein_actor(coords_local, radii, color, clip_normal=None,
                  clip_origin=(0, 0, 0), surface=True, opacity=1.0):
    """Molecular surface (smooth, from a vdW occupancy field) or CPK spheres,
    clipped to a half-space for a cutaway."""
    if surface:
        from scipy.spatial import cKDTree
        lo = coords_local.min(0) - 5; hi = coords_local.max(0) + 5
        sp = 0.8
        gx = np.arange(lo[0], hi[0], sp); gy = np.arange(lo[1], hi[1], sp)
        gz = np.arange(lo[2], hi[2], sp)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        q = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
        tree = cKDTree(coords_local)
        dist, idx = tree.query(q, k=1, workers=-1)
        probe = 1.4                       # SES-like inflation to close crevices
        field = (dist - radii[idx] - probe).reshape(GX.shape)
        img = vtk.vtkImageData()
        img.SetDimensions(*GX.shape)
        img.SetSpacing(sp, sp, sp); img.SetOrigin(lo[0], lo[1], lo[2])
        arr = numpy_support.numpy_to_vtk(
            field.transpose(2, 1, 0).ravel(order="F"), deep=True)
        img.GetPointData().SetScalars(arr)
        mc = vtk.vtkFlyingEdges3D(); mc.SetInputData(img); mc.SetValue(0, 0.0)
        mc.Update()
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputConnection(mc.GetOutputPort())
        conn.SetExtractionModeToLargestRegion(); conn.Update()
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(conn.GetOutputPort())
        smooth.SetNumberOfIterations(40); smooth.SetPassBand(0.04)
        smooth.NonManifoldSmoothingOn(); smooth.NormalizeCoordinatesOn(); smooth.Update()
        norms = vtk.vtkPolyDataNormals(); norms.SetInputConnection(smooth.GetOutputPort())
        norms.Update()
        src = norms
    else:
        pts = vtk.vtkPoints()
        for c in coords_local:
            pts.InsertNextPoint(*c)
        poly = vtk.vtkPolyData(); poly.SetPoints(pts)
        rar = numpy_support.numpy_to_vtk(radii.astype(float))
        rar.SetName("r"); poly.GetPointData().SetScalars(rar)
        sph = vtk.vtkSphereSource(); sph.SetRadius(1.0)
        sph.SetThetaResolution(16); sph.SetPhiResolution(16)
        glyph = vtk.vtkGlyph3D(); glyph.SetSourceConnection(sph.GetOutputPort())
        glyph.SetInputData(poly); glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(1.0); glyph.Update()
        src = glyph
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(src.GetOutputPort())
    mapper.ScalarVisibilityOff()
    if clip_normal is not None:
        plane = vtk.vtkPlane(); plane.SetOrigin(*clip_origin)
        plane.SetNormal(*clip_normal)
        mapper.AddClippingPlane(plane)
    act = vtk.vtkActor(); act.SetMapper(mapper)
    p = act.GetProperty()
    p.SetColor(*to_rgb(color)); p.SetInterpolationToPhong()
    p.SetSpecular(0.25); p.SetSpecularPower(25); p.SetDiffuse(0.85); p.SetAmbient(0.2)
    p.SetOpacity(opacity)
    return act


def rod_actor(rx, c, color=ROD):
    sq = vtk.vtkSuperquadricSource()
    sq.SetAxisOfSymmetry(2)               # z
    sq.SetThetaResolution(48); sq.SetPhiResolution(48)
    sq.SetThetaRoundness(1.0)             # round cross-section
    sq.SetPhiRoundness(0.35)              # near-cylindrical
    sq.SetScale(rx, rx, c)
    sq.SetSize(1.0)
    sq.Update()
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(sq.GetOutputPort())
    act = vtk.vtkActor(); act.SetMapper(mapper)
    p = act.GetProperty(); p.SetColor(*to_rgb(color)); p.SetInterpolationToPhong()
    p.SetSpecular(0.4); p.SetSpecularPower(40); p.SetDiffuse(0.85); p.SetAmbient(0.2)
    return act


def render(pid, geo, opt, val, size=(900, 1300), out=None):
    coords, radii = geom.load_atoms(DATA + FILES[pid])
    g = geo[pid]
    axis = np.array(g["axis"]); axis /= np.linalg.norm(axis)
    cmid = np.array(g["centerline_A"]).mean(0)
    u, v, d = _basis(axis)
    R = np.vstack([u, v, d])
    rel = coords - cmid
    zc = rel @ d; perp = np.linalg.norm(rel - np.outer(zc, d), axis=1)
    keep = (perp < 20) & (np.abs(zc) < g["length_nm"] * 10 / 2 + 7)
    local = rel[keep] @ R.T
    rad = radii[keep]

    ren = vtk.vtkRenderer(); ren.SetBackground(1, 1, 1)
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren); rw.SetSize(*size); rw.SetMultiSamples(8)

    o = opt[pid]["opt_at_length"]
    rx = o["rx"] * 10 * 0.9
    c = min(val[pid]["a_nav_nm"], g["length_nm"] / 2 * 0.96) * 10
    # lumen centre (mean of centreline) and bottleneck offset in transverse plane
    bc = (np.array(g["bottleneck_center"]) - cmid) @ R.T
    lc = (np.array(g["centerline_A"]).mean(0) - cmid) @ R.T  # ~origin
    cu = 0.5 * (bc[0] + lc[0]); cv = 0.5 * (bc[1] + lc[1])
    # clip the protein just in front of the rod so the rod is always exposed
    ren.AddActor(protein_actor(local, rad, CHCOL[pid], clip_normal=(0, 1, 0),
                               clip_origin=(0, cv - rx * 0.3, 0),
                               surface=False, opacity=1.0))
    rod = rod_actor(rx, c)
    rod.SetPosition(cu, cv, 0)
    ren.AddActor(rod)

    # lighting
    ren.AutomaticLightCreationOff()
    for pos, inten in [((-1, -1, 1.2), 1.0), ((1, -0.4, 0.6), 0.5),
                       ((0, -1, -0.6), 0.35)]:
        L = vtk.vtkLight(); L.SetPosition(*pos); L.SetFocalPoint(0, 0, 0)
        L.SetIntensity(inten); L.SetLightTypeToSceneLight(); ren.AddLight(L)

    cam = ren.GetActiveCamera()
    ren.ResetCamera()
    cam.SetViewUp(0, 0, 1)
    # place camera along -y looking at +y (into the cutaway)
    fp = np.array(cam.GetFocalPoint())
    dist = cam.GetDistance()
    cam.SetPosition(fp[0] + 0.15 * dist, fp[1] - dist, fp[2] + 0.12 * dist)
    cam.SetViewUp(0, 0, 1)
    ren.ResetCameraClippingRange()
    cam.Zoom(1.35)

    try:
        ren.UseDepthPeelingOn(); rw.SetAlphaBitPlanes(1)
        ren.SetMaximumNumberOfPeels(6); ren.SetOcclusionRatio(0.1)
    except Exception:
        pass

    rw.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGBA(); w2i.ReadFrontBufferOff(); w2i.Update()
    out = out or f"figs/mol_{pid}.png"
    wr = vtk.vtkPNGWriter(); wr.SetFileName(out)
    wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    return out


if __name__ == "__main__":
    geo = json.load(open("results/channel_geometry.json"))
    opt = json.load(open("results/optimization.json"))
    val = json.load(open("results/validation.json"))
    for pid in FILES:
        o = render(pid, geo, opt, val)
        print("wrote", o)
