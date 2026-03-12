#  Copyright 2025 The RICHIO Contributors
#
#  This file is part of RICHIO.
#
#  RICHIO is free software: you can redistribute it and/or modify it under
#  the terms of the European Union Public License version 1.2 or later, as
#  published by the European Commission.
#
#  RICHIO is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the European Union Public License for more details.
#
#  You should have received a copy of the EUPL in an/all official language(s) of
#  the European Union along with RICHIO.  If not, see <https://eupl.eu>.

"""
Analytical Voronoi-based slice and projection backends.

Algorithm
---------
**Slice** (area-weighted mean on the slicing plane)::

    result[i,j] = Σ_k  f_k · A(Pₖ ∩ pixel[i,j])
                / Σ_k        A(Pₖ ∩ pixel[i,j])

where Pₖ is the 2-D Voronoi polygon of generator k clipped to the image bbox.

**Projection** (column integral)::

    result[i,j] = Σ_k  f_k · (V_k / A₂D_k) · A(Pₖ ∩ pixel[i,j]) / A_pixel

where A₂D_k = area(Pₖ clipped to bbox), V_k is the 3-D cell volume, and
A_pixel = dx·dy.  The factor V_k / A₂D_k is the effective column length of
cell k.  Summing over pixels: Σᵢⱼ result[i,j]·A_pixel = Σ_k f_k·V_k (mass-
conservative).

Clipping kernel
---------------
Sutherland, I.E. & Hodgman, G.W. (1974).  Reentrant polygon clipping.
*Communications of the ACM*, **17** (1), 32–42.

The kernel is applied at two levels:

1. **Per cell** : clip the raw Voronoi polygon to the image bbox.  This keeps
   virtual far-vertices (used for infinite Voronoi edges) from blowing up the
   per-cell pixel-range scan.
2. **Per pixel**: clip the already-bbox-clipped polygon to the pixel rectangle
   and compute the intersection area via the shoelace formula.

Both passes run inside a single ``@njit`` accumulation loop.  Two scratch
buffers of shape ``(_MAX_CLIP_VERTS, 2)`` are allocated *once* at the start of
the loop, eliminating heap pressure in the inner loop.

Voronoi construction
--------------------
``scipy.spatial.Voronoi`` is used.  Eight sentinel points placed on a circle
of radius 3× the bbox diagonal surround the domain so that all real generator
cells are finite (no ``-1`` ridge vertex indices remain for real cells).  The
rare case of a real cell still carrying an infinite edge is handled by
extending the ridge to a virtual far-vertex before packing into ``flat_verts``.

Performance notes
-----------------
* For N ≳ 10⁵ the dominant cost is ``scipy.spatial.Voronoi`` (O(N log N)).
  Consider ``pyvoro`` (Voro++ wrapper) for parallel construction.
* The numba accumulation runs in O(N · k̄) where k̄ ≈ 4–6 is the average
  number of pixels overlapped per cell.
"""

from collections import defaultdict

import numpy as np
from numba import njit
from scipy.spatial import Voronoi

__all__ = ["_voronoi_slice", "_voronoi_project"]

# Maximum polygon vertex count after all Sutherland-Hodgman clipping stages.
# A Voronoi cell has on average ~6 vertices; each clip stage adds at most one
# vertex per clip edge (4 for bbox, 4 for pixel = 8 extra).  64 is generous.
_MAX_CLIP_VERTS = 64


# ---------------------------------------------------------------------------
# Sutherland-Hodgman clipping kernel (numba JIT, cache=True)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sh_clip(verts, n, x0, x1, y0, y1, out):
    """Clip a convex polygon to an axis-aligned rectangle.

    Implementation of Sutherland & Hodgman (1974), §3.  The polygon is clipped
    sequentially against the four half-planes x≥x0, x≤x1, y≥y0, y≤y1,
    ping-ponging between two internal scratch buffers.

    Parameters
    ----------
    verts : array, shape (≥n, 2), float64
        Input polygon vertices in CCW order (as returned by scipy.Voronoi).
    n : int
        Number of valid vertices in *verts*.
    x0, x1, y0, y1 : float
        Clip rectangle (x0 < x1, y0 < y1).
    out : array, shape (_MAX_CLIP_VERTS, 2), float64
        Pre-allocated output buffer.  Caller must not rely on content beyond
        the returned count.

    Returns
    -------
    int
        Number of vertices written to *out*.  Returns 0 when the polygon is
        entirely outside the rectangle (degenerate result).
    """
    MAX = _MAX_CLIP_VERTS
    buf0 = np.empty((MAX, 2))
    buf1 = np.empty((MAX, 2))

    # Initialise buf0 from input
    for k in range(n):
        buf0[k, 0] = verts[k, 0]
        buf0[k, 1] = verts[k, 1]
    n0 = n

    # ---- x >= x0  (read buf0 → write buf1) --------------------------------
    n1 = 0
    for i in range(n0):
        ip = n0 - 1 if i == 0 else i - 1
        sx = buf0[ip, 0];  sy = buf0[ip, 1]
        ex = buf0[i,  0];  ey = buf0[i,  1]
        if ex >= x0:
            if sx < x0:                                        # S outside → output intersection
                t = (x0 - sx) / (ex - sx)
                buf1[n1, 0] = x0;  buf1[n1, 1] = sy + t * (ey - sy);  n1 += 1
            buf1[n1, 0] = ex;  buf1[n1, 1] = ey;  n1 += 1    # always output E
        elif sx >= x0:                                         # E outside, S inside → output intersection
            t = (x0 - sx) / (ex - sx)
            buf1[n1, 0] = x0;  buf1[n1, 1] = sy + t * (ey - sy);  n1 += 1
    if n1 < 3:
        return 0

    # ---- x <= x1  (read buf1 → write buf0) --------------------------------
    n0 = 0
    for i in range(n1):
        ip = n1 - 1 if i == 0 else i - 1
        sx = buf1[ip, 0];  sy = buf1[ip, 1]
        ex = buf1[i,  0];  ey = buf1[i,  1]
        if ex <= x1:
            if sx > x1:
                t = (x1 - sx) / (ex - sx)
                buf0[n0, 0] = x1;  buf0[n0, 1] = sy + t * (ey - sy);  n0 += 1
            buf0[n0, 0] = ex;  buf0[n0, 1] = ey;  n0 += 1
        elif sx <= x1:
            t = (x1 - sx) / (ex - sx)
            buf0[n0, 0] = x1;  buf0[n0, 1] = sy + t * (ey - sy);  n0 += 1
    if n0 < 3:
        return 0

    # ---- y >= y0  (read buf0 → write buf1) --------------------------------
    n1 = 0
    for i in range(n0):
        ip = n0 - 1 if i == 0 else i - 1
        sx = buf0[ip, 0];  sy = buf0[ip, 1]
        ex = buf0[i,  0];  ey = buf0[i,  1]
        if ey >= y0:
            if sy < y0:
                t = (y0 - sy) / (ey - sy)
                buf1[n1, 0] = sx + t * (ex - sx);  buf1[n1, 1] = y0;  n1 += 1
            buf1[n1, 0] = ex;  buf1[n1, 1] = ey;  n1 += 1
        elif sy >= y0:
            t = (y0 - sy) / (ey - sy)
            buf1[n1, 0] = sx + t * (ex - sx);  buf1[n1, 1] = y0;  n1 += 1
    if n1 < 3:
        return 0

    # ---- y <= y1  (read buf1 → write buf0) --------------------------------
    n0 = 0
    for i in range(n1):
        ip = n1 - 1 if i == 0 else i - 1
        sx = buf1[ip, 0];  sy = buf1[ip, 1]
        ex = buf1[i,  0];  ey = buf1[i,  1]
        if ey <= y1:
            if sy > y1:
                t = (y1 - sy) / (ey - sy)
                buf0[n0, 0] = sx + t * (ex - sx);  buf0[n0, 1] = y1;  n0 += 1
            buf0[n0, 0] = ex;  buf0[n0, 1] = ey;  n0 += 1
        elif sy <= y1:
            t = (y1 - sy) / (ey - sy)
            buf0[n0, 0] = sx + t * (ex - sx);  buf0[n0, 1] = y1;  n0 += 1
    if n0 < 3:
        return 0

    for k in range(n0):
        out[k, 0] = buf0[k, 0]
        out[k, 1] = buf0[k, 1]
    return n0


# ---------------------------------------------------------------------------
# Accumulation kernels (numba JIT)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _accumulate_slice(flat_verts, offsets, counts, data_vals,
                      bx0, bx1, by0, by1, dx, dy, gx0, gy0, nx, ny,
                      out_num, out_den):
    """Area-weighted accumulation for slice.

    Parameters
    ----------
    flat_verts : (total_verts, 2) float64
        Concatenated raw Voronoi polygon vertices.
    offsets, counts : (N_cells,) int64
        Start index and vertex count of each cell in *flat_verts*.
    data_vals : (N_cells,) float64
        Field value for each cell (stripped of units).
    bx0, bx1, by0, by1 : float
        Image bounding box (same units as coordinates).
    dx, dy : float
        Pixel width and height.
    gx0, gy0 : float
        Left / bottom edge of the pixel grid (= bx0, by0).
    nx, ny : int
        Grid dimensions.
    out_num, out_den : (nx, ny) float64
        Accumulation arrays (modified in-place).
    """
    bbox_clipped  = np.empty((_MAX_CLIP_VERTS, 2))
    pixel_clipped = np.empty((_MAX_CLIP_VERTS, 2))

    for ci in range(len(counts)):
        n = counts[ci]
        if n < 3:
            continue
        verts = flat_verts[offsets[ci]:offsets[ci] + n]

        # Level-1 clip: raw polygon → bbox
        nc = _sh_clip(verts, n, bx0, bx1, by0, by1, bbox_clipped)
        if nc < 3:
            continue

        f = data_vals[ci]

        # Bounding box of bbox-clipped polygon → candidate pixel range
        xmn = bbox_clipped[0, 0];  xmx = bbox_clipped[0, 0]
        ymn = bbox_clipped[0, 1];  ymx = bbox_clipped[0, 1]
        for k in range(1, nc):
            v0 = bbox_clipped[k, 0];  v1 = bbox_clipped[k, 1]
            if v0 < xmn: xmn = v0
            if v0 > xmx: xmx = v0
            if v1 < ymn: ymn = v1
            if v1 > ymx: ymx = v1

        ix0 = max(0,      int((xmn - gx0) / dx))
        ix1 = min(nx - 1, int((xmx - gx0) / dx))
        iy0 = max(0,      int((ymn - gy0) / dy))
        iy1 = min(ny - 1, int((ymx - gy0) / dy))

        # Level-2 clip: bbox-clipped polygon → each candidate pixel
        for ix in range(ix0, ix1 + 1):
            px0 = gx0 + ix * dx;  px1 = px0 + dx
            for iy in range(iy0, iy1 + 1):
                py0 = gy0 + iy * dy;  py1 = py0 + dy
                ncp = _sh_clip(bbox_clipped, nc, px0, px1, py0, py1, pixel_clipped)
                if ncp < 3:
                    continue
                # Shoelace formula (signed area, CCW → positive)
                area = 0.0
                for i in range(ncp):
                    j = (i + 1) % ncp
                    area += (pixel_clipped[i, 0] * pixel_clipped[j, 1]
                             - pixel_clipped[j, 0] * pixel_clipped[i, 1])
                area = 0.5 * abs(area)
                if area > 0.0:
                    out_num[ix, iy] += f * area
                    out_den[ix, iy] += area


@njit(cache=True)
def _accumulate_proj(flat_verts, offsets, counts, data_vals, volumes,
                     bx0, bx1, by0, by1, dx, dy, gx0, gy0, nx, ny, out):
    """Column-integral accumulation for projection.

    For each cell k the contribution to pixel (ix, iy) is::

        f_k * (V_k / A2D_k) * A_int / A_pixel

    where A2D_k is the bbox-clipped 2-D Voronoi area (effective column cross-
    section), A_int is the pixel intersection area, and A_pixel = dx * dy.

    Parameters
    ----------
    flat_verts, offsets, counts, data_vals : see _accumulate_slice
    volumes : (N_cells,) float64
        3-D cell volume (same length units cubed as coordinates).
    out : (nx, ny) float64
        Output array (modified in-place); units = data_unit * length_unit.
    """
    bbox_clipped  = np.empty((_MAX_CLIP_VERTS, 2))
    pixel_clipped = np.empty((_MAX_CLIP_VERTS, 2))
    A_pixel = dx * dy

    for ci in range(len(counts)):
        n = counts[ci]
        if n < 3:
            continue
        verts = flat_verts[offsets[ci]:offsets[ci] + n]

        nc = _sh_clip(verts, n, bx0, bx1, by0, by1, bbox_clipped)
        if nc < 3:
            continue

        # Bbox-clipped area = effective column cross-section of this cell
        a2d = 0.0
        for i in range(nc):
            j = (i + 1) % nc
            a2d += (bbox_clipped[i, 0] * bbox_clipped[j, 1]
                    - bbox_clipped[j, 0] * bbox_clipped[i, 1])
        a2d = 0.5 * abs(a2d)
        if a2d <= 0.0:
            continue

        # Column-length weight:  f * V / A2D
        weight = data_vals[ci] * volumes[ci] / a2d

        xmn = bbox_clipped[0, 0];  xmx = bbox_clipped[0, 0]
        ymn = bbox_clipped[0, 1];  ymx = bbox_clipped[0, 1]
        for k in range(1, nc):
            v0 = bbox_clipped[k, 0];  v1 = bbox_clipped[k, 1]
            if v0 < xmn: xmn = v0
            if v0 > xmx: xmx = v0
            if v1 < ymn: ymn = v1
            if v1 > ymx: ymx = v1

        ix0 = max(0,      int((xmn - gx0) / dx))
        ix1 = min(nx - 1, int((xmx - gx0) / dx))
        iy0 = max(0,      int((ymn - gy0) / dy))
        iy1 = min(ny - 1, int((ymx - gy0) / dy))

        for ix in range(ix0, ix1 + 1):
            px0 = gx0 + ix * dx;  px1 = px0 + dx
            for iy in range(iy0, iy1 + 1):
                py0 = gy0 + iy * dy;  py1 = py0 + dy
                ncp = _sh_clip(bbox_clipped, nc, px0, px1, py0, py1, pixel_clipped)
                if ncp < 3:
                    continue
                a = 0.0
                for i in range(ncp):
                    j = (i + 1) % ncp
                    a += (pixel_clipped[i, 0] * pixel_clipped[j, 1]
                          - pixel_clipped[j, 0] * pixel_clipped[i, 1])
                a = 0.5 * abs(a)
                if a > 0.0:
                    out[ix, iy] += weight * a / A_pixel


# ---------------------------------------------------------------------------
# Python helpers: sentinel injection + cell extraction
# ---------------------------------------------------------------------------

def _add_sentinels(pts2d, x0, x1, y0, y1, n_sent=8):
    """Surround real generators with sentinel points on a large circle.

    The sentinels close off all otherwise-infinite Voronoi edges so that
    every real generator obtains a fully finite (bounded) Voronoi cell.

    Parameters
    ----------
    pts2d : (N, 2) float64
    x0, x1, y0, y1 : float  -  image bounding box
    n_sent : int  -  number of sentinel points (default 8)

    Returns
    -------
    pts_aug : (N + n_sent, 2) float64
    n_real  : int  =  N
    """
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    r  = 3.0 * max(x1 - x0, y1 - y0)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sent, endpoint=False)
    sentinels = np.column_stack([cx + r * np.cos(angles),
                                 cy + r * np.sin(angles)])
    return np.vstack([pts2d, sentinels]), len(pts2d)


def _build_flat_cells(vor, n_real, far_dist):
    """Extract Voronoi polygon vertices for the first *n_real* generators.

    For finite cells (no ``-1`` in the region) scipy already returns the
    vertices in CCW order; they are copied directly.  For the rare infinite
    cells (generators on the point-cloud convex hull that the sentinels did
    not fully enclose) each infinite ridge is extended to a virtual far-vertex
    at distance *far_dist* in the outward normal direction, and the resulting
    polygon is sorted CCW by angle from the generator.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
    n_real : int  -  number of real (non-sentinel) generators
    far_dist : float  -  extension distance for infinite ridges

    Returns
    -------
    flat_verts : (total_verts, 2) float64
    offsets    : (N_valid,) int64
    counts     : (N_valid,) int64
    gen_mask   : (n_real,) bool  -  True if the cell was successfully extracted
    """
    center = vor.points[:n_real].mean(axis=0)

    # gen_ridges is only needed for the infinite-cell fallback.  With sentinels
    # it is never triggered for real cells, so build it lazily to avoid the
    # O(N) dict-construction cost on every call.
    gen_ridges = None

    pieces      = []
    counts_list = []
    gen_mask    = np.zeros(n_real, dtype=bool)

    for gi in range(n_real):
        region = vor.regions[vor.point_region[gi]]

        if len(region) < 3:
            continue                                  # degenerate, skip

        if -1 not in region:
            # Happy path: all vertices are finite and already CCW (scipy guarantee)
            verts = vor.vertices[region].copy()

        else:
            # Fallback: extend each infinite ridge to a virtual far-vertex.
            # Build the ridge map only on the first encounter.
            if gen_ridges is None:
                gen_ridges = defaultdict(list)
                for ridx, (g1, g2) in enumerate(vor.ridge_points):
                    gen_ridges[int(g1)].append(ridx)
                    gen_ridges[int(g2)].append(ridx)

            seen_vi  = set()
            vlist    = []
            for ridx in gen_ridges[gi]:
                rv = vor.ridge_vertices[ridx]
                if -1 not in rv:
                    for vi in rv:
                        if vi not in seen_vi:
                            seen_vi.add(vi)
                            vlist.append(vor.vertices[vi])
                else:
                    # Find the other generator sharing this ridge
                    g1, g2 = int(vor.ridge_points[ridx, 0]), int(vor.ridge_points[ridx, 1])
                    other  = g2 if g1 == gi else g1
                    # Outward normal direction
                    tang = vor.points[other] - vor.points[gi]
                    norm = np.array([-tang[1], tang[0]], dtype=np.float64)
                    mid  = 0.5 * (vor.points[gi] + vor.points[other])
                    if np.dot(mid - center, norm) < 0.0:
                        norm = -norm
                    nlen = np.hypot(norm[0], norm[1])
                    if nlen < 1e-300:
                        continue
                    norm /= nlen
                    # Finite endpoint of the infinite ridge
                    finite_vi = rv[0] if rv[1] == -1 else rv[1]
                    if finite_vi not in seen_vi:
                        seen_vi.add(finite_vi)
                        vlist.append(vor.vertices[finite_vi])
                    vlist.append(vor.vertices[finite_vi] + far_dist * norm)

            if len(vlist) < 3:
                continue

            # Sort CCW by angle from the generator (not pre-sorted unlike finite cells)
            verts = np.array(vlist, dtype=np.float64)
            gp    = vor.points[gi]
            order = np.argsort(np.arctan2(verts[:, 1] - gp[1], verts[:, 0] - gp[0]))
            verts = verts[order]

        pieces.append(verts)
        counts_list.append(len(verts))
        gen_mask[gi] = True

    if not pieces:
        raise RuntimeError("_build_flat_cells: no valid Voronoi cells found.")

    flat_verts = np.ascontiguousarray(np.vstack(pieces), dtype=np.float64)
    counts     = np.array(counts_list, dtype=np.int64)
    offsets    = np.zeros(len(counts), dtype=np.int64)
    offsets[1:] = np.cumsum(counts[:-1])

    return flat_verts, offsets, counts, gen_mask


# ---------------------------------------------------------------------------
# Public backends (called from data.py)
# ---------------------------------------------------------------------------

def _voronoi_slice(data_f, Xf, Yf, x0, x1, y0, y1, nx, ny, vol_f=None):
    """Analytical Voronoi slice.

    All arrays must be plain float64 numpy arrays with consistent units
    (coordinate stripping is performed by the caller in ``data.py``).

    Parameters
    ----------
    data_f : (N,) float64  -  field values
    Xf, Yf : (N,) float64  -  in-plane coordinates of generators
    x0, x1, y0, y1 : float  -  image bbox
    nx, ny : int  -  output grid dimensions
    vol_f : (N,) float64, optional  -  cell volumes; used to compute the
        spatial filter margin.  If None a conservative default is used.

    Returns
    -------
    result : (nx, ny) float64  -  area-weighted mean field; NaN where uncovered
        (uncovered pixels are filled by a nearest-neighbour fallback).
    """
    # --- spatial filter: keep only cells that can affect the image bbox ------
    # A cell whose generator lies more than ~r_eff outside the bbox cannot
    # contribute to any pixel (its Voronoi polygon will be clipped to zero).
    # Using vol^(1/3) as the effective cell radius; margin = 2× 99th percentile.
    if vol_f is not None and len(vol_f):
        margin = 2.0 * float(np.percentile(vol_f ** (1.0 / 3.0), 99))
    else:
        margin = 0.05 * max(x1 - x0, y1 - y0)   # conservative fallback
    keep = ((Xf >= x0 - margin) & (Xf <= x1 + margin) &
            (Yf >= y0 - margin) & (Yf <= y1 + margin))
    n_in = int(keep.sum())
    print(f"  voronoi slice : {len(Xf):,} → {n_in:,} cells after bbox filter "
          f"(margin={margin:.3e})")
    if n_in == 0:
        raise ValueError("_voronoi_slice: no cells inside the image bbox.")
    Xf_k     = Xf[keep];     Yf_k     = Yf[keep]
    data_f_k = data_f[keep]
    # -------------------------------------------------------------------------

    pts2d = np.column_stack([Xf_k, Yf_k])
    far   = 3.0 * max(x1 - x0, y1 - y0)

    pts_aug, n_real = _add_sentinels(pts2d, x0, x1, y0, y1)
    vor = Voronoi(pts_aug)
    flat_verts, offsets, counts, gen_mask = _build_flat_cells(vor, n_real, far)

    gen_idx   = np.where(gen_mask)[0]
    data_vals = data_f_k[gen_idx].astype(np.float64)

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    out_num = np.zeros((nx, ny), dtype=np.float64)
    out_den = np.zeros((nx, ny), dtype=np.float64)

    _accumulate_slice(flat_verts, offsets, counts, data_vals,
                      x0, x1, y0, y1, dx, dy, x0, y0, nx, ny,
                      out_num, out_den)

    covered = out_den > 0.0
    result  = np.where(covered, out_num / np.where(covered, out_den, 1.0), np.nan)

    # NN fallback for pixels not covered by any cell (should be rare)
    n_miss = int(np.sum(~covered))
    if n_miss > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts2d)
        xc = x0 + (np.arange(nx) + 0.5) * dx
        yc = y0 + (np.arange(ny) + 0.5) * dy
        gx, gy = np.meshgrid(xc, yc, indexing="ij")
        miss = ~covered
        _, nn = tree.query(np.stack([gx[miss], gy[miss]], axis=-1))
        result[miss] = data_f_k[nn]

    return result


def _voronoi_project(data_f, vol_f, Xf, Yf, x0, x1, y0, y1, nx, ny):
    """Analytical Voronoi projection.

    Parameters
    ----------
    data_f : (N,) float64  -  field values
    vol_f  : (N,) float64  -  3-D cell volumes (coord_unit³)
    Xf, Yf : (N,) float64  -  transverse (projected) coordinates
    x0, x1, y0, y1 : float  -  image bbox
    nx, ny : int  -  output grid dimensions

    Returns
    -------
    out : (nx, ny) float64
        Column integral; units = data_unit * coord_unit (caller applies unyt).
    """
    # --- spatial filter -------------------------------------------------------
    # Margin based on the 99th-percentile effective cell radius (vol^1/3).
    # Cells further than this outside the bbox cannot affect any pixel.
    margin = 2.0 * float(np.percentile(vol_f ** (1.0 / 3.0), 99))
    keep   = ((Xf >= x0 - margin) & (Xf <= x1 + margin) &
              (Yf >= y0 - margin) & (Yf <= y1 + margin))
    n_in   = int(keep.sum())
    print(f"  voronoi proj  : {len(Xf):,} → {n_in:,} cells after bbox filter "
          f"(margin={margin:.3e})")
    if n_in == 0:
        raise ValueError("_voronoi_project: no cells inside the image bbox.")
    Xf     = Xf[keep];     Yf     = Yf[keep]
    data_f = data_f[keep]; vol_f  = vol_f[keep]
    # -------------------------------------------------------------------------

    pts2d = np.column_stack([Xf, Yf])
    far   = 3.0 * max(x1 - x0, y1 - y0)

    pts_aug, n_real = _add_sentinels(pts2d, x0, x1, y0, y1)
    vor = Voronoi(pts_aug)
    flat_verts, offsets, counts, gen_mask = _build_flat_cells(vor, n_real, far)

    gen_idx   = np.where(gen_mask)[0]
    data_vals = data_f[gen_idx].astype(np.float64)
    volumes   = vol_f[gen_idx].astype(np.float64)

    dx  = (x1 - x0) / nx
    dy  = (y1 - y0) / ny
    out = np.zeros((nx, ny), dtype=np.float64)

    _accumulate_proj(flat_verts, offsets, counts, data_vals, volumes,
                     x0, x1, y0, y1, dx, dy, x0, y0, nx, ny, out)

    return out
