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

"""``yt`` volume-rendering backend for RICH snapshots.

Turns a :class:`richio.render.grid.UniformGrid` into a ``yt`` dataset and
produces:

* :func:`volume_image` — a single depth-cued still (the opacity transfer
  function gives a sense of depth that a flat projection lacks);
* :func:`volume_movie` — a rotating-camera fly-around encoded to mp4/gif.

Notes
-----
``yt``'s volume renderer is **software** (CPU) — it does not use the GPU.
Parallelism therefore comes from two cheap, embarrassingly-parallel handles:

* the k-d tree resampling (:func:`richio.render.grid.to_uniform_grid`) is done
  **once** and reused for every frame of a movie;
* movie frames are independent, so they can be split across MPI ranks via the
  ``frame_indices`` argument (see ``scripts/render_movie_mpi.py``).
"""

import os
import tempfile

import numpy as np

from richio.render.grid import UniformGrid, to_uniform_grid

# Multiplier for all on-frame text (scale bar, colorbar, time label, axis triad).
# Default 1.0 keeps existing sizes; bump via RICHIO_FONT_SCALE for projector-readable
# text in conference renders (the per-element sizes are otherwise hardcoded below).
_FONT_SCALE = float(os.environ.get("RICHIO_FONT_SCALE", "1.0"))


def _require_yt():
    try:
        import yt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "richio.render's yt backend requires yt. "
            "Install the optional dependencies with:  pip install 'richio[render]'"
        ) from exc
    return yt


def _auto_bounds(values, log=True, pmin=8.0, pmax=99.9):
    """Pick transfer-function bounds from data percentiles.

    Ignores the (huge, near-empty) density floor by clipping to positive finite
    values, then spans from a low percentile (``pmin``, drops the diffuse
    background) up to ``pmax`` near the maximum so the **densest core is kept in
    range** rather than clipped — that clipping was why the innermost region was
    invisible with a tighter ``pmax``.
    """
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if v.size == 0:
        raise ValueError("No positive/finite values to set transfer-function bounds.")
    lo, hi = np.percentile(v, [pmin, pmax])
    if hi <= lo:
        hi = lo * 10 if log else lo + 1.0
    return float(lo), float(hi)


def _nice_ceil(x):
    """Smallest 1/2/5 × 10^k that is ≥ ``x`` — a clean axis limit."""
    if x <= 0:
        return 1.0
    k = np.floor(np.log10(x))
    for m in (1.0, 2.0, 5.0, 10.0):
        cand = m * 10.0 ** k
        if cand >= x:
            return float(cand)
    return float(10.0 ** (k + 1))


def _sym_bounds(values, pct=99.5, round_nice=False):
    """Symmetric bound ``V`` for a signed field, so the scale spans ``[-V, V]``.

    With ``round_nice`` the bound is rounded up to a clean 1/2/5×10^k value, so
    the symlog colorbar lands on tidy decade/half-decade ticks.
    """
    v = np.abs(np.asarray(values).ravel())
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    bound = float(np.percentile(v, pct)) or 1.0
    return _nice_ceil(bound) if round_nice else bound


def _mpl_norm(norm, vmin, vmax, linthresh=1.0):
    """Build the matplotlib color norm for ``"log"`` / ``"linear"`` / ``"symlog"``."""
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    if norm == "symlog":
        v = max(abs(vmin), abs(vmax))
        return SymLogNorm(linthresh=linthresh, vmin=-v, vmax=v, base=10)
    if norm == "linear":
        return Normalize(vmin=vmin, vmax=vmax)
    return LogNorm(vmin=vmin, vmax=vmax)


def _make_alpha_ramp(gamma):
    """Return an opacity ``scale_func`` ``alpha(value) = t**gamma`` (t normalised).

    ``gamma`` controls how much *diffuse* gas (mid densities) shows:

    * high ``gamma`` (e.g. ``3``) — steep; only the densest gas accumulates
      opacity, so the core/stream read solid and faint gas stays transparent;
    * low ``gamma`` (e.g. ``1.5``) — gentler; mid-density gas gets opacity too,
      giving more visible haze/volume.

    A near-zero floor is kept deliberately (``t=0 -> 0``) so empty space never
    fogs up the deep box — "more haze" comes from lowering ``gamma``, not from
    adding a floor.
    """

    def _ramp(values, mi, ma):
        t = np.clip((np.asarray(values) - mi) / (ma - mi), 0.0, 1.0)
        return t ** gamma

    return _ramp


def to_yt(grid: UniformGrid, *, dataset_name: str = "rich"):
    """Wrap a :class:`UniformGrid` as an in-memory ``yt`` dataset.

    :param grid: The uniform field cube produced by
                 :func:`richio.render.grid.to_uniform_grid`.
    :param dataset_name: Cosmetic name attached to the dataset.
    :returns: A ``yt`` stream dataset.  Fields live under ``("stream", name)``.
    """
    yt = _require_yt()
    data = {name: (grid.fields[name], grid.units[name]) for name in grid.fields}
    ds = yt.load_uniform_grid(
        data,
        grid.dims,
        length_unit=grid.length_unit,
        bbox=np.asarray(grid.bbox, dtype="float64"),
        nprocs=1,
        sim_time=float(grid.time) if grid.time is not None else 0.0,
        dataset_name=dataset_name,
    )
    return ds


def _rodrigues(vec, axis, angle):
    """Rotate *vec* about *axis* by *angle* radians (Rodrigues' formula)."""
    axis = np.asarray(axis, dtype="float64")
    axis = axis / np.linalg.norm(axis)
    vec = np.asarray(vec, dtype="float64")
    c, s = np.cos(angle), np.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _camera_offset(radius, azimuth_deg, elevation_deg):
    """World-space camera offset from the focus for a z-up spherical placement."""
    a = np.deg2rad(azimuth_deg)
    e = np.deg2rad(elevation_deg)
    return radius * np.array(
        [np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)], dtype="float64"
    )


def _camera_basis(azimuth_deg, elevation_deg, rot_axis, angle_deg):
    """Orthonormal screen basis ``(forward, up, right)`` for the orientation gizmo.

    Pure directions (no domain size needed), matching the same orbit math as
    :func:`_camera_vectors`: ``forward`` is the line of sight (into the screen),
    ``up`` the screen-vertical, ``right`` the screen-horizontal.  A world axis
    ``e`` projects to image coordinates ``(e·right, e·up)``.
    """
    off0 = _camera_offset(1.0, azimuth_deg, elevation_deg)
    north0 = np.array([0.0, 0.0, 1.0])
    ang = np.deg2rad(angle_deg)
    off = _rodrigues(off0, rot_axis, ang)
    north = _rodrigues(north0, rot_axis, ang)

    forward = -off / np.linalg.norm(off)
    up = north - np.dot(north, forward) * forward
    if np.linalg.norm(up) < 1e-6:  # looking down the pole — pick any up ⟂ forward
        alt = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        up = alt - np.dot(alt, forward) * forward
    up = up / np.linalg.norm(up)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    return forward, up, right


def _draw_triad(fig, basis, *, rect=(0.015, 0.015, 0.15, 0.15), flip_x=False):
    """Draw a small 3-D orientation gizmo (x/y/z arrows) in figure-corner *rect*."""
    forward, up, right = basis
    sx_sign = -1.0 if flip_x else 1.0  # match a horizontally-reversed image
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    axes = [
        (np.array([1.0, 0.0, 0.0]), "#ff5555", "x"),
        (np.array([0.0, 1.0, 0.0]), "#55ff55", "y"),
        (np.array([0.0, 0.0, 1.0]), "#5599ff", "z"),
    ]
    # Draw axes pointing away from the viewer first so nearer ones sit on top.
    for e, color, name in sorted(axes, key=lambda t: -float(np.dot(t[0], forward))):
        sx, sy = sx_sign * float(np.dot(e, right)), float(np.dot(e, up))
        depth = float(np.dot(e, forward))  # >0 ⇒ into the screen (away)
        alpha = 0.45 if depth > 0.05 else 1.0
        ax.annotate(
            "", xy=(sx, sy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, alpha=alpha),
        )
        ax.text(sx * 1.28, sy * 1.28, name, color=color, alpha=alpha,
                fontsize=11 * _FONT_SCALE, ha="center", va="center", weight="bold")


def _make_scene(
    grid,
    field,
    *,
    log,
    vmin,
    vmax,
    n_layers,
    cmap,
    grey_opacity,
    resolution,
    tf_mode="map",
    alpha=1.0,
    gamma=2.5,
):
    """Build a configured scene (dataset + volume source + transfer function).

    ``tf_mode``:

    * ``"map"`` (default) — a *continuous* colour map over ``[vmin, vmax]`` with
      opacity ramping up toward high density (:func:`_make_alpha_ramp`).  Reveals
      the internal density gradient, including the dense core, rather than a few
      isolated shells.
    * ``"layers"`` — yt's discrete Gaussian iso-density shells
      (``add_layers``); good for picking out specific level sets.

    ``alpha`` scales overall opacity (raise to make the volume more solid).
    """
    yt = _require_yt()
    ds = to_yt(grid)
    fld = ("stream", field)

    sc = yt.create_scene(ds, field=fld)
    source = sc[0]
    source.set_field(fld)
    source.set_log(log)

    lo, hi = _auto_bounds(grid.fields[field], log=log)
    if vmin is None:
        vmin = lo
    if vmax is None:
        vmax = hi
    b_lo, b_hi = (np.log10(vmin), np.log10(vmax)) if log else (vmin, vmax)

    tf = yt.ColorTransferFunction((b_lo, b_hi))
    if tf_mode == "layers":
        tf.add_layers(n_layers, colormap=cmap)
    else:
        tf.map_to_colormap(
            b_lo, b_hi, colormap=cmap, scale=alpha, scale_func=_make_alpha_ramp(gamma)
        )
    tf.grey_opacity = bool(grey_opacity)
    source.set_transfer_function(tf)

    cam = sc.camera
    cam.resolution = (int(resolution), int(resolution))
    return sc, ds, fld, vmin, vmax


def _camera_vectors(ds, length_unit, azimuth_deg, elevation_deg, rot_axis, angle_deg, zoom):
    """Orbit geometry shared by the volume camera and the off-axis projection.

    The base placement (``azimuth_deg``/``elevation_deg``) is rotated by
    ``angle_deg`` about ``rot_axis``; ``angle_deg`` is what a movie sweeps, so
    every frame is computed from absolute angles (stateless ⇒ safe to split
    arbitrary frame subsets across processes).

    :returns: ``(center, off, normal, north, width, depth)`` — all plain
              ndarrays / floats in *length_unit*.  ``off`` is the camera offset
              from the centre, ``normal`` the (unit) line of sight toward the
              centre, ``width`` the in-plane field of view and ``depth`` the
              along-axis integration length (full cube diagonal).
    """
    center = ds.domain_center.to(length_unit).value.astype("float64")
    extent = ds.domain_width.to(length_unit).value.astype("float64")
    L = float(np.max(extent))
    radius = 1.5 * L

    off0 = _camera_offset(radius, azimuth_deg, elevation_deg)
    north0 = np.array([0.0, 0.0, 1.0], dtype="float64")

    ang = np.deg2rad(angle_deg)
    off = _rodrigues(off0, rot_axis, ang)
    north = _rodrigues(north0, rot_axis, ang)
    normal = -off / np.linalg.norm(off)

    # Looking (anti)parallel to the up vector (e.g. exactly top-down) leaves the
    # camera roll undefined and yt rejects an aligned normal/north — pick a
    # fallback up axis orthogonal to the line of sight.
    nhat = north / np.linalg.norm(north)
    if abs(float(np.dot(normal, nhat))) > 0.999:
        alt = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        north = alt - np.dot(alt, normal) * normal
        north = north / np.linalg.norm(north)

    width = L / float(zoom)
    depth = float(np.sqrt(3.0) * L)
    return center, off, normal, north, width, depth


def _place_camera(cam, ds, length_unit, azimuth_deg, elevation_deg, rot_axis, angle_deg, zoom):
    """Position the volume-rendering camera on the orbit (see :func:`_camera_vectors`)."""
    center, off, _normal, north, width, _depth = _camera_vectors(
        ds, length_unit, azimuth_deg, elevation_deg, rot_axis, angle_deg, zoom
    )
    pos = center + off
    cam.focus = ds.arr(center, length_unit)
    cam.set_position(ds.arr(pos, length_unit), north_vector=north)
    cam.switch_orientation(normal_vector=(center - pos), north_vector=north)
    cam.set_width(ds.quan(width, length_unit))


def _scalar_display(ax, arr, cmap, norm, norm_name, flip_x):
    """Colour-map a 2-D scalar projection, rendering empty cells as background.

    ``LogNorm`` already treats ``<= 0`` as "bad" (so masked/empty cells fall to
    the black background).  For ``symlog`` zero maps to the *centre* of the
    colormap instead, which for a diverging map is white — indistinguishable
    from genuinely zero-energy gas and swallowing white overlays.  So for symlog
    we blank exact zeros (masked-out selection cells, or empty columns of a
    weighted projection) to ``NaN`` and paint them the background colour, keeping
    empty regions consistently black across all norms.  Returns the colormap used.
    """
    import matplotlib as mpl

    cmap_obj = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).copy()
    cmap_obj.set_bad("black")
    data = np.asarray(arr, dtype="float64")
    if norm_name == "symlog":
        data = np.where(data == 0.0, np.nan, data)
    ax.set_facecolor("black")
    ax.imshow(data.T, origin="lower", cmap=cmap_obj, norm=norm)
    if flip_x:
        ax.invert_xaxis()
    return cmap_obj


def _annotate_ax(ax, annotate):
    """Draw a corner text label (e.g. a snapshot time) on *ax*."""
    if annotate:
        ax.text(0.02, 0.965, annotate, transform=ax.transAxes, color="w",
                fontsize=14 * _FONT_SCALE, va="top", ha="left")


def _draw_scalebar(ax, frac, label, *, color="w"):
    """Draw a map-style horizontal scale bar at the bottom-centre of *ax*.

    *frac* is the bar length as a fraction of the image width and *label* the
    text drawn above it (e.g. ``"10 r_t"``).  Uses axes-fraction coordinates, so
    it sits in the same screen place regardless of an x-axis flip (a scale bar is
    symmetric anyway).  The caller sizes *frac* to a physical length using the
    rendered field of view (camera width).
    """
    if not frac:
        return
    frac = float(np.clip(frac, 0.02, 0.9))
    y = 0.05
    x0, x1 = 0.5 - frac / 2.0, 0.5 + frac / 2.0
    ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=color, lw=2.5 * _FONT_SCALE,
            solid_capstyle="butt", clip_on=False)
    for xt in (x0, x1):  # end caps
        ax.plot([xt, xt], [y - 0.012, y + 0.012], transform=ax.transAxes,
                color=color, lw=2.5 * _FONT_SCALE, clip_on=False)
    if label:
        ax.text(0.5, y + 0.022, label, transform=ax.transAxes, color=color,
                fontsize=12 * _FONT_SCALE, ha="center", va="bottom")


def _compose_with_colorbar(filename, img_or_data, *, is_rgb, cmap, norm, vmin, vmax,
                           label, linthresh=1.0, annotate=None, triad=None, flip_x=False,
                           scalebar_frac=None, scalebar_label=None):
    """Save *img_or_data* with a dark-themed matplotlib colorbar beside it.

    ``is_rgb=True``  → *img_or_data* is an already-rendered RGB(A) image (volume
    render); we just display it and draw the colorbar from ``cmap``/``vmin``/
    ``vmax``.  ``is_rgb=False`` → it is a 2-D scalar field (a projection) which we
    colour-map directly with the same norm (``"log"``/``"linear"``/``"symlog"``).

    We build the bar ourselves rather than using yt's ``save_annotated`` because
    that labels ticks with *linear* field values (all ~0 for these tiny
    densities); the matplotlib norm gives a correct (log/symlog) axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.cm import ScalarMappable
    import matplotlib.pyplot as plt

    norm_name = norm
    norm = _mpl_norm(norm, vmin, vmax, linthresh)

    if is_rgb:
        h, w = img_or_data.shape[0], img_or_data.shape[1]
    else:
        h, w = img_or_data.shape[0], img_or_data.shape[1]

    fig = plt.figure(figsize=(w / 100.0 * 1.14, h / 100.0), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 0.84, 1.0])
    if is_rgb:
        ax.imshow(img_or_data)
        if flip_x:
            ax.invert_xaxis()
        cmap_cb = cmap
    else:
        cmap_cb = _scalar_display(ax, img_or_data, cmap, norm, norm_name, flip_x)
    ax.axis("off")
    _annotate_ax(ax, annotate)
    _draw_scalebar(ax, scalebar_frac, scalebar_label)

    cax = fig.add_axes([0.865, 0.12, 0.022, 0.76])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_cb), cax=cax)
    cb.set_label(label, color="w", fontsize=12 * _FONT_SCALE)
    if norm_name == "symlog":
        # Default symlog ticks are too sparse (often only ±10^0, 0).  Add 1/2/5
        # majors + decade minors so the (a)symmetric log scaling is legible.
        from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator

        cb.ax.yaxis.set_major_locator(
            SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0, 2.0, 5.0)))
        cb.ax.yaxis.set_minor_locator(
            SymmetricalLogLocator(base=10, linthresh=linthresh, subs=tuple(range(1, 10))))
        cb.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: ("0" if v == 0 else f"{v:g}")))
        cb.update_ticks()
    cb.ax.yaxis.set_tick_params(color="w", labelcolor="w", which="both",
                                labelsize=10 * _FONT_SCALE)
    cb.ax.yaxis.set_tick_params(which="minor", length=2)
    cb.outline.set_edgecolor("w")
    if triad is not None:
        _draw_triad(fig, triad, flip_x=flip_x)
    fig.savefig(filename, facecolor="black", dpi=100)
    plt.close(fig)


def _save_scene(sc, filename, *, sigma_clip, colorbar, field, norm, grid, vmin, vmax, cmap,
                linthresh=1.0, annotate=None, triad=None, flip_x=False,
                scalebar_frac=None, scalebar_label=None):
    """Save volume-render scene *sc* to *filename* (optionally with a colorbar)."""
    if (not colorbar and not annotate and triad is None and not flip_x
            and not scalebar_frac):
        sc.save(filename, sigma_clip=sigma_clip)
        return

    import matplotlib.pyplot as plt

    tmp = filename + ".vrtmp.png"
    sc.save(tmp, sigma_clip=sigma_clip)
    try:
        img = plt.imread(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    unit = grid.units.get(field, "")
    label = f"{field}" + (f"  [{unit}]" if unit else "")
    _compose_with_colorbar(
        filename, img, is_rgb=True, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        label=label, linthresh=linthresh, annotate=annotate, triad=triad, flip_x=flip_x,
        scalebar_frac=scalebar_frac, scalebar_label=scalebar_label,
    )


def _make_projection(
    grid, field, *, azimuth, elevation, rot_axis, angle, zoom, resolution,
    weight=None, ds=None,
):
    """Off-axis line-of-sight integral of *field* from the camera orbit angle.

    Returns ``(arr, unit)`` where *arr* is the 2-D projected field
    (``∫ field dl`` for ``weight=None`` — i.e. column density for ``density``)
    and *unit* its unit string.  No transfer function / no absorption, so the
    result is purely additive and free of viewing-side occlusion.  Pass a
    pre-built *ds* (from :func:`to_yt`) to reuse it across many angles.
    """
    yt = _require_yt()
    if ds is None:
        ds = to_yt(grid)
    fld = ("stream", field)
    lu = grid.length_unit

    center, _off, normal, north, width, depth = _camera_vectors(
        ds, lu, azimuth, elevation, rot_axis, angle, zoom
    )
    wfield = ("stream", weight) if weight else None
    buff = yt.off_axis_projection(
        ds,
        ds.arr(center, lu),
        normal,
        ds.quan(width, lu),
        int(resolution),
        fld,
        weight=wfield,
        north_vector=north,
        depth=ds.quan(depth, lu),
        method="integrate",
    )
    arr = np.asarray(buff, dtype="float64")
    unit = str(getattr(buff, "units", "")) or grid.units.get(field, "")
    return arr, unit


def _save_projection(arr, filename, *, colorbar, cmap, norm, vmin, vmax, label,
                     linthresh=1.0, annotate=None, triad=None, flip_x=False,
                     scalebar_frac=None, scalebar_label=None):
    """Colour-map a 2-D projection *arr* and save it (optionally with a colorbar)."""
    if colorbar:
        _compose_with_colorbar(
            filename, arr, is_rgb=False, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
            label=label, linthresh=linthresh, annotate=annotate, triad=triad, flip_x=flip_x,
            scalebar_frac=scalebar_frac, scalebar_label=scalebar_label,
        )
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm_name = norm
    norm = _mpl_norm(norm, vmin, vmax, linthresh)
    h, w = arr.shape
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    _scalar_display(ax, arr, cmap, norm, norm_name, flip_x)
    ax.axis("off")
    _annotate_ax(ax, annotate)
    _draw_scalebar(ax, scalebar_frac, scalebar_label)
    if triad is not None:
        _draw_triad(fig, triad, flip_x=flip_x)
    fig.savefig(filename, facecolor="black", dpi=100)
    plt.close(fig)


def _projection_label(field, unit, weighted=False):
    if field == "bernoulli":
        return r"$B / \Delta\epsilon$"
    # Plain integral => column quantity; weighted => line-of-sight mean.
    prefix = "" if weighted else "column "
    return f"{prefix}{field}" + (f"  [{unit}]" if unit else "")


def volume_image(
    snap,
    field: str = "density",
    res: int = 256,
    *,
    grid: UniformGrid | None = None,
    box_size="auto",
    selection=None,
    unit_system: str = "cgs",
    mode: str = "volume",
    weight: str | None = None,
    log: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    tf_mode: str = "map",
    alpha: float = 20.0,
    gamma: float = 2.5,
    n_layers: int = 5,
    cmap: str = "magma",
    norm: str = "log",
    linthresh: float = 1.0,
    grey_opacity: bool = True,
    colorbar: bool = True,
    sigma_clip: float | None = 4.0,
    resolution: int = 1024,
    azimuth: float = 0.0,
    elevation: float = 20.0,
    zoom: float = 1.4,
    rot_axis=(0.0, 0.0, 1.0),
    annotate: str | None = None,
    axis_triad: bool = False,
    flip_x: bool = False,
    scalebar_frac: float | None = None,
    scalebar_label: str | None = None,
    filename: str | None = None,
    return_scene: bool = False,
):
    """Render a single image of *field* — depth-cued volume render or projection.

    :param mode: ``"volume"`` (default) — emission–absorption volume render with
                 the opacity transfer function (depth/solidity, but the
                 ``grey_opacity`` absorption makes brightness depend on viewing
                 side).  ``"projection"`` — off-axis line-of-sight integral
                 (``∫field dl``; column density for ``density``): additive, no
                 absorption, no occlusion, quantitative.  In projection mode the
                 transfer-function knobs (``alpha``/``gamma``/``grey_opacity``/
                 ``n_layers``/``sigma_clip``) are ignored.
    :param weight: Projection weight field (``mode="projection"`` only).
                   ``None`` (default) gives the plain integral (column density);
                   e.g. ``"density"`` gives a density-weighted mean.

    :param snap: A loaded :class:`richio.data.Snapshot`.
    :param field: Field to render (default ``"density"``).
    :param res: Grid resolution for the resampling (cubic, or ``(nx,ny,nz)``).
    :param grid: Pre-built :class:`UniformGrid` to reuse; if ``None`` one is
                 built from *snap*.
    :param box_size: Domain bounds for resampling.  ``"auto"`` (default) fits a
                     tight box around the dense region (:func:`tight_box`);
                     ``None`` uses the full ``snap.box``; or pass explicit
                     ``[x0,y0,z0,x1,y1,z1]``.
    :param selection: Boolean cell mask passed through to the resampling.
    :param unit_system: Base unit system for field values (default ``"cgs"``).
    :param log: Render in log space (default ``True``).
    :param vmin, vmax: Transfer-function bounds (in the field's units, *not*
                       log).  ``None`` auto-selects from data percentiles
                       (spanning up to near the max so the dense core is kept).
    :param tf_mode: ``"map"`` (default) for a continuous colour map with opacity
                    ramping toward high density (shows the core), or
                    ``"layers"`` for discrete iso-density shells.
    :param alpha: Overall opacity scale for ``tf_mode="map"``.
    :param n_layers: Number of shells when ``tf_mode="layers"``.
    :param cmap: Colormap for the transfer function (default ``"twilight"``,
                 richio's default).
    :param grey_opacity: If ``True``, opacity accumulates with depth (more
                         solid look); if ``False``, emission-only.
    :param colorbar: If ``True`` (default) composite a matplotlib **log**
                     colorbar (``LogNorm``) beside the image.
    :param sigma_clip: Brightness clip (higher ⇒ darker, less blown-out).
    :param resolution: Output image side length in pixels.
    :param azimuth: Camera azimuth in degrees.
    :param elevation: Camera elevation above the x-y plane in degrees.
    :param zoom: Zoom factor (>1 enlarges the object).
    :param rot_axis: Axis the (azimuth/elevation) base view is placed relative
                     to — also the orbit axis used by :func:`volume_movie`.
    :param filename: If given, save the rendered PNG here.
    :param return_scene: If ``True`` also return the ``yt`` ``Scene`` for
                         further customisation.
    :returns: ``grid`` (the :class:`UniformGrid` used), or ``(scene, grid)`` if
              *return_scene*.
    """
    if grid is None:
        grid = to_uniform_grid(
            snap, field, res=res, box_size=box_size, selection=selection,
            unit_system=unit_system,
        )

    triad = _camera_basis(azimuth, elevation, rot_axis, 0.0) if axis_triad else None

    if mode == "projection":
        arr, unit = _make_projection(
            grid, field, azimuth=azimuth, elevation=elevation, rot_axis=rot_axis,
            angle=0.0, zoom=zoom, resolution=resolution, weight=weight,
        )
        # Only derive the bounds that were not supplied.  Deriving them from this
        # frame's data would fail on an empty frame (e.g. a selection with no
        # cells in the box early on); when both limits are fixed upstream — as
        # evolution_movie always does — we must not touch the data at all.
        auto_lo = auto_hi = None
        if vmin is None or vmax is None:
            if norm == "symlog":
                v = _sym_bounds(arr, round_nice=True)
                auto_lo, auto_hi = -v, v
            else:
                auto_lo, auto_hi = _auto_bounds(arr, log=(norm == "log"))
        vlo = vmin if vmin is not None else auto_lo
        vhi = vmax if vmax is not None else auto_hi
        if filename is not None:
            _save_projection(arr, filename, colorbar=colorbar, cmap=cmap, norm=norm,
                             vmin=vlo, vmax=vhi, linthresh=linthresh,
                             label=_projection_label(field, unit, weighted=weight is not None),
                             annotate=annotate, triad=triad, flip_x=flip_x,
                             scalebar_frac=scalebar_frac, scalebar_label=scalebar_label)
        return grid

    sc, ds, _, vlo, vhi = _make_scene(
        grid, field, log=log, vmin=vmin, vmax=vmax, n_layers=n_layers,
        cmap=cmap, grey_opacity=grey_opacity, resolution=resolution,
        tf_mode=tf_mode, alpha=alpha, gamma=gamma,
    )
    _place_camera(
        sc.camera, ds, grid.length_unit, azimuth, elevation, rot_axis, 0.0, zoom
    )

    if filename is not None:
        _save_scene(sc, filename, sigma_clip=sigma_clip, colorbar=colorbar,
                    field=field, norm=norm, grid=grid, vmin=vlo, vmax=vhi, cmap=cmap,
                    linthresh=linthresh, annotate=annotate, triad=triad, flip_x=flip_x,
                    scalebar_frac=scalebar_frac, scalebar_label=scalebar_label)

    if return_scene:
        return sc, grid
    return grid


def volume_movie(
    snap,
    field: str = "density",
    res: int = 256,
    *,
    grid: UniformGrid | None = None,
    box_size="auto",
    selection=None,
    unit_system: str = "cgs",
    mode: str = "volume",
    weight: str | None = None,
    n_frames: int = 180,
    total_angle: float = 360.0,
    rot_axis=(0.0, 0.0, 1.0),
    azimuth: float = 0.0,
    elevation: float = 20.0,
    zoom: float = 1.4,
    log: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    tf_mode: str = "map",
    alpha: float = 20.0,
    gamma: float = 2.5,
    n_layers: int = 5,
    cmap: str = "magma",
    norm: str = "log",
    linthresh: float = 1.0,
    grey_opacity: bool = True,
    colorbar: bool = False,
    sigma_clip: float | None = 4.0,
    resolution: int = 1024,
    filename: str = "render.mp4",
    fps: int = 30,
    frames_dir: str | None = None,
    frame_indices=None,
    encode: bool = True,
    keep_frames: bool = False,
    n_jobs: int = 1,
    verbose: bool = True,
):
    """Render a rotating-camera fly-around of *field* and encode it to a movie.

    The camera orbits the domain centre about ``rot_axis`` through
    ``total_angle`` degrees over ``n_frames`` frames.  Each frame is computed
    from an **absolute** angle, so rendering an arbitrary subset
    (``frame_indices``) is safe — that is the hook used for MPI frame-parallel
    rendering on a cluster (see ``scripts/render_movie_mpi.py``).

    :param n_frames: Total frames in the full movie.
    :param total_angle: Sweep angle in degrees (``360`` = full turn).
    :param frames_dir: Directory for per-frame PNGs.  A temp dir is used when
                       ``None`` and frames are deleted unless *keep_frames*.
    :param frame_indices: Iterable of frame indices to render in this call
                          (default: all).  Used to split work across ranks.
    :param encode: If ``True``, stitch the PNGs in *frames_dir* into *filename*.
                   Set ``False`` on worker ranks; have one rank encode afterwards.
    :param keep_frames: Keep the PNG frames after encoding.
    :param n_jobs: Render frames across this many forked worker processes on the
                   local node (``yt``'s renderer is single-threaded, so this is
                   the cheap way to use a multi-core node).  The grid is shared
                   copy-on-write.  ``1`` (default) renders serially.
    :param fps: Frames per second of the output movie.
    :param filename: Output movie path (``.mp4``, ``.gif``, ...).
    :returns: ``dict`` with keys ``frames_dir``, ``frame_paths`` (those rendered
              here) and ``filename`` (``None`` if not encoded).

    All transfer-function / camera keywords match :func:`volume_image`.
    """
    if grid is None:
        grid = to_uniform_grid(
            snap, field, res=res, box_size=box_size, selection=selection,
            unit_system=unit_system,
        )

    angles = azimuth + np.linspace(0.0, total_angle, n_frames, endpoint=False)
    if frame_indices is None:
        frame_indices = range(n_frames)
    frame_indices = list(frame_indices)

    own_temp = frames_dir is None
    if own_temp:
        frames_dir = tempfile.mkdtemp(prefix="richio_vr_")
    else:
        os.makedirs(frames_dir, exist_ok=True)

    # In projection mode fix the colour scale once (from a few sample angles) so
    # every frame — and every parallel worker — shares identical vmin/vmax.
    if mode == "projection" and (vmin is None or vmax is None):
        samples = []
        for frac in (0.0, 0.25, 0.5):
            a = azimuth + frac * total_angle
            arr, _u = _make_projection(
                grid, field, azimuth=0.0, elevation=elevation, rot_axis=rot_axis,
                angle=a, zoom=zoom, resolution=resolution, weight=weight,
            )
            samples.append(np.asarray(arr).ravel())
        allv = np.concatenate(samples)
        if norm == "symlog":
            v = _sym_bounds(allv, round_nice=True)
            rlo, rhi = -v, v
        else:
            rlo, rhi = _auto_bounds(allv, log=(norm == "log"))
        vmin = rlo if vmin is None else vmin
        vmax = rhi if vmax is None else vmax

    scene_kw = dict(
        log=log, vmin=vmin, vmax=vmax, tf_mode=tf_mode, alpha=alpha, gamma=gamma,
        n_layers=n_layers, cmap=cmap, grey_opacity=grey_opacity,
        resolution=resolution,
    )
    cam_kw = dict(elevation=elevation, rot_axis=rot_axis, zoom=zoom,
                  mode=mode, weight=weight, norm=norm, linthresh=linthresh)

    if n_jobs and n_jobs > 1 and len(frame_indices) > 1:
        frame_paths = _render_frames_parallel(
            grid, field, frame_indices, angles, n_frames, frames_dir,
            scene_kw, cam_kw, sigma_clip, colorbar, n_jobs, verbose,
        )
    else:
        frame_paths = _render_frame_subset(
            grid, field, frame_indices, angles, n_frames, frames_dir,
            scene_kw, cam_kw, sigma_clip, colorbar, verbose,
        )

    out = {"frames_dir": frames_dir, "frame_paths": frame_paths, "filename": None}

    if encode:
        out["filename"] = _encode_movie(frames_dir, n_frames, filename, fps, verbose)
        if own_temp and not keep_frames:
            _cleanup_frames(frames_dir)

    return out


def _render_frame_subset(
    grid, field, frame_indices, angles, n_frames, frames_dir,
    scene_kw, cam_kw, sigma_clip, colorbar, verbose,
):
    """Render *frame_indices* to PNGs (single process); branch on render mode."""
    if cam_kw.get("mode") == "projection":
        return _render_projection_frames(
            grid, field, frame_indices, angles, n_frames, frames_dir,
            scene_kw, cam_kw, colorbar, verbose,
        )

    sc, ds, _, vlo, vhi = _make_scene(grid, field, **scene_kw)
    frame_paths = []
    for idx in frame_indices:
        _place_camera(
            sc.camera, ds, grid.length_unit, 0.0, cam_kw["elevation"],
            cam_kw["rot_axis"], angles[idx], cam_kw["zoom"],
        )
        path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        _save_scene(sc, path, sigma_clip=sigma_clip, colorbar=colorbar,
                    field=field, norm=cam_kw["norm"], linthresh=cam_kw["linthresh"],
                    grid=grid, vmin=vlo, vmax=vhi, cmap=scene_kw["cmap"])
        frame_paths.append(path)
        if verbose:
            print(f"[richio.render] frame {idx + 1}/{n_frames} -> {path}", flush=True)
    return frame_paths


def _render_projection_frames(
    grid, field, frame_indices, angles, n_frames, frames_dir,
    scene_kw, cam_kw, colorbar, verbose,
):
    """Render off-axis projection frames (one yt dataset reused for all angles)."""
    ds = to_yt(grid)
    vlo, vhi = scene_kw["vmin"], scene_kw["vmax"]
    frame_paths = []
    for idx in frame_indices:
        arr, unit = _make_projection(
            grid, field, ds=ds, azimuth=0.0, elevation=cam_kw["elevation"],
            rot_axis=cam_kw["rot_axis"], angle=angles[idx], zoom=cam_kw["zoom"],
            resolution=scene_kw["resolution"], weight=cam_kw["weight"],
        )
        path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        _save_projection(arr, path, colorbar=colorbar, cmap=scene_kw["cmap"],
                         norm=cam_kw["norm"], linthresh=cam_kw["linthresh"],
                         vmin=vlo, vmax=vhi,
                         label=_projection_label(field, unit,
                                                 weighted=cam_kw["weight"] is not None))
        frame_paths.append(path)
        if verbose:
            print(f"[richio.render] frame {idx + 1}/{n_frames} -> {path}", flush=True)
    return frame_paths


# Populated in the parent before forking so workers inherit the (large) grid
# via copy-on-write instead of pickling it across the process boundary.
_MP_STATE: dict = {}


def _mp_render_chunk(chunk):
    s = _MP_STATE
    return _render_frame_subset(
        s["grid"], s["field"], chunk, s["angles"], s["n_frames"],
        s["frames_dir"], s["scene_kw"], s["cam_kw"], s["sigma_clip"],
        s["colorbar"], verbose=False,
    )


def _render_frames_parallel(
    grid, field, frame_indices, angles, n_frames, frames_dir,
    scene_kw, cam_kw, sigma_clip, colorbar, n_jobs, verbose,
):
    """Render frames across *n_jobs* forked workers (single node).

    Each worker builds its own yt scene once and renders a round-robin slice of
    the frames.  The grid is shared copy-on-write through ``fork`` — set as a
    module global before the pool is created, never pickled.
    """
    import multiprocessing as mp

    n_jobs = min(n_jobs, len(frame_indices))
    chunks = [frame_indices[i::n_jobs] for i in range(n_jobs)]

    # Set the shared state in the parent BEFORE forking so children inherit the
    # (large) grid copy-on-write rather than receiving it pickled via initargs.
    _MP_STATE.update(
        grid=grid, field=field, angles=angles, n_frames=n_frames,
        frames_dir=frames_dir, scene_kw=scene_kw, cam_kw=cam_kw,
        sigma_clip=sigma_clip, colorbar=colorbar,
    )
    if verbose:
        print(
            f"[richio.render] rendering {len(frame_indices)} frames "
            f"across {n_jobs} workers",
            flush=True,
        )
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            results = pool.map(_mp_render_chunk, chunks)
    finally:
        _MP_STATE.clear()
    frame_paths = [p for sub in results for p in sub]
    return sorted(frame_paths)


def _encode_movie(frames_dir, n_frames, filename, fps, verbose=True):
    """Stitch ``frame_*.png`` in *frames_dir* into a movie via imageio."""
    import imageio.v2 as imageio

    paths = [os.path.join(frames_dir, f"frame_{i:05d}.png") for i in range(n_frames)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} to encode.")

    writer = imageio.get_writer(filename, fps=fps)
    try:
        for p in paths:
            writer.append_data(imageio.imread(p))
    finally:
        writer.close()
    if verbose:
        print(f"[richio.render] encoded {len(paths)} frames -> {filename}", flush=True)
    return filename


def _cleanup_frames(frames_dir):
    import shutil

    shutil.rmtree(frames_dir, ignore_errors=True)
