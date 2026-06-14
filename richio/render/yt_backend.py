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


def _compose_with_colorbar(filename, img_or_data, *, is_rgb, cmap, log, vmin, vmax, label):
    """Save *img_or_data* with a dark-themed matplotlib log colorbar beside it.

    ``is_rgb=True``  → *img_or_data* is an already-rendered RGB(A) image (volume
    render); we just display it and draw the colorbar from ``cmap``/``vmin``/
    ``vmax``.  ``is_rgb=False`` → it is a 2-D scalar field (a projection) which we
    colour-map directly with the same ``LogNorm``.

    We build the bar ourselves rather than using yt's ``save_annotated`` because
    that labels ticks with *linear* field values (all ~0 for these tiny
    densities); a :class:`~matplotlib.colors.LogNorm` gives a correct log axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize
    import matplotlib.pyplot as plt

    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)

    if is_rgb:
        h, w = img_or_data.shape[0], img_or_data.shape[1]
    else:
        h, w = img_or_data.shape[0], img_or_data.shape[1]

    fig = plt.figure(figsize=(w / 100.0 * 1.14, h / 100.0), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 0.84, 1.0])
    if is_rgb:
        ax.imshow(img_or_data)
    else:
        ax.imshow(img_or_data.T, origin="lower", cmap=cmap, norm=norm)
    ax.axis("off")

    cax = fig.add_axes([0.865, 0.12, 0.022, 0.76])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(label, color="w")
    cb.ax.yaxis.set_tick_params(color="w", labelcolor="w")
    cb.outline.set_edgecolor("w")
    fig.savefig(filename, facecolor="black", dpi=100)
    plt.close(fig)


def _save_scene(sc, filename, *, sigma_clip, colorbar, field, log, grid, vmin, vmax, cmap):
    """Save volume-render scene *sc* to *filename* (optionally with a colorbar)."""
    if not colorbar:
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
        filename, img, is_rgb=True, cmap=cmap, log=log, vmin=vmin, vmax=vmax, label=label
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


def _save_projection(arr, filename, *, colorbar, cmap, log, vmin, vmax, label):
    """Colour-map a 2-D projection *arr* and save it (optionally with a colorbar)."""
    if colorbar:
        _compose_with_colorbar(
            filename, arr, is_rgb=False, cmap=cmap, log=log, vmin=vmin, vmax=vmax, label=label
        )
        return

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LogNorm, Normalize
    import matplotlib.pyplot as plt

    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    h, w = arr.shape
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(arr.T, origin="lower", cmap=cmap, norm=norm)
    ax.axis("off")
    fig.savefig(filename, facecolor="black", dpi=100)
    plt.close(fig)


def _projection_label(field, unit):
    return f"column {field}" + (f"  [{unit}]" if unit else "")


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
    grey_opacity: bool = True,
    colorbar: bool = True,
    sigma_clip: float | None = 4.0,
    resolution: int = 1024,
    azimuth: float = 0.0,
    elevation: float = 20.0,
    zoom: float = 1.4,
    rot_axis=(0.0, 0.0, 1.0),
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

    if mode == "projection":
        arr, unit = _make_projection(
            grid, field, azimuth=azimuth, elevation=elevation, rot_axis=rot_axis,
            angle=0.0, zoom=zoom, resolution=resolution, weight=weight,
        )
        vlo, vhi = _auto_bounds(arr, log=log)
        if vmin is not None:
            vlo = vmin
        if vmax is not None:
            vhi = vmax
        if filename is not None:
            _save_projection(arr, filename, colorbar=colorbar, cmap=cmap, log=log,
                             vmin=vlo, vmax=vhi, label=_projection_label(field, unit))
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
                    field=field, log=log, grid=grid, vmin=vlo, vmax=vhi, cmap=cmap)

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
        rlo, rhi = _auto_bounds(np.concatenate(samples), log=log)
        vmin = rlo if vmin is None else vmin
        vmax = rhi if vmax is None else vmax

    scene_kw = dict(
        log=log, vmin=vmin, vmax=vmax, tf_mode=tf_mode, alpha=alpha, gamma=gamma,
        n_layers=n_layers, cmap=cmap, grey_opacity=grey_opacity,
        resolution=resolution,
    )
    cam_kw = dict(elevation=elevation, rot_axis=rot_axis, zoom=zoom,
                  mode=mode, weight=weight)

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
                    field=field, log=scene_kw["log"], grid=grid,
                    vmin=vlo, vmax=vhi, cmap=scene_kw["cmap"])
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
                         log=scene_kw["log"], vmin=vlo, vmax=vhi,
                         label=_projection_label(field, unit))
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
