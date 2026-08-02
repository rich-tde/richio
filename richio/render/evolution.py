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

"""Time-evolution movies — one frame per snapshot.

Where :func:`~richio.render.yt_backend.volume_movie` holds the snapshot fixed
and sweeps the camera, :func:`evolution_movie` holds the camera fixed (optionally
rotating) and advances the **snapshot**, so the movie shows the system evolving
in time.

To avoid flicker, the **box** and **colour scale** are fixed once from a
reference snapshot and reused for every frame; only the data (and, optionally,
the camera angle) change.  Unlike a rotation movie — which builds one grid and
reuses it — each frame here resamples a *different* snapshot, so frames are
distributed across worker processes (each loads its own snapshot).
"""

import os
import tempfile

import numpy as np

from richio.render.grid import densest_box, tight_box, to_uniform_grid
from richio.render.yt_backend import (
    _auto_bounds,
    _cleanup_frames,
    _encode_movie,
    _make_projection,
    _sym_bounds,
    volume_image,
)


def _read_tfb(path, snapnum):
    """Read the ``tfb_<n>.txt`` time (in fallback-time units) next to a snapshot."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        with open(os.path.join(d, f"tfb_{snapnum}.txt")) as fh:
            return float(fh.read().strip())
    except Exception:
        return None


def _days_per_tfb(ref_snap, ref_path):
    """Calibrate days-per-``tfb`` from a snapshot that has both a time and a tfb."""
    t = getattr(ref_snap, "time", None)
    if t is None:
        return None
    try:
        days = float(np.atleast_1d(t).to("day")[0])
    except Exception:
        return None
    tfb = _read_tfb(ref_path, ref_snap.snapnum)
    return days / tfb if tfb else None


def _calibrate_days_per_tfb(snapshots):
    """Find a snapshot with both a stored time and a tfb file, to convert tfb→days.

    Prefers the (light, early) ``.h5`` snapshots; NPY snapshots here don't store a
    time, so the conversion factor must come from an h5 one.
    """
    import richio

    h5s = [p for p in snapshots if p.endswith((".h5", ".hdf5"))]
    for p in h5s + list(snapshots):
        try:
            factor = _days_per_tfb(richio.load(p), p)
            if factor:
                return factor
        except Exception:
            continue
    return None


def _snapshot_days(snap, path, days_per_tfb):
    """Time in days for a snapshot: from its stored time, else tfb × calibration."""
    t = getattr(snap, "time", None)
    if t is not None:
        try:
            return float(np.atleast_1d(t).to("day")[0])
        except Exception:
            pass
    tfb = _read_tfb(path, snap.snapnum)
    if tfb is not None and days_per_tfb is not None:
        return tfb * days_per_tfb
    return None


def _evolution_label(snap, path, days_per_tfb):
    """Frame label: time in fallback times, time in days, and snapshot number.

    ``t_fb`` comes first because it is the physically meaningful clock for a TDE
    (and the only one every snapshot carries — NPY snapshots store no time, so the
    days line needs *days_per_tfb* from :func:`_calibrate_days_per_tfb`).  A line
    whose value is unavailable is simply dropped.
    """
    parts = []
    tfb = _read_tfb(path, getattr(snap, "snapnum", -1))
    if tfb is not None:
        parts.append(rf"$t = {tfb:.2f}\,t_{{\rm fb}}$")
    d = _snapshot_days(snap, path, days_per_tfb)
    if d is not None:
        parts.append(f"t = {d:.2f} d")
    n = getattr(snap, "snapnum", -1)
    if n >= 0:
        parts.append(f"snap {n}")
    return "\n".join(parts) or None


def _grid_fields(field, weight):
    """Fields to resample: the rendered field plus the weight field if distinct.

    A weighted projection (e.g. density-weighted temperature) needs the weight
    field present in the same yt dataset, so it must be on the grid too.
    """
    fields = [field]
    if weight and weight != field:
        fields.append(weight)
    return fields


def _build_grid(snap, field, weight, coords, res, box, unit_system, derived_kw,
                selection_fn=None, workers=8):
    """Resample the (possibly derived) field — and its weight — onto the grid.

    When *selection_fn* is given, cells outside the selection are **zeroed**
    before resampling (field *and* weight), so column maps show only the selected
    gas and density-weighted means use only selected cells.  We zero rather than
    drop cells: dropping would let the k-d-tree smear selected values across the
    emptied space via nearest-neighbour, inventing structure that isn't there.
    """
    from richio.render.derived import DERIVED_FIELDS

    cx, cy, cz = coords
    if field in DERIVED_FIELDS:
        fields = {field: DERIVED_FIELDS[field](snap, **derived_kw)}
        if weight and weight != field:
            fields[weight] = snap._get_data(weight)
    else:
        fields = _grid_fields(field, weight)

    if selection_fn is not None:
        mask = np.asarray(selection_fn(snap), dtype=bool)
        # Materialise to a {name: array} map (unit-bearing) so we can zero the
        # unselected cells while keeping units for the cgs conversion downstream.
        if not isinstance(fields, dict):
            fields = {f: snap._get_data(f) for f in fields}
        fields = {name: arr * mask for name, arr in fields.items()}

    return to_uniform_grid(
        snap, fields, res=res, X=cx, Y=cy, Z=cz,
        box_size=box, unit_system=unit_system, workers=workers,
    )


def _resolve_box(ref_snap, field, box, box_kind, disk_radius, coords, box_field=None):
    """Pick the fixed bounding box from the reference snapshot.

    *box_field* (if given) selects the field whose distribution defines the box,
    independent of the field being rendered — e.g. frame a dissipation movie on
    the *density* box so it matches the density movie.
    """
    if box is not None and not isinstance(box, str):
        return box
    bf = box_field or field
    if box_kind == "disk":
        return densest_box(ref_snap, bf, radius=disk_radius, coords=coords)
    return tight_box(ref_snap, bf, coords=coords)  # cubic, around the dense region


# Worker state set in the parent before forking (config is small + picklable, so
# it is also passed explicitly; the heavy per-snapshot data is loaded in-worker).
def _evo_render_frame(task):
    """Render a single evolution frame (own process loads its own snapshot)."""
    import richio

    idx, path, angle, cfg = task
    snap = richio.load(path)
    grid = _build_grid(snap, cfg["field"], cfg["weight"], cfg["coords"], cfg["res"],
                       cfg["box"], cfg["unit_system"], cfg["derived_kw"],
                       cfg["selection_fn"], cfg["workers"])
    annotate = _evolution_label(snap, path, cfg["days_per_tfb"]) if cfg["annotate_time"] else None
    out_path = os.path.join(cfg["frames_dir"], f"frame_{idx:05d}.png")
    _render_one(snap, grid, angle, annotate, out_path, cfg)
    return out_path


def _render_one(snap, grid, angle, annotate, out_path, cfg):
    """Render a single frame from a pre-built grid (shared by evolution + spins)."""
    volume_image(
        snap,
        cfg["field"],
        grid=grid,
        mode=cfg["mode"],
        weight=cfg["weight"],
        axis_triad=cfg["axis_triad"],
        flip_x=cfg["flip_x"],
        log=cfg["log"],
        norm=cfg["norm"],
        linthresh=cfg["linthresh"],
        vmin=cfg["vmin"],
        vmax=cfg["vmax"],
        cmap=cfg["cmap"],
        tf_mode=cfg["tf_mode"],
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        grey_opacity=cfg["grey_opacity"],
        colorbar=cfg["colorbar"],
        sigma_clip=cfg["sigma_clip"],
        resolution=cfg["resolution"],
        azimuth=angle,
        elevation=cfg["elevation"],
        zoom=cfg["zoom"],
        rot_axis=cfg["rot_axis"],
        annotate=annotate,
        scalebar_frac=cfg["scalebar_frac"],
        scalebar_label=cfg["scalebar_label"],
        filename=out_path,
    )


def _evo_render_job(job):
    """Render every frame for one snapshot — loaded once, grid built once.

    *job* = ``(path, frames, cfg)`` where *frames* is a list of
    ``(global_index, azimuth)``.  An ordinary evolution snapshot has a single
    frame; a snapshot that is also a spin point has its evolution frame followed
    by the azimuth sweep, all sharing the one grid (so the expensive resample
    happens once per snapshot regardless of how many frames it contributes).
    """
    import richio

    path, frames, cfg = job
    snap = richio.load(path)
    grid = _build_grid(snap, cfg["field"], cfg["weight"], cfg["coords"], cfg["res"],
                       cfg["box"], cfg["unit_system"], cfg["derived_kw"],
                       cfg["selection_fn"], cfg["workers"])
    annotate = _evolution_label(snap, path, cfg["days_per_tfb"]) if cfg["annotate_time"] else None
    out = []
    for gidx, angle in frames:
        out_path = os.path.join(cfg["frames_dir"], f"frame_{gidx:05d}.png")
        _render_one(snap, grid, float(angle), annotate, out_path, cfg)
        out.append(out_path)
    return out


def _build_spin_jobs(snapshots, angles, spin_at, spin_frames, final_spin_turns):
    """Plan the multi-spin movie as one job per snapshot, in time order.

    Returns ``(jobs, total)`` where *jobs* is a list of ``(path, frames)`` and
    *frames* is ``[(global_index, azimuth), ...]``.  Each snapshot gets its
    evolution frame; a snapshot in *spin_at* (or the last one) additionally gets
    a full-turn azimuth sweep — ``final_spin_turns`` turns for the final
    snapshot — at ``spin_frames`` frames per turn.  Global indices run
    sequentially in time order so the encoder stitches them straight through.
    """
    n = len(snapshots)
    spin_set = {int(i) % n for i in (spin_at or [])}
    jobs = []
    gidx = 0
    for i in range(n):
        frames = [(gidx, float(angles[i]))]
        gidx += 1
        turns = (1 if i in spin_set else 0) + (int(final_spin_turns) if i == n - 1 else 0)
        if turns > 0 and spin_frames > 0:
            nsf = turns * int(spin_frames)
            sweep = float(angles[i]) + np.linspace(0.0, 360.0 * turns, nsf, endpoint=False)
            for a in sweep[1:]:  # skip the first (duplicates the evolution frame)
                frames.append((gidx, float(a)))
                gidx += 1
        jobs.append((snapshots[i], frames))
    return jobs, gidx


def _run_spin_jobs(jobs, cfg, n_jobs, verbose):
    """Render every job (snapshot) — frame-parallel across snapshots."""
    tasks = [(path, frames, cfg) for path, frames in jobs]
    frame_paths = []
    if n_jobs and n_jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(n_jobs, len(tasks))) as pool:
            for k, paths in enumerate(pool.imap_unordered(_evo_render_job, tasks)):
                frame_paths.extend(paths)
                if verbose:
                    print(f"[evolution] job {k + 1}/{len(tasks)} done "
                          f"({len(paths)} frames)", flush=True)
    else:
        for t in tasks:
            paths = _evo_render_job(t)
            frame_paths.extend(paths)
            if verbose:
                print(f"[evolution] job -> {len(paths)} frames", flush=True)
    return frame_paths


def evolution_movie(
    snapshots,
    field: str = "density",
    *,
    mode: str = "projection",
    box=None,
    box_kind: str = "wide",
    box_field: str | None = None,
    disk_radius: float = 250.0,
    ref_index: int = -1,
    coords=("X", "Y", "Z"),
    res: int = 256,
    resolution: int = 1024,
    workers: int = 8,
    unit_system: str = "cgs",
    vmin: float | None = None,
    vmax: float | None = None,
    log: bool = True,
    norm: str = "log",
    linthresh: float = 1.0,
    m_bh: float = 1e4,
    m_star: float = 0.5,
    r_star: float = 0.47,
    cmap: str = "magma",
    weight: str | None = None,
    tf_mode: str = "map",
    alpha: float = 20.0,
    gamma: float = 2.5,
    grey_opacity: bool = True,
    sigma_clip: float | None = 4.0,
    colorbar: bool = True,
    annotate_time: bool = True,
    azimuth: float = 45.0,
    elevation: float = 26.0,
    zoom: float = 1.1,
    rotate: bool = False,
    total_angle: float = 360.0,
    spin_frames: int = 0,
    spin_angle: float = 360.0,
    spin_at=None,
    final_spin_turns: int = 1,
    rot_axis=(0.0, 0.0, 1.0),
    axis_triad: bool = True,
    flip_x: bool = False,
    selection_fn=None,
    scalebar_frac: float | None = None,
    scalebar_label: str | None = None,
    fps: int = 24,
    filename: str = "evolution.mp4",
    frames_dir: str | None = None,
    n_jobs: int = 1,
    encode: bool = True,
    keep_frames: bool = False,
    verbose: bool = True,
):
    """Render a time-evolution movie: one frame per snapshot in *snapshots*.

    :param snapshots: Ordered list of snapshot paths (one frame each).
    :param field: Field to render.
    :param mode: ``"projection"`` (column density, no occlusion) or ``"volume"``.
    :param box: Explicit ``[x0,y0,z0,x1,y1,z1]`` box held fixed across time;
                ``None`` derives it from the reference snapshot.
    :param box_kind: When *box* is ``None`` — ``"wide"`` (cubic :func:`tight_box`
                     around the debris) or ``"disk"`` (:func:`densest_box`
                     close-up of the central disk).
    :param box_field: Field whose distribution defines the auto box, if different
                      from the rendered *field*.  E.g. render a *dissipation*
                      movie with ``box_field="density"`` so it shares the density
                      movie's framing and the two are directly comparable.
    :param disk_radius: Half-width (code length) for ``box_kind="disk"``.
    :param ref_index: Which snapshot fixes the box and colour scale.  Default
                      ``-1`` (the last, i.e. largest/most-evolved).
    :param vmin, vmax: Fixed colour limits.  ``None`` derives them from the
                       reference snapshot so brightness is comparable over time.
    :param annotate_time: Stamp each frame with its snapshot time.
    :param rotate: If ``True`` the camera also orbits (``total_angle`` over the
                   sequence) while time advances; otherwise the camera is fixed.
    :param spin_frames: If ``>0``, append this many trailing frames that hold on
                        the final (reference) snapshot and orbit the camera
                        through ``spin_angle`` degrees — a smooth "evolution then
                        spin the final state" ending.  Starts at the same fixed
                        ``azimuth`` as the evolution, so the join is seamless.
    :param spin_angle: Sweep of the trailing spin in degrees (default ``360``).
    :param spin_at: Optional list of snapshot indices after which to insert a
                    full 360° camera spin (a "pause and look around" at chosen
                    times).  When given, the movie uses the multi-spin path: each
                    snapshot is loaded once and contributes its evolution frame
                    plus, if listed here (or last), a spin sweep — and the final
                    snapshot spins ``final_spin_turns`` times.  ``None`` keeps the
                    simple path with at most one trailing spin (``spin_frames``).
    :param final_spin_turns: Number of full turns at the final snapshot when
                             *spin_at* is used (``spin_frames`` frames per turn).
    :param selection_fn: Optional callable ``snap -> bool ndarray (N,)``.  Cells
                         outside the mask are zeroed before resampling, so only
                         the selected gas is rendered (see :func:`_build_grid`).
    :param scalebar_frac: Length of a map-style scale bar as a fraction of the
                          image width; ``None`` draws none.
    :param scalebar_label: Text label for the scale bar (e.g. ``"10 r_t"``).
    :param n_jobs: Worker processes (each loads its own snapshot).  Frames are
                   independent across snapshots ⇒ embarrassingly parallel.
    :param filename: Output movie path.
    :returns: ``dict`` with ``frames_dir``, ``frame_paths`` and ``filename``.

    Volume-only knobs (``alpha``/``gamma``/``tf_mode``/``grey_opacity``/
    ``sigma_clip``) are ignored in projection mode; ``weight`` applies only to
    projection.
    """
    import richio

    snapshots = list(snapshots)
    n = len(snapshots)
    if n == 0:
        raise ValueError("No snapshots given.")

    own_temp = frames_dir is None
    frames_dir = tempfile.mkdtemp(prefix="richio_evo_") if own_temp else frames_dir
    os.makedirs(frames_dir, exist_ok=True)

    derived_kw = dict(m_bh=m_bh, m_star=m_star, r_star=r_star, coords=tuple(coords))

    # Reference snapshot fixes the box and the colour scale (avoids flicker); it
    # is also the snapshot spun at the end (the most-evolved final state).
    ref_snap = richio.load(snapshots[ref_index])
    fixed_box = _resolve_box(ref_snap, field, box, box_kind, disk_radius, coords, box_field)

    ref_grid = None
    if vmin is None or vmax is None or spin_frames > 0:
        ref_grid = _build_grid(ref_snap, field, weight, coords, res, fixed_box,
                               unit_system, derived_kw, selection_fn, workers)
    if vmin is None or vmax is None:
        if mode == "projection":
            arr, _u = _make_projection(
                ref_grid, field, azimuth=azimuth, elevation=elevation,
                rot_axis=rot_axis, angle=0.0, zoom=zoom, resolution=resolution,
                weight=weight,
            )
            ref_vals = arr
        else:
            ref_vals = ref_grid.fields[field]
        if norm == "symlog":
            v = _sym_bounds(ref_vals, round_nice=True)
            rlo, rhi = -v, v
        else:
            rlo, rhi = _auto_bounds(ref_vals, log=(norm == "log"))
        vmin = rlo if vmin is None else vmin
        vmax = rhi if vmax is None else vmax

    if rotate:
        angles = azimuth + np.linspace(0.0, total_angle, n, endpoint=False)
    else:
        angles = np.full(n, azimuth)

    days_per_tfb = _days_per_tfb(ref_snap, snapshots[ref_index])
    if days_per_tfb is None:
        days_per_tfb = _calibrate_days_per_tfb(snapshots)

    cfg = dict(
        field=field, mode=mode, weight=weight, coords=tuple(coords),
        res=res, resolution=resolution, derived_kw=derived_kw,
        unit_system=unit_system, box=fixed_box, vmin=vmin, vmax=vmax, log=log,
        norm=norm, linthresh=linthresh,
        cmap=cmap, tf_mode=tf_mode, alpha=alpha, gamma=gamma,
        grey_opacity=grey_opacity, colorbar=colorbar, sigma_clip=sigma_clip,
        elevation=elevation, zoom=zoom, rot_axis=rot_axis,
        annotate_time=annotate_time, axis_triad=axis_triad, flip_x=flip_x,
        days_per_tfb=days_per_tfb, frames_dir=frames_dir,
        selection_fn=selection_fn, workers=workers,
        scalebar_frac=scalebar_frac, scalebar_label=scalebar_label,
    )

    # Multi-spin path: interleave evolution frames with full-turn camera spins at
    # chosen snapshots (and a multi-turn spin at the end).  Each snapshot loads
    # once and renders all its frames; the whole list is frame-parallel.
    if spin_at is not None:
        jobs, total = _build_spin_jobs(snapshots, angles, spin_at, spin_frames,
                                       final_spin_turns)
        frame_paths = _run_spin_jobs(jobs, cfg, n_jobs, verbose)
        out = {"frames_dir": frames_dir, "frame_paths": sorted(frame_paths),
               "filename": None}
        if encode:
            out["filename"] = _encode_movie(frames_dir, total, filename, fps, verbose)
            if own_temp and not keep_frames:
                _cleanup_frames(frames_dir)
        return out

    tasks = [(i, snapshots[i], float(angles[i]), cfg) for i in range(n)]

    if verbose:
        print(f"[evolution] {n} frames, mode={mode}, box_kind={box_kind}, "
              f"rotate={rotate}, n_jobs={n_jobs}", flush=True)

    frame_paths = []
    if n_jobs and n_jobs > 1 and n > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(n_jobs, n)) as pool:
            for k, p in enumerate(pool.imap_unordered(_evo_render_frame, tasks)):
                frame_paths.append(p)
                if verbose:
                    print(f"[evolution] {k + 1}/{n} frames done", flush=True)
    else:
        for t in tasks:
            frame_paths.append(_evo_render_frame(t))
            if verbose:
                print(f"[evolution] {len(frame_paths)}/{n} -> {frame_paths[-1]}", flush=True)

    # Trailing "spin": hold on the final (reference) snapshot and orbit the
    # camera, so the movie resolves the final state instead of cutting off.  The
    # spin starts at the same fixed `azimuth` as the evolution, so the join is
    # seamless, and reuses the same box, colour scale and reference grid.
    total = n + spin_frames
    if spin_frames > 0:
        ref_label = (
            _evolution_label(ref_snap, snapshots[ref_index], days_per_tfb)
            if annotate_time else None
        )
        spin_angles = azimuth + np.linspace(0.0, spin_angle, spin_frames, endpoint=False)
        for k in range(spin_frames):
            path = os.path.join(frames_dir, f"frame_{n + k:05d}.png")
            volume_image(
                ref_snap, field, grid=ref_grid, mode=mode, weight=weight,
                axis_triad=axis_triad, flip_x=flip_x,
                log=log, norm=norm, linthresh=linthresh,
                vmin=vmin, vmax=vmax, cmap=cmap, tf_mode=tf_mode,
                alpha=alpha, gamma=gamma, grey_opacity=grey_opacity,
                colorbar=colorbar, sigma_clip=sigma_clip, resolution=resolution,
                azimuth=float(spin_angles[k]), elevation=elevation, zoom=zoom,
                rot_axis=rot_axis, annotate=ref_label,
                scalebar_frac=scalebar_frac, scalebar_label=scalebar_label,
                filename=path,
            )
            frame_paths.append(path)
            if verbose:
                print(f"[evolution] spin {k + 1}/{spin_frames} -> {path}", flush=True)

    out = {"frames_dir": frames_dir, "frame_paths": sorted(frame_paths), "filename": None}
    if encode:
        out["filename"] = _encode_movie(frames_dir, total, filename, fps, verbose)
        if own_temp and not keep_frames:
            _cleanup_frames(frames_dir)
    return out
