"""Analysis helpers for native RICH simulation cells.

Optical-depth integrals and photospheres
--------------------------------------

The ray geometry is continuous, while extinction coefficients and cell sizes
are sampled from the nearest native cell. No uniform resampling grid is built.

This module provides array functions, with no CLI or import-time calculation.
It reads and writes no files. For example, to view a snapshot along an oblique
inward ray direction::

    coordinates = unyt.unyt_array([snap.X, snap.Y, snap.Z]).T
    alpha = richio.opacity.rosseland_alpha(snap.T, snap.rho)
    result = richio.analysis.find_photosphere(
        coordinates, alpha, snap.volume, snap.box,
        points=unyt.unyt_array([20, 0, 0], "Rsun"), direction=[1, 0, -1],
    )
    print(result["found"], result["position"])

Results are in-memory dictionaries; full output schemas and sampling
conventions are documented on each function below.
"""

import numpy as np
from scipy.spatial import KDTree
import unyt as u

from richio.units import units


def _lengths(value, length_unit):
    """Convert lengths, interpreting bare coordinates in RICH code units."""
    if not isinstance(value, u.unyt_array):
        value = u.unyt_array(value, units.lscale)
    return value.to(length_unit)


def _points(value, length_unit, name):
    value = _lengths(value, length_unit)
    if value.ndim not in (1, 2) or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (R, 3).")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite coordinates.")
    return value.reshape(-1, 3), value.ndim == 1


def _dimensionless(value, name, *, positive=False):
    if isinstance(value, u.unyt_array):
        value = value.to_value("dimensionless")
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar.")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def _prepare(coordinates, alpha, volume, tree):
    for name, value in (("coordinates", coordinates), ("alpha", alpha), ("volume", volume)):
        if not isinstance(value, u.unyt_array):
            raise TypeError(f"{name} must be a unitful unyt array.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError("coordinates must have shape (N, 3), with N > 0.")
    if coordinates.units.dimensions != u.dimensions.length:
        raise ValueError("coordinates must have length units.")
    if alpha.shape != (len(coordinates),) or volume.shape != (len(coordinates),):
        raise ValueError("alpha and volume must have shape (N,), matching coordinates.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite.")
    if alpha.units != coordinates.units**-1:
        alpha = alpha.to(coordinates.units**-1)
    if volume.units != coordinates.units**3:
        volume = volume.to(coordinates.units**3)
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("alpha must be finite and nonnegative.")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("volume must be finite and positive.")
    if tree is None:
        tree = KDTree(np.asarray(coordinates))
    elif tree.n != len(coordinates) or tree.m != 3:
        raise ValueError("tree must contain the same N three-dimensional coordinates.")
    return alpha, volume ** (1 / 3), tree


def _settings(target_tau, cell_fraction, max_delta_tau, max_steps):
    if target_tau is not None:
        target_tau = _dimensionless(target_tau, "target_tau")
    cell_fraction = _dimensionless(cell_fraction, "cell_fraction", positive=True)
    if max_delta_tau is not None:
        max_delta_tau = _dimensionless(max_delta_tau, "max_delta_tau", positive=True)
    if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    return target_tau, cell_fraction, max_delta_tau


def _step_limit(alpha, cell_size, cell_fraction, max_delta_tau):
    limit = cell_fraction * cell_size
    if max_delta_tau is not None:
        with np.errstate(divide="ignore"):
            limit = np.minimum(limit, max_delta_tau / alpha)
    return limit


def _integrate(
    alpha, cell_size, tree, p0, p1, target_tau, cell_fraction, max_delta_tau, workers, max_steps
):
    """Advance active rays in batches; all physical arithmetic stays unitful."""
    length_unit = p0.units
    delta = p1 - p0
    length = np.sqrt(np.sum(delta**2, axis=1))
    direction = np.zeros(delta.shape)
    nonzero = length > 0 * length_unit
    direction[nonzero] = (delta[nonzero] / length[nonzero, None]).to_value("dimensionless")
    position = p0.copy()
    distance = u.unyt_array(np.zeros(len(p0)), length_unit)
    tau = u.unyt_array(np.zeros(len(p0)), "dimensionless")
    reached = np.full(len(p0), target_tau == 0)
    steps = np.zeros(len(p0), dtype=np.int64)
    cell = tree.query(np.asarray(position), workers=workers)[1]

    while np.any(active := (distance < length) & ~reached):
        rows = np.flatnonzero(active)
        if np.any(steps[rows] >= max_steps):
            raise RuntimeError("Optical-depth integration exceeded max_steps on an active ray.")
        remaining = length[rows] - distance[rows]
        alpha0 = alpha[cell[rows]]
        h = np.minimum(
            remaining, _step_limit(alpha0, cell_size[cell[rows]], cell_fraction, max_delta_tau)
        )
        endpoint = p0[rows] + direction[rows] * (distance[rows] + h)[:, None]
        endpoint_cell = tree.query(np.asarray(endpoint), workers=workers)[1]

        # A coarse exterior cell must not take a long step into a small, opaque
        # endpoint cell. Shorten and resample; do not reuse a rejected alpha1.
        for _ in range(max_steps):
            endpoint_limit = _step_limit(
                alpha[endpoint_cell], cell_size[endpoint_cell], cell_fraction, max_delta_tau
            )
            retry = h > endpoint_limit * (1 + 1e-12)
            if not np.any(retry):
                break
            h[retry] = endpoint_limit[retry]
            endpoint[retry] = (
                p0[rows[retry]]
                + direction[rows[retry]] * (distance[rows[retry]] + h[retry])[:, None]
            )
            endpoint_cell[retry] = tree.query(np.asarray(endpoint[retry]), workers=workers)[1]
        else:
            raise RuntimeError("Optical-depth endpoint refinement exceeded max_steps.")
        if np.any(np.all(endpoint == position[rows], axis=1)):
            raise RuntimeError("Optical-depth step is too small to advance its coordinates.")

        alpha1 = alpha[endpoint_cell]
        delta_tau = (0.5 * (alpha0 + alpha1) * h).to("dimensionless")
        hit = np.zeros(len(rows), dtype=bool)
        if target_tau is not None:
            hit = tau[rows] + delta_tau >= target_tau
            if np.any(hit):
                needed = target_tau - tau[rows[hit]]
                a0, a1, full_h = alpha0[hit], alpha1[hit], h[hit]
                discriminant = a0**2 + 2 * (a1 - a0) * needed / full_h
                discriminant = np.maximum(discriminant, 0 * a0.units**2)
                # Stable positive root of the integral of linearly varying alpha.
                h[hit] = np.minimum(2 * needed / (a0 + np.sqrt(discriminant)), full_h)
                delta_tau[hit] = needed
                reached[rows[hit]] = True

        distance[rows] += h
        finished = (~hit) & (h >= remaining)
        distance[rows[finished]] = length[rows[finished]]
        position[rows] = p0[rows] + direction[rows] * distance[rows, None]
        position[rows[finished]] = p1[rows[finished]]
        tau[rows] += delta_tau
        steps[rows] += 1
        cell[rows] = endpoint_cell

    return {
        "position": position,
        "distance": distance,
        "tau": tau,
        "reached": reached,
        "steps": steps,
    }


def _reshape_result(result, scalar):
    return {name: value[0] for name, value in result.items()} if scalar else result


def integrate_optical_depth(
    coordinates,
    alpha,
    volume,
    p0,
    p1,
    *,
    target_tau=None,
    cell_fraction=0.25,
    max_delta_tau=None,
    tree=None,
    workers=1,
    max_steps=100000,
):
    r"""Integrate native-cell extinction along straight, adaptively sampled rays.

    A step initially has length
    :math:`h=\min(f V_i^{1/3},\Delta\tau_{\max}/\alpha_i,L_{\rm remaining})`,
    using the nearest cell at the previous accepted point. The optical-depth
    bound is omitted when ``max_delta_tau=None``; zero alpha imposes no optical
    bound. If the proposed endpoint has a smaller allowed step, shorten and
    retry until both endpoint bounds hold. This guards sampled transitions;
    it does not locate Voronoi faces or guarantee detection of a thin cell
    skipped entirely between endpoints. ``V**(1/3)`` is a characteristic size,
    not an exact ray chord through an irregular cell.

    Each accepted interval adds the trapezoid ``(alpha0 + alpha1) * h / 2``.
    If it reaches ``target_tau``, solve the quadratic integral of linearly
    varying alpha inside that interval, then stop that ray. No history is
    retained. Native fields use nearest-neighbor sampling, not a gridded or
    smooth field interpolation. The linear interpolation applies only to
    the quadrature and the final within-interval crossing.

    :param coordinates: Unitful native centers, shape ``(N, 3)``.
    :param alpha: Unitful extinction coefficient, shape ``(N,)``, inverse length.
                  This is ``kappa * rho``, not opacity per gram.
    :param volume: Positive native cell volumes, shape ``(N,)``.
    :param p0: Start coordinate ``(3,)`` or batch ``(R, 3)``. Bare coordinates
               use RICH code-length units; unitful coordinates are converted.
    :param p1: End coordinate(s), with the same convention as ``p0``. A single
               endpoint broadcasts against a batch of the other endpoint.
    :param target_tau: Stop at this nonnegative optical depth; ``None``
                       integrates to every endpoint. Default ``None``.
    :param cell_fraction: Maximum step as a fraction of local ``V**(1/3)``.
    :param max_delta_tau: Optional local optical-depth step bound. Default
                          ``None`` avoids excessive steps in opaque interiors.
    :param tree: Optional SciPy KDTree/cKDTree with the same cell ordering and
                 numerical coordinates in **coordinates.units**. Only its
                 dimensions and number of cells are checked. Otherwise build
                 one tree and reuse it for all rays.
    :param workers: Threads used by nearest-neighbor queries.
    :param max_steps: Maximum accepted steps per ray (also bounds endpoint
                      retries); exceeding it raises ``RuntimeError``.
    :returns: Dictionary with ``position``, ``distance`` from p0 (both in the
              coordinate length unit), dimensionless ``tau``, boolean
              ``reached`` (whether the requested target was attained), and
              integer accepted ``steps``. A single ray returns a ``(3,)``
              position and scalar other fields; batches retain their ray axis.
              Without a target, ``reached`` is always false. The function
              does not clip rays to a box; use :func:`find_photosphere` for
              an outside-to-inside integration through a simulation domain.
    """
    alpha, cell_size, tree = _prepare(coordinates, alpha, volume, tree)
    target_tau, cell_fraction, max_delta_tau = _settings(
        target_tau, cell_fraction, max_delta_tau, max_steps
    )
    p0, scalar0 = _points(p0, coordinates.units, "p0")
    p1, scalar1 = _points(p1, coordinates.units, "p1")
    shape = np.broadcast_shapes(p0.shape, p1.shape)
    p0 = u.unyt_array(np.broadcast_to(p0, shape), coordinates.units)
    p1 = u.unyt_array(np.broadcast_to(p1, shape), coordinates.units)
    result = _integrate(
        alpha,
        cell_size,
        tree,
        p0,
        p1,
        target_tau,
        cell_fraction,
        max_delta_tau,
        workers,
        max_steps,
    )
    return _reshape_result(result, scalar0 and scalar1)


def find_photosphere(
    coordinates,
    alpha,
    volume,
    box,
    points,
    direction,
    *,
    target_tau=2 / 3,
    cell_fraction=0.25,
    max_delta_tau=0.1,
    tree=None,
    workers=1,
    max_steps=100000,
):
    r"""Find an optical-depth surface by entering the actual simulation box.

    Each ``point`` anchors an infinite line. ``direction`` is the **inward
    travel direction**, not the direction toward the observer: ``(0, 0, -1)``
    enters from the +z side and ``(0, 0, 1)`` from the -z side. Normalize it,
    intersect the line with ``box``, and integrate from entry to exit using
    :func:`integrate_optical_depth` until :math:`\tau=2/3` by default.

    There is no opacity assumed beyond the supplied box. The starting point
    for integration is a box face, irrespective of where the line anchor
    lies. Use the full snapshot box, not the plotted region or the extrema
    of native cell centers. All adaptive stepping, nearest-cell sampling,
    trapezoidal quadrature and final quadratic interpolation follow
    :func:`integrate_optical_depth`.

    :param coordinates: Unitful native centers, shape ``(N, 3)``.
    :param alpha: Native inverse-length extinction coefficients, shape ``(N,)``.
    :param volume: Positive native volumes, shape ``(N,)``.
    :param box: Actual bounds ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
                Bare bounds use RICH code-length units.
    :param points: Unitful or code-length line anchors, ``(3,)`` or ``(R, 3)``.
    :param direction: Nonzero inward travel vector ``(3,)``, or ``(R, 3)``
                      matching a batch of anchors. Its magnitude is ignored.
    :param target_tau: Nonnegative target optical depth; default ``2/3``.
    :param cell_fraction: Maximum fraction of the sampled cell size; ``0.25``.
    :param max_delta_tau: Local optical-depth step bound; ``0.1``. ``None``
                          disables this bound while retaining the size bound.
    :param tree: Optional tree in precisely ``coordinates.units`` and native
                 cell ordering, as in :func:`integrate_optical_depth`.
    :param workers: Threads used by nearest-neighbor queries.
    :param max_steps: Maximum accepted steps per ray; default ``100000``.
    :returns: Integral result dictionary plus ``found``, ``entry`` and ``exit``.
              ``position`` is NaN when the target is not reached. A thin ray
              retains its actual integrated ``tau`` and full traversed distance.
              A line missing the box has NaN position, entry, exit, distance
              and tau, false ``found``/``reached``, and zero steps. Output
              length units match ``coordinates``; scalar/batch shapes follow
              ``points``. ``distance`` measures travel inward from entry.
    """
    alpha, cell_size, tree = _prepare(coordinates, alpha, volume, tree)
    target_tau, cell_fraction, max_delta_tau = _settings(
        target_tau, cell_fraction, max_delta_tau, max_steps
    )
    if target_tau is None:
        raise ValueError("find_photosphere requires a target_tau.")
    length_unit = coordinates.units
    points, scalar = _points(points, length_unit, "points")
    box = _lengths(box, length_unit)
    if box.shape != (6,) or not np.all(np.isfinite(box)) or np.any(box[3:] <= box[:3]):
        raise ValueError("box must contain six finite, increasing per-axis bounds.")
    if isinstance(direction, u.unyt_array):
        direction = direction.to_value("dimensionless")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) and direction.shape != points.shape:
        raise ValueError("direction must have shape (3,) or match the ray batch (R, 3).")
    direction = np.broadcast_to(direction, points.shape)
    norm = np.linalg.norm(direction, axis=1)
    if not np.all(np.isfinite(norm)) or np.any(norm == 0):
        raise ValueError("direction must be finite and nonzero for every ray.")
    direction = direction / norm[:, None]

    lower = u.unyt_array(np.full(points.shape, -np.inf), length_unit)
    upper = u.unyt_array(np.full(points.shape, np.inf), length_unit)
    intersects = np.ones(len(points), dtype=bool)
    for axis in range(3):
        moving = direction[:, axis] != 0
        intersects &= moving | (
            (points[:, axis] >= box[axis]) & (points[:, axis] <= box[axis + 3])
        )
        a = (box[axis] - points[moving, axis]) / direction[moving, axis]
        b = (box[axis + 3] - points[moving, axis]) / direction[moving, axis]
        lower[moving, axis] = np.minimum(a, b)
        upper[moving, axis] = np.maximum(a, b)
    start, stop = np.max(lower, axis=1), np.min(upper, axis=1)
    intersects &= start <= stop
    entry = points + direction * start[:, None]
    exit_point = points + direction * stop[:, None]
    entry[~intersects] = np.nan
    exit_point[~intersects] = np.nan
    result = {
        "position": u.unyt_array(np.full(points.shape, np.nan), length_unit),
        "distance": u.unyt_array(np.full(len(points), np.nan), length_unit),
        "tau": u.unyt_array(np.full(len(points), np.nan), "dimensionless"),
        "reached": np.zeros(len(points), dtype=bool),
        "steps": np.zeros(len(points), dtype=np.int64),
    }
    if np.any(intersects):
        integrated = _integrate(
            alpha,
            cell_size,
            tree,
            entry[intersects],
            exit_point[intersects],
            target_tau,
            cell_fraction,
            max_delta_tau,
            workers,
            max_steps,
        )
        for name, value in integrated.items():
            result[name][intersects] = value
    result["found"] = result["reached"].copy()
    result["position"][~result["found"]] = np.nan
    result.update(entry=entry, exit=exit_point)
    return _reshape_result(result, scalar)
