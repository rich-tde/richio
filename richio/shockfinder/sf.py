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
#
#  The shockfinder algorithm in RICHIO is based on Schaal+14
#  <https://arxiv.org/abs/1407.4117>.

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math as _math
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

from numba import njit, prange
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Voronoi, cKDTree

if TYPE_CHECKING:
    from richio.data import Snapshot

# Scalar-or-array argument/return for the closed-form Rankine-Hugoniot helpers.
Floats = float | NDArray[np.floating]

# ---------------------------------------------------------------------------- #
#                             Analytical Functions                             #
# ---------------------------------------------------------------------------- #


def delta(M: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Dimensionless entropy jump across a shock of Mach number *M*.

    :param M: Mach number upstream of the shock (M ≥ 1).
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Dimensionless entropy jump δ.
    :rtype: float or array-like
    """
    R = 1 / ((gamma - 1) / (gamma + 1) + 2 / (gamma + 1) / M**2)  # R = rho2/rho1
    delta = (
        2
        / (gamma * (gamma - 1) * M**2 * R)
        * ((2 * gamma * M**2 - (gamma - 1)) / (gamma + 1) - R**gamma)
    )
    return delta


def R2M(R: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Mach number from density compression ratio ρ₂/ρ₁ (Rankine-Hugoniot).

    :param R: Density ratio ρ₂/ρ₁ across the shock.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Mach number M ≥ 1.
    :rtype: float or array-like
    """
    return 1 / np.sqrt((gamma + 1) / (2 * R) - (gamma - 1) / 2)


rho2rho1M = R2M  # alias


def M2R(M: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Density compression ratio ρ₂/ρ₁ from Mach number (Rankine-Hugoniot).

    :param M: Mach number upstream of the shock.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Density ratio ρ₂/ρ₁.
    :rtype: float or array-like
    """
    return (gamma + 1) * M**2 / ((gamma - 1) * M**2 + 2)


Mrho2rho1 = M2R  # alias


def MT2T1(M: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Temperature jump T₂/T₁ across a shock of Mach number *M*.

    :param M: Upstream Mach number.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Temperature ratio T₂/T₁.
    :rtype: float or array-like
    """
    return (2 * gamma * M**2 - (gamma - 1)) * ((gamma - 1) * M**2 + 2) / ((gamma + 1) ** 2 * M**2)


def MP2P1(M: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Pressure jump P₂/P₁ across a shock of Mach number *M*.

    :param M: Upstream Mach number.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Pressure ratio P₂/P₁.
    :rtype: float or array-like
    """
    return (2 * gamma * M**2) / (gamma + 1) - (gamma - 1) / (gamma + 1)


def T2T1M(T2_T1: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Mach number inferred from temperature jump T₂/T₁.

    :param T2_T1: Observed temperature ratio across the shock.
    :param gamma: Adiabatic index.
    :returns: Mach number M.
    :rtype: float or array-like
    """
    a = 2 * gamma * (gamma - 1)
    minusb = gamma * 2 - 6 * gamma + T2_T1 * (gamma + 1) ** 2 + 1
    return np.sqrt((minusb + np.sqrt(minusb**2 + 8 * a * (gamma - 1))) / (2 * a))


def P2P1M(P2_P1: Floats, gamma: Floats = 5 / 3) -> Floats:
    """Mach number inferred from pressure jump P₂/P₁.

    :param P2_P1: Observed pressure ratio across the shock.
    :param gamma: Adiabatic index.
    :returns: Mach number M.
    :rtype: float or array-like
    """
    return np.sqrt((P2_P1 * (gamma + 1) + gamma - 1) / (2 * gamma))


# ---------------------------------------------------------------------------- #
#                              Voronoi Graph                                   #
# ---------------------------------------------------------------------------- #


def build_voronoi(
    snap: "Snapshot | None" = None,
    X: str | ArrayLike = "X",
    Y: str | ArrayLike = "Y",
    Z: str | ArrayLike = "Z",
) -> SimpleNamespace:
    """Build a Voronoi adjacency graph from snapshot cell centres. Suitable for
    small data (<1,000,000 cells).

    Constructs the full Voronoi tessellation via
    :class:`scipy.spatial.Voronoi` and encodes the cell-neighbour relation as
    a Compressed Sparse Row (CSR) structure for efficient traversal in the
    numba kernels.

    :param snap: Loaded RICH snapshot providing ``X``, ``Y``, ``Z``
                 coordinates.  Optional - pass ``None`` and supply ``X``,
                 ``Y``, ``Z`` as array-likes directly.
    :param X: Field name resolved against ``snap``, or an array-like of
              x-coordinates when ``snap`` is ``None``.
    :param Y: Field name resolved against ``snap``, or an array-like of
              y-coordinates when ``snap`` is ``None``.
    :param Z: Field name resolved against ``snap``, or an array-like of
              z-coordinates when ``snap`` is ``None``.
    :returns: Namespace with attributes:

              * ``positions`` - ``float64 (N, 3)`` cell centres in code_length.
              * ``neighbor_data`` - ``int32`` flat adjacency list (CSR values).
              * ``neighbor_ptr`` - ``int32`` CSR row-pointer array, length N+1.
    :rtype: :class:`types.SimpleNamespace`
    """

    if snap is not None:
        X = snap._get_data(X).v
        Y = snap._get_data(Y).v
        Z = snap._get_data(Z).v
    else:
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

    positions = np.stack([X, Y, Z], axis=-1)

    print(f"Building Voronoi for {len(positions):,} cells ...", flush=True)
    vor = Voronoi(positions)

    adj = defaultdict(list)
    for i, j in vor.ridge_points:
        adj[i].append(j)
        adj[j].append(i)

    n = len(positions)
    sizes = np.array([len(adj[k]) for k in range(n)], dtype=np.int32)
    neighbor_ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(sizes, out=neighbor_ptr[1:])
    neighbor_data = np.array([j for k in range(n) for j in adj[k]], dtype=np.int32)

    print(f"Done.  Mean neighbours/cell: {sizes.mean():.1f}", flush=True)
    return SimpleNamespace(
        positions=positions,
        neighbor_data=neighbor_data,
        neighbor_ptr=neighbor_ptr,
    )


def build_knn(
    snap: "Snapshot | None" = None,
    X: str | ArrayLike = "X",
    Y: str | ArrayLike = "Y",
    Z: str | ArrayLike = "Z",
    k: int = 48,
    cells: ArrayLike | None = None,
    workers: int = -1,
    batch: int = 4_000_000,
) -> SimpleNamespace:
    """Build a *k*-nearest-neighbour graph as a drop-in for :func:`build_voronoi`.

    Returns the same ``SimpleNamespace(positions, neighbor_data, neighbor_ptr)``
    CSR structure, so every shockfinder kernel and public function consumes it
    unchanged.  Instead of the exact Voronoi adjacency, each cell's candidate
    list is its *k* nearest generators (via :class:`scipy.spatial.cKDTree`).

    This is valid because the analytical next-cell kernel (:func:`_next_cell`)
    already recovers the Voronoi face per query: a ray leaves cell *i* at the
    *smallest* positive bisector-crossing parameter, whose global minimum over
    all generators is always attained at a true Voronoi face neighbour.
    Minimising over the *k*-NN superset therefore returns the identical cell
    whenever *k* is large enough to contain that neighbour - extra candidates
    can never yield a nearer crossing.  Unlike the global tessellation this is
    cheap (≈ linear memory, KD-tree not grid) and robust to extreme clustering,
    so it scales to the ~80M-cell hi-res snapshots that ``build_voronoi`` cannot.

    The KD-tree is always built on the *full* point set (a neighbour may be any
    cell), but the per-cell neighbour list is only ever consumed for cells the
    shockfinder traces from - the conditions 1 & 2 candidates (see
    :func:`shock_candidates`).  Passing those via ``cells`` restricts the query
    to that subset (≈ a quarter of cells on TDE data), cutting query time and
    graph memory ~4x while producing an identical catalogue (the omitted rows
    are never indexed).

    :param snap: Loaded RICH snapshot, or ``None`` to pass coordinates directly.
    :param X: Field name resolved against ``snap``, or an array-like of
              x-coordinates when ``snap`` is ``None``.  (``Y``, ``Z`` likewise.)
    :param k: Number of nearest neighbours per cell.  Must exceed the local
              Voronoi face degree to match the exact traversal (mean ≈ 15;
              ``48`` matched exact Voronoi on >99.7% of test queries).
    :param cells: Optional global indices of the cells to build neighbour rows
                  for (e.g. :func:`shock_candidates` output).  ``None`` builds
                  rows for all N cells; otherwise only these rows are populated
                  and every other CSR row is empty.
    :param workers: Threads for the KD-tree query; ``-1`` uses all cores.
    :param batch: Query this many points at a time to bound peak memory.
    :returns: Namespace with ``positions`` ``float64 (N, 3)``, ``neighbor_data``
              ``int32`` flat adjacency (k per populated row), ``neighbor_ptr``
              ``int64`` CSR row-pointer of length N+1.
    :rtype: :class:`types.SimpleNamespace`
    """

    if snap is not None:
        X = snap._get_data(X).v
        Y = snap._get_data(Y).v
        Z = snap._get_data(Z).v
    else:
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

    positions = np.ascontiguousarray(np.stack([X, Y, Z], axis=-1), dtype=np.float64)
    n = len(positions)

    if cells is None:
        qidx = np.arange(n, dtype=np.int64)
    else:
        qidx = np.unique(np.asarray(cells, dtype=np.int64))  # sorted, deduplicated
        if qidx.size and (qidx[0] < 0 or qidx[-1] >= n):
            raise ValueError("`cells` contains out-of-range indices")
    m = len(qidx)

    print(f"Building k-NN graph (k={k}) for {m:,} / {n:,} cells ...", flush=True)
    tree = cKDTree(positions)

    # Neighbour rows for the queried cells, in sorted-qidx order.
    rows = np.empty(m * k, dtype=np.int32)
    for start in range(0, m, batch):
        stop = min(start + batch, m)
        qb = qidx[start:stop]
        _, ids = tree.query(positions[qb], k=k + 1, workers=workers)
        # Drop each row's self-match (distance 0). It is normally present
        # exactly once; if absent (≥k+1 coincident generators) drop the
        # farthest instead so every row keeps exactly k neighbours.
        is_self = ids == qb[:, None]
        no_self = ~is_self.any(axis=1)
        if no_self.any():
            is_self[no_self, -1] = True
        keep = ids[~is_self].reshape(stop - start, k)
        rows[start * k : stop * k] = keep.reshape(-1)

    neighbor_data = rows
    if cells is None:
        # Uniform stride k; int64 ptr avoids overflow for large N (n*k > 2^31).
        neighbor_ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    else:
        # Sparse CSR: k-wide window at each queried cell, empty rows elsewhere.
        # rows are in sorted-qidx order, which matches cumsum(sizes) offsets.
        sizes = np.zeros(n, dtype=np.int64)
        sizes[qidx] = k
        neighbor_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(sizes, out=neighbor_ptr[1:])

    print(f"Done.  {k} neighbours/cell.", flush=True)
    return SimpleNamespace(
        positions=positions,
        neighbor_data=neighbor_data,
        neighbor_ptr=neighbor_ptr,
    )


# ---------------------------------------------------------------------------- #
#                          Internal – numba JIT kernels                        #
# ---------------------------------------------------------------------------- #


@njit(cache=True)
def _next_cell(
    positions: NDArray[np.float64],
    neighbor_data: NDArray[np.integer],
    neighbor_ptr: NDArray[np.integer],
    idx: int,
    dx: float,
    dy: float,
    dz: float,
) -> int:
    """JIT kernel: find the next Voronoi cell along a ray (Springel 2010 §2).

    The perpendicular-bisector face between generators xᵢ and xⱼ has normal
    **n** = xⱼ - xᵢ.  A ray from xᵢ in direction **d̂** crosses it at
    t = |**n**|² / (2 **n**·**d̂**).  Returns the neighbour with the smallest
    positive *t*, or ``-1`` at a domain boundary.

    :param positions: Cell centre coordinates, shape ``(N, 3)``.
    :param neighbor_data: CSR flat adjacency array.
    :param neighbor_ptr: CSR row-pointer array, length N+1.
    :param idx: Index of the current cell.
    :param dx: x-component of the ray direction (need not be normalised).
    :param dy: y-component.
    :param dz: z-component.
    :returns: Global index of the neighbouring cell the ray enters, or ``-1``.
    :rtype: int
    """
    xi = positions[idx, 0]
    yi = positions[idx, 1]
    zi = positions[idx, 2]
    norm = _math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        return -1
    dx /= norm
    dy /= norm
    dz /= norm
    best_t, best_j = _math.inf, -1
    for k in range(neighbor_ptr[idx], neighbor_ptr[idx + 1]):
        j = neighbor_data[k]
        nx = positions[j, 0] - xi
        ny = positions[j, 1] - yi
        nz = positions[j, 2] - zi
        nd = nx * dx + ny * dy + nz * dz
        if nd <= 0.0:
            continue
        t = (nx * nx + ny * ny + nz * nz) / (2.0 * nd)
        if t < best_t:
            best_t = t
            best_j = j
    return best_j


@njit(parallel=True, cache=True)
def _condition3_kernel(
    positions: NDArray[np.float64],
    neighbor_data: NDArray[np.integer],
    neighbor_ptr: NDArray[np.integer],
    candidates: NDArray[np.int64],
    T_np: NDArray[np.float64],
    P_np: NDArray[np.float64],
    ds_np: NDArray[np.float64],
    DlogT_min: float,
    DlogP_min: float,
) -> NDArray[np.bool_]:
    """JIT kernel: evaluate Schaal+14 condition 3 for candidate shock cells.

    Traces a ray in ±**ds** from each candidate to find the pre- and
    post-shock cells, then checks whether the T and P jumps exceed the given
    thresholds on a log₁₀ scale.

    :param positions: Cell centres, shape ``(N, 3)``.
    :param neighbor_data: CSR flat adjacency array.
    :param neighbor_ptr: CSR row-pointer array, length N+1.
    :param candidates: Indices of cells satisfying conditions 1 & 2,
                       shape ``(M,)``.
    :param T_np: Proxy temperature array (P/rho), shape ``(N,)``.
    :param P_np: Pressure array, shape ``(N,)``.
    :param ds_np: Pre-normalised shock direction for each candidate,
                  shape ``(M, 3)``.
    :param DlogT_min: Minimum log₁₀(T₂/T₁) threshold (Schaal+14: 0.11).
    :param DlogP_min: Minimum log₁₀(P₂/P₁) threshold (Schaal+14: 0.27).
    :returns: Boolean array of length *M*; ``True`` for confirmed shock cells.
    :rtype: :class:`numpy.ndarray` (bool)
    """
    result = np.zeros(len(candidates), dtype=np.bool_)
    for k in prange(len(candidates)):  # each result[k] write is independent
        idx = candidates[k]
        dx = ds_np[k, 0]
        dy = ds_np[k, 1]
        dz = ds_np[k, 2]
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            continue
        i_pre = _next_cell(positions, neighbor_data, neighbor_ptr, idx, dx, dy, dz)
        i_post = _next_cell(positions, neighbor_data, neighbor_ptr, idx, -dx, -dy, -dz)
        if i_pre < 0 or i_post < 0:
            continue
        T_pre = T_np[i_pre]
        T_post = T_np[i_post]
        P_pre = P_np[i_pre]
        P_post = P_np[i_post]
        if T_pre <= 0.0 or T_post <= 0.0 or P_pre <= 0.0 or P_post <= 0.0:
            continue
        if _math.log10(T_post / T_pre) >= DlogT_min and _math.log10(P_post / P_pre) >= DlogP_min:
            result[k] = True
    return result


@njit(cache=True)
def _ray_tracer(
    positions: NDArray[np.float64],
    neighbor_data: NDArray[np.integer],
    neighbor_ptr: NDArray[np.integer],
    shock_mask: NDArray[np.bool_],
    idx_shock: NDArray[np.int64],
    idx_cell: int,
    divV_shock: NDArray[np.float64],
    ds_shock: NDArray[np.float64],
    sign: int,
    max_steps: int,
) -> int:
    """JIT kernel: walk along ±**ds** from a shock cell until leaving the zone.

    Steps cell-by-cell through the Voronoi graph in the ``sign * ds`` direction
    until a non-shock cell is reached (returned) or the walk is rejected
    (returns ``-1``).  Rejects are triggered by unphysical divergence increase
    or a reversed shock direction.

    :param positions: Cell centres, shape ``(N, 3)``.
    :param neighbor_data: CSR flat adjacency array.
    :param neighbor_ptr: CSR row-pointer array, length N+1.
    :param shock_mask: Boolean mask, shape ``(N,)``; ``True`` for shock cells.
    :param idx_shock: Sorted indices of shock cells, shape ``(S,)``.
    :param idx_cell: Local index within *idx_shock* for the starting cell.
    :param divV_shock: Velocity divergence for each shock cell, shape ``(S,)``.
    :param ds_shock: Shock direction for each shock cell, shape ``(S, 3)``.
    :param sign: ``+1`` to walk toward the pre-shock side; ``-1`` toward
                 post-shock.
    :param max_steps: Maximum number of Voronoi hops before giving up.
    :returns: Global cell index of the first non-shock cell found, or ``-1``.
    :rtype: int
    """
    current = idx_shock[idx_cell]
    dx = sign * ds_shock[idx_cell, 0]
    dy = sign * ds_shock[idx_cell, 1]
    dz = sign * ds_shock[idx_cell, 2]
    norm = _math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        return -1
    dx /= norm
    dy /= norm
    dz /= norm
    divV_curr = divV_shock[idx_cell]

    for _ in range(max_steps):
        nxt = _next_cell(positions, neighbor_data, neighbor_ptr, current, dx, dy, dz)
        if nxt < 0:
            return -1
        if not shock_mask[nxt]:
            return nxt  # exited → found pre/post cell

        pos = np.searchsorted(idx_shock, nxt)
        if pos < len(idx_shock) and idx_shock[pos] == nxt:
            divV_nxt = divV_shock[pos]
            if divV_nxt < divV_curr:
                return -1  # unphysical divergence
            if dx * ds_shock[pos, 0] + dy * ds_shock[pos, 1] + dz * ds_shock[pos, 2] < 0.0:
                return nxt  # shock direction reversed
            divV_curr = divV_nxt
        current = nxt
    return -1


@njit(parallel=True, cache=True)
def _shock_surface_kernel(
    positions: NDArray[np.float64],
    neighbor_data: NDArray[np.integer],
    neighbor_ptr: NDArray[np.integer],
    T_np: NDArray[np.float64],
    P_np: NDArray[np.float64],
    rho_np: NDArray[np.float64],
    shock_mask: NDArray[np.bool_],
    idx_shock: NDArray[np.int64],
    divV_shock: NDArray[np.float64],
    ds_shock: NDArray[np.float64],
    gamma: float,
    max_steps: int = 500,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """JIT kernel: full shock-surface finder (Schaal+14 §2.2).

    For each shock-zone cell traces pre- and post-shock rays via
    :func:`_ray_tracer`, checks that the temperature jump is ≥ 1, and
    computes three independent Mach-number estimates from the T, P, and rho
    Rankine-Hugoniot relations.

    Parallel strategy: pre-allocate full-size buffers so each ``prange``
    iteration writes exclusively to index *i* (no shared counter); a
    sequential compaction pass collects accepted entries afterwards.

    :param positions: Cell centres, shape ``(N, 3)``.
    :param neighbor_data: CSR flat adjacency array.
    :param neighbor_ptr: CSR row-pointer array, length N+1.
    :param T_np: Proxy temperature P/rho, shape ``(N,)``.
    :param P_np: Pressure, shape ``(N,)``.
    :param rho_np: Density, shape ``(N,)``.
    :param shock_mask: Boolean shock-zone mask, shape ``(N,)``.
    :param idx_shock: Sorted shock-zone cell indices, shape ``(S,)``.
    :param divV_shock: Velocity divergence for shock cells, shape ``(S,)``.
    :param ds_shock: Shock direction for shock cells, shape ``(S, 3)``.
    :param gamma: Adiabatic index.
    :param max_steps: Maximum ray-tracing hops. Defaults to ``500``.
    :returns: Tuple ``(M_T, M_P, M_rho, surf_idx, pre_idx, post_idx)`` - the
              three Mach arrays and the surface / pre- / post-shock cell
              indices, each of length *M* (accepted surface cells).
    :rtype: tuple
    """
    n = len(idx_shock)

    # Full-size buffers: thread i owns slot i exclusively → no race condition
    valid = np.zeros(n, np.bool_)
    M_T_buf = np.zeros(n, np.float64)
    M_P_buf = np.zeros(n, np.float64)
    M_rho_buf = np.zeros(n, np.float64)
    pre_buf = np.empty(n, np.int64)
    post_buf = np.empty(n, np.int64)

    for i in prange(n):
        if ds_shock[i, 0] == 0.0 and ds_shock[i, 1] == 0.0 and ds_shock[i, 2] == 0.0:
            continue

        post = _ray_tracer(
            positions,
            neighbor_data,
            neighbor_ptr,
            shock_mask,
            idx_shock,
            i,
            divV_shock,
            ds_shock,
            -1,
            max_steps,
        )
        if post < 0:
            continue
        pre = _ray_tracer(
            positions,
            neighbor_data,
            neighbor_ptr,
            shock_mask,
            idx_shock,
            i,
            divV_shock,
            ds_shock,
            +1,
            max_steps,
        )
        if pre < 0:
            continue

        T_jump = T_np[post] / T_np[pre]
        if T_jump < 1.0 or T_np[pre] <= 0.0:
            continue

        P_jump = P_np[post] / P_np[pre]
        rho_jump = rho_np[post] / rho_np[pre]

        a = 2.0 * gamma * (gamma - 1.0)
        mb = 2.0 * gamma - 6.0 * gamma + T_jump * (gamma + 1.0) ** 2 + 1.0
        M_T_buf[i] = _math.sqrt(
            max((mb + _math.sqrt(max(mb * mb + 8.0 * a * (gamma - 1.0), 0.0))) / (2.0 * a), 0.0)
        )
        M_P_buf[i] = _math.sqrt(max((P_jump * (gamma + 1.0) + gamma - 1.0) / (2.0 * gamma), 0.0))
        dR = gamma + 1.0 - rho_jump * (gamma - 1.0)
        M_rho_buf[i] = _math.sqrt(2.0 * rho_jump / dR) if dR > 0.0 else 0.0
        pre_buf[i] = pre
        post_buf[i] = post
        valid[i] = True

    # Sequential compaction - valid[i] is already committed before this loop
    count = 0
    M_T = np.empty(n, np.float64)
    M_P = np.empty(n, np.float64)
    M_rho = np.empty(n, np.float64)
    i_surf = np.empty(n, np.int64)
    i_pre = np.empty(n, np.int64)
    i_post = np.empty(n, np.int64)
    for i in range(n):
        if valid[i]:
            M_T[count] = M_T_buf[i]
            M_P[count] = M_P_buf[i]
            M_rho[count] = M_rho_buf[i]
            i_surf[count] = i
            i_pre[count] = pre_buf[i]
            i_post[count] = post_buf[i]
            count += 1

    return (M_T[:count], M_P[:count], M_rho[:count], i_surf[:count], i_pre[:count], i_post[:count])


# ---------------------------------------------------------------------------- #
#                    Sophisticated – thread-pool chunk kernels                 #
# ---------------------------------------------------------------------------- #
# Numba releases the GIL inside @njit functions, so Python's ThreadPoolExecutor
# achieves true CPU parallelism without data copying or pickling overhead.
# The chunk kernels below are serial slices of the main kernels; the thread
# pool dispatches them across OS threads while they share the same numpy arrays.


@njit(cache=True)
def _condition3_kernel_chunk(
    positions: NDArray[np.float64],
    neighbor_data: NDArray[np.integer],
    neighbor_ptr: NDArray[np.integer],
    candidates: NDArray[np.int64],
    T_np: NDArray[np.float64],
    P_np: NDArray[np.float64],
    ds_np: NDArray[np.float64],
    DlogT_min: float,
    DlogP_min: float,
    k_start: int,
    k_end: int,
) -> NDArray[np.bool_]:
    """Serial condition-3 check for ``candidates[k_start:k_end]``.

    :param k_start: First candidate index (inclusive).
    :param k_end: Last candidate index (exclusive).
    :returns: Boolean array of length ``k_end - k_start``.
    :rtype: :class:`numpy.ndarray` (bool)
    """
    chunk = k_end - k_start
    result = np.zeros(chunk, dtype=np.bool_)
    for kk in range(chunk):
        k = k_start + kk
        idx = candidates[k]
        dx = ds_np[k, 0]
        dy = ds_np[k, 1]
        dz = ds_np[k, 2]
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            continue
        i_pre = _next_cell(positions, neighbor_data, neighbor_ptr, idx, dx, dy, dz)
        i_post = _next_cell(positions, neighbor_data, neighbor_ptr, idx, -dx, -dy, -dz)
        if i_pre < 0 or i_post < 0:
            continue
        T_pre = T_np[i_pre]
        T_post = T_np[i_post]
        P_pre = P_np[i_pre]
        P_post = P_np[i_post]
        if T_pre <= 0.0 or T_post <= 0.0 or P_pre <= 0.0 or P_post <= 0.0:
            continue
        if _math.log10(T_post / T_pre) >= DlogT_min and _math.log10(P_post / P_pre) >= DlogP_min:
            result[kk] = True
    return result


@njit(cache=True)
def _shock_surface_kernel_chunk(
    positions: NDArray[np.float64],
    neighbor_data: NDArray[np.integer],
    neighbor_ptr: NDArray[np.integer],
    T_np: NDArray[np.float64],
    P_np: NDArray[np.float64],
    rho_np: NDArray[np.float64],
    shock_mask: NDArray[np.bool_],
    idx_shock: NDArray[np.int64],
    divV_shock: NDArray[np.float64],
    ds_shock: NDArray[np.float64],
    gamma: float,
    i_start: int,
    i_end: int,
    max_steps: int = 500,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Serial shock-surface kernel for ``idx_shock[i_start:i_end]``.

    Reads the full shared arrays (``idx_shock``, ``divV_shock``, ``ds_shock``)
    because :func:`_ray_tracer` may traverse shock cells outside the chunk
    boundary.

    :param i_start: First shock-cell index in *idx_shock* (inclusive).
    :param i_end: Last shock-cell index in *idx_shock* (exclusive).
    :returns: Tuple ``(M_T, M_P, M_rho, surf_idx, pre_idx, post_idx)`` for
              accepted cells in the chunk.
    :rtype: tuple
    """
    chunk = i_end - i_start
    M_T = np.empty(chunk, np.float64)
    M_P = np.empty(chunk, np.float64)
    M_rho = np.empty(chunk, np.float64)
    i_surf = np.empty(chunk, np.int64)
    i_pre = np.empty(chunk, np.int64)
    i_post = np.empty(chunk, np.int64)
    count = 0
    for ii in range(chunk):
        i = i_start + ii
        if ds_shock[i, 0] == 0.0 and ds_shock[i, 1] == 0.0 and ds_shock[i, 2] == 0.0:
            continue
        post = _ray_tracer(
            positions,
            neighbor_data,
            neighbor_ptr,
            shock_mask,
            idx_shock,
            i,
            divV_shock,
            ds_shock,
            -1,
            max_steps,
        )
        if post < 0:
            continue
        pre = _ray_tracer(
            positions,
            neighbor_data,
            neighbor_ptr,
            shock_mask,
            idx_shock,
            i,
            divV_shock,
            ds_shock,
            +1,
            max_steps,
        )
        if pre < 0:
            continue
        T_jump = T_np[post] / T_np[pre]
        if T_jump < 1.0 or T_np[pre] <= 0.0:
            continue
        P_jump = P_np[post] / P_np[pre]
        rho_jump = rho_np[post] / rho_np[pre]
        a = 2.0 * gamma * (gamma - 1.0)
        mb = 2.0 * gamma - 6.0 * gamma + T_jump * (gamma + 1.0) ** 2 + 1.0
        M_T[count] = _math.sqrt(
            max((mb + _math.sqrt(max(mb * mb + 8.0 * a * (gamma - 1.0), 0.0))) / (2.0 * a), 0.0)
        )
        M_P[count] = _math.sqrt(max((P_jump * (gamma + 1.0) + gamma - 1.0) / (2.0 * gamma), 0.0))
        dR = gamma + 1.0 - rho_jump * (gamma - 1.0)
        M_rho[count] = _math.sqrt(2.0 * rho_jump / dR) if dR > 0.0 else 0.0
        i_surf[count] = i  # position within idx_shock
        i_pre[count] = pre
        i_post[count] = post
        count += 1
    return (M_T[:count], M_P[:count], M_rho[:count], i_surf[:count], i_pre[:count], i_post[:count])


# ---------------------------------------------------------------------------- #
#                                  Shock Zone                                  #
# ---------------------------------------------------------------------------- #


def shock_candidates(snap: "Snapshot", gamma: float = 5 / 3) -> SimpleNamespace:
    """Compute shock-zone *candidate* cells (Schaal+14 conditions 1 & 2).

    Conditions 1 (∇·v < 0) and 2 (∇T·∇P > 0) are cheap, purely local cuts.
    Condition 3 (the face T/P jump) and the surface ray tracing only ever call
    :func:`_next_cell` on cells passing 1 & 2 - ``_condition3_kernel`` on these
    candidates, ``_ray_tracer`` only on the shock zone (a subset).  So the
    neighbour graph is only needed for these candidates: passing this set to
    :func:`build_knn` (its ``cells`` argument) restricts the KD-tree query to
    the ~quarter of cells that matter instead of all N.

    :param snap: Loaded RICH snapshot providing ``divV``, ``P``, ``rho`` and the
                 pressure/density gradient fields.
    :param gamma: Adiabatic index.  Accepted for signature parity with the
                  ``find_shock_*`` functions; not used by conditions 1 & 2.
    :returns: Namespace with attributes:

              * ``candidates`` - ``int64 (M,)`` global indices passing 1 & 2.
              * ``ds`` - ``float64 (N, 3)`` shock direction −∇T/|∇T| (zero where
                |∇T| = 0), reused by :func:`find_shock_zone` to avoid a second
                gradient pass.
    :rtype: :class:`types.SimpleNamespace`
    """
    # Condition 1 - converging flow
    cond1 = snap.divV.v < 0

    # Temperature gradient  ∇T ∝ ∇P/rho - P ∇ρ/ρ²
    P_v = snap.P.v
    rho_v = snap.rho.v
    grad_P = np.stack([snap.DpDx.v, snap.DpDy.v, snap.DpDz.v], axis=-1)
    grad_rho = np.stack([snap.DrhoDx.v, snap.DrhoDy.v, snap.DrhoDz.v], axis=-1)
    grad_T = (grad_P.T / rho_v - P_v * grad_rho.T / rho_v**2).T

    # Condition 2 - temperature gradient aligned with pressure gradient
    cond2 = np.einsum("ij,ij->i", grad_T, grad_P) > 0

    # Shock direction: −∇T / |∇T|  (points from post-shock → pre-shock)
    norm_gT = np.linalg.norm(grad_T, axis=-1, keepdims=True)
    ds = np.where(norm_gT > 0, -grad_T / norm_gT, 0.0)

    candidates = np.ascontiguousarray(np.where(cond1 & cond2)[0], dtype=np.int64)
    return SimpleNamespace(candidates=candidates, ds=ds)


def find_shock_zone(
    snap: "Snapshot",
    vor: SimpleNamespace,
    gamma: float = 5 / 3,
    mach_min: float = 1.3,
) -> NDArray[np.bool_]:
    """Identify shock-zone cells via the three Schaal+14 conditions.

    * **Condition 1** - converging flow: ∇·v < 0.
    * **Condition 2** - temperature gradient aligned with pressure gradient:
      ∇T · ∇P > 0.
    * **Condition 3** - measurable T and P jump across the Voronoi face
      (evaluated by :func:`_condition3_kernel`; thresholds log₁₀ΔT ≥ 0.11,
      log₁₀ΔP ≥ 0.27).

    :param snap: Loaded RICH snapshot.  Must provide ``DrhoDx``, ``DrhoDy``,
                 ``DrhoDz`` for the temperature-gradient calculation.
    :param vor: Voronoi graph from :func:`build_voronoi`.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Boolean array of shape ``(N,)``; ``True`` for shock-zone cells.
    :rtype: :class:`numpy.ndarray` (bool)
    """
    # Conditions 1 & 2 (and the shock direction ds) - shared helper
    cand = shock_candidates(snap, gamma)
    candidates = cand.candidates

    # Condition 3 - Voronoi/k-NN + numba
    # T ∝ P/rho: we use the ratio directly (constant μmH/kB cancels)
    P_v = snap.P.v
    rho_v = snap.rho.v
    T_np = np.ascontiguousarray(P_v / rho_v, dtype=np.float64)
    P_np = np.ascontiguousarray(P_v, dtype=np.float64)

    ds_cand = np.ascontiguousarray(cand.ds[candidates], dtype=np.float64)

    DlogT_min = np.log10(MT2T1(mach_min))  # = 0.11 when M = 1.3
    DlogP_min = np.log10(MP2P1(mach_min))  # = 0.27 when M = 1.3
    cond3_mask = _condition3_kernel(
        vor.positions,
        vor.neighbor_data,
        vor.neighbor_ptr,
        candidates,
        T_np,
        P_np,
        ds_cand,
        DlogT_min=DlogT_min,
        DlogP_min=DlogP_min,
    )

    shock_zone = np.zeros(len(snap), dtype=bool)
    shock_zone[candidates[cond3_mask]] = True
    return shock_zone


# ---------------------------------------------------------------------------- #
#                                 Shock Surface                                #
# ---------------------------------------------------------------------------- #


def find_shock_surface(
    snap: "Snapshot",
    vor: SimpleNamespace,
    shock_zone: NDArray[np.bool_],
    gamma: float = 5 / 3,
) -> SimpleNamespace:
    """Locate shock-surface cells and estimate Mach numbers (Schaal+14 §2.2).

    For each cell in the shock zone, traces rays in ±**ds** until exiting the
    zone, then computes three independent Rankine-Hugoniot Mach-number estimates
    from the T, P, and rho jumps between the pre- and post-shock cells.

    :param snap: Loaded RICH snapshot providing ``P``, ``rho``, ``divV``, and
                 gradient fields.
    :param vor: Voronoi graph from :func:`build_voronoi`.
    :param shock_zone: Boolean shock-zone mask from :func:`find_shock_zone`,
                       shape ``(N,)``.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :returns: Namespace with attributes:

              * ``surface_mask`` - ``bool (N,)`` global mask for shock-surface cells.
              * ``pre_mask``     - ``bool (N,)`` mask for pre-shock cells.
              * ``post_mask``    - ``bool (N,)`` mask for post-shock cells.
              * ``mach_T``       - ``float (M,)`` Mach number from temperature jump.
              * ``mach_P``       - ``float (M,)`` Mach number from pressure jump.
              * ``mach_rho``     - ``float (M,)`` Mach number from density jump.
    :rtype: :class:`types.SimpleNamespace`
    """
    P_v = snap.P.v
    rho_v = snap.rho.v
    grad_P = np.stack([snap.DpDx.v, snap.DpDy.v, snap.DpDz.v], axis=-1)
    grad_rho = np.stack([snap.DrhoDx.v, snap.DrhoDy.v, snap.DrhoDz.v], axis=-1)
    grad_T = (grad_P.T / rho_v - P_v * grad_rho.T / rho_v**2).T

    norm_gT = np.linalg.norm(grad_T, axis=-1, keepdims=True)
    ds = np.where(norm_gT > 0, -grad_T / norm_gT, 0.0)

    T_np = np.ascontiguousarray(P_v / rho_v, dtype=np.float64)
    P_np = np.ascontiguousarray(P_v, dtype=np.float64)
    rho_np = np.ascontiguousarray(rho_v, dtype=np.float64)

    shock_mask = np.asarray(shock_zone, dtype=np.bool_)
    idx_shock = np.ascontiguousarray(np.where(shock_zone)[0], dtype=np.int64)

    divV_shock = np.ascontiguousarray(snap.divV.v[idx_shock], dtype=np.float64)
    ds_shock = np.ascontiguousarray(ds[idx_shock], dtype=np.float64)

    M_T, M_P, M_rho, i_surf, i_pre, i_post = _shock_surface_kernel(
        vor.positions,
        vor.neighbor_data,
        vor.neighbor_ptr,
        T_np,
        P_np,
        rho_np,
        shock_mask,
        idx_shock,
        divV_shock,
        ds_shock,
        gamma,
    )

    surface_mask = np.zeros(len(snap), dtype=bool)
    pre_mask = np.zeros(len(snap), dtype=bool)
    post_mask = np.zeros(len(snap), dtype=bool)

    surface_mask[idx_shock[i_surf]] = True
    pre_mask[i_pre] = True
    post_mask[i_post] = True

    return SimpleNamespace(
        surface_mask=surface_mask,
        pre_mask=pre_mask,
        post_mask=post_mask,
        mach_T=M_T,
        mach_P=M_P,
        mach_rho=M_rho,
    )


# ---------------------------------------------------------------------------- #
#                     Sophisticated – thread-pool public API                   #
# ---------------------------------------------------------------------------- #


def find_shock_zone_threaded(
    snap: "Snapshot",
    vor: SimpleNamespace,
    gamma: float = 5 / 3,
    n_workers: int | None = None,
    mach_min: float = 1.3,
) -> NDArray[np.bool_]:
    """Thread-pool parallel version of :func:`find_shock_zone`.

    Dispatches condition-3 evaluation across ``n_workers`` OS threads using
    :class:`~concurrent.futures.ThreadPoolExecutor`.  Because numba releases
    the GIL inside ``@njit`` functions the threads run compiled code in true
    parallel on separate CPU cores without any data copying.

    Compared to the ``prange`` version this gives explicit control over chunk
    size and allows interleaving with other Python work between chunks.

    :param snap: Loaded RICH snapshot.
    :param vor: Voronoi graph from :func:`build_voronoi`.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :param n_workers: Number of threads.  Defaults to ``os.cpu_count()``.
    :returns: Boolean shock-zone mask, shape ``(N,)``.
    :rtype: :class:`numpy.ndarray` (bool)
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    cand = shock_candidates(snap, gamma)
    candidates = cand.candidates

    P_v = snap.P.v
    rho_v = snap.rho.v
    T_np = np.ascontiguousarray(P_v / rho_v, dtype=np.float64)
    P_np = np.ascontiguousarray(P_v, dtype=np.float64)
    ds_cand = np.ascontiguousarray(cand.ds[candidates], dtype=np.float64)

    M = len(candidates)
    chunk_sz = max(1, (M + n_workers - 1) // n_workers)
    chunks = [(k, min(k + chunk_sz, M)) for k in range(0, M, chunk_sz)]

    DlogT_min = np.log10(MT2T1(mach_min))  # = 0.11 when M = 1.3
    DlogP_min = np.log10(MP2P1(mach_min))  # = 0.27 when M = 1.3

    def _run(k_start_end):
        k0, k1 = k_start_end
        return _condition3_kernel_chunk(
            vor.positions,
            vor.neighbor_data,
            vor.neighbor_ptr,
            candidates,
            T_np,
            P_np,
            ds_cand,
            DlogT_min,
            DlogP_min,
            k0,
            k1,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_run, chunks))

    cond3_mask = np.concatenate(parts)
    shock_zone = np.zeros(len(snap), dtype=bool)
    shock_zone[candidates[cond3_mask]] = True
    return shock_zone


def find_shock_surface_threaded(
    snap: "Snapshot",
    vor: SimpleNamespace,
    shock_zone: NDArray[np.bool_],
    gamma: float = 5 / 3,
    n_workers: int | None = None,
) -> SimpleNamespace:
    """Thread-pool parallel version of :func:`find_shock_surface`.

    Splits the shock-zone cell list into ``n_workers`` contiguous chunks and
    dispatches each to a thread running :func:`_shock_surface_kernel_chunk`.
    Numba's GIL release means the ray-tracing runs in true parallel; all
    threads share the same numpy arrays with zero copying.

    :param snap: Loaded RICH snapshot.
    :param vor: Voronoi graph from :func:`build_voronoi`.
    :param shock_zone: Boolean shock-zone mask from :func:`find_shock_zone`.
    :param gamma: Adiabatic index.  Defaults to 5/3.
    :param n_workers: Number of threads.  Defaults to ``os.cpu_count()``.
    :returns: Same namespace as :func:`find_shock_surface`.
    :rtype: :class:`types.SimpleNamespace`
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    P_v = snap.P.v
    rho_v = snap.rho.v
    grad_P = np.stack([snap.DpDx.v, snap.DpDy.v, snap.DpDz.v], axis=-1)
    grad_rho = np.stack([snap.DrhoDx.v, snap.DrhoDy.v, snap.DrhoDz.v], axis=-1)
    grad_T = (grad_P.T / rho_v - P_v * grad_rho.T / rho_v**2).T
    norm_gT = np.linalg.norm(grad_T, axis=-1, keepdims=True)
    ds = np.where(norm_gT > 0, -grad_T / norm_gT, 0.0)

    T_np = np.ascontiguousarray(P_v / rho_v, dtype=np.float64)
    P_np = np.ascontiguousarray(P_v, dtype=np.float64)
    rho_np = np.ascontiguousarray(rho_v, dtype=np.float64)
    shock_mask = np.asarray(shock_zone, dtype=np.bool_)
    idx_shock = np.ascontiguousarray(np.where(shock_zone)[0], dtype=np.int64)
    divV_shock = np.ascontiguousarray(snap.divV.v[idx_shock], dtype=np.float64)
    ds_shock = np.ascontiguousarray(ds[idx_shock], dtype=np.float64)

    S = len(idx_shock)
    chunk_sz = max(1, (S + n_workers - 1) // n_workers)
    chunks = [(i, min(i + chunk_sz, S)) for i in range(0, S, chunk_sz)]

    def _run(i_start_end):
        i0, i1 = i_start_end
        return _shock_surface_kernel_chunk(
            vor.positions,
            vor.neighbor_data,
            vor.neighbor_ptr,
            T_np,
            P_np,
            rho_np,
            shock_mask,
            idx_shock,
            divV_shock,
            ds_shock,
            gamma,
            i0,
            i1,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_run, chunks))

    M_T_all = np.concatenate([r[0] for r in parts])
    M_P_all = np.concatenate([r[1] for r in parts])
    M_rho_all = np.concatenate([r[2] for r in parts])
    i_surf_all = np.concatenate([r[3] for r in parts])
    i_pre_all = np.concatenate([r[4] for r in parts])
    i_post_all = np.concatenate([r[5] for r in parts])

    surface_mask = np.zeros(len(snap), dtype=bool)
    pre_mask = np.zeros(len(snap), dtype=bool)
    post_mask = np.zeros(len(snap), dtype=bool)
    surface_mask[idx_shock[i_surf_all]] = True
    pre_mask[i_pre_all] = True
    post_mask[i_post_all] = True

    return SimpleNamespace(
        surface_mask=surface_mask,
        pre_mask=pre_mask,
        post_mask=post_mask,
        mach_T=M_T_all,
        mach_P=M_P_all,
        mach_rho=M_rho_all,
    )
