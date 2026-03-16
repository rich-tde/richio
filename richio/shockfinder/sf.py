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

import math as _math
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from numba import njit
from scipy.spatial import Voronoi


# ---------------------------------------------------------------------------- #
#                             Analytical Functions                             #
# ---------------------------------------------------------------------------- #

def delta(M, gamma=5/3):
    R = 1/((gamma - 1)/(gamma + 1) + 2/(gamma + 1)/M**2) # R = rho2/rho1
    delta = 2/(gamma*(gamma - 1) * M**2 * R) * ((2*gamma*M**2 - (gamma - 1))/(gamma + 1) - R**gamma)
    return delta

def R2M(R, gamma=5/3):
    """Mach number from compression ratio rho2/rho1 (Rankine-Hugoniot)."""
    return 1/np.sqrt((gamma + 1)/(2*R) - (gamma - 1)/2)
rho2rho1M = R2M  # alias

def M2R(M, gamma=5/3):
    """Compression ratio rho2/rho1 from Mach number (Rankine-Hugoniot)."""
    return (gamma + 1)*M**2 / ((gamma - 1)*M**2 + 2)
Mrho2rho1 = M2R  # alias

def MT2T1(M, gamma=5/3):
    """Temperature jump T2/T1 from Mach number."""
    return (2*gamma*M**2 - (gamma - 1)) * ((gamma - 1)*M**2 + 2) / ((gamma + 1)**2 * M**2)

def MP2P1(M, gamma=5/3):
    """Pressure jump P2/P1 from Mach number."""
    return (2*gamma*M**2)/(gamma + 1) - (gamma - 1)/(gamma + 1)

def T2T1M(T2_T1, gamma):
    """Mach number from temperature jump T2/T1."""
    a = 2 * gamma * (gamma - 1)
    minusb = gamma * 2 - 6 * gamma + T2_T1 * (gamma + 1)**2 + 1
    return np.sqrt((minusb + np.sqrt(minusb**2 + 8 * a * (gamma - 1))) / (2 * a))

def P2P1M(P2_P1, gamma):
    """Mach number from pressure jump P2/P1."""
    return np.sqrt((P2_P1 * (gamma + 1) + gamma - 1) / (2 * gamma))


# ---------------------------------------------------------------------------- #
#                              Voronoi Graph                                   #
# ---------------------------------------------------------------------------- #

def build_voronoi(snap):
    """Build Voronoi adjacency graph from snap cell centres.

    Parameters
    ----------
    snap : richio Snapshot

    Returns
    -------
    SimpleNamespace with:
        positions     : float64 (N, 3)  cell centres in code_length
        neighbor_data : int32            flat adjacency list (CSR values)
        neighbor_ptr  : int32            CSR row pointers, length N+1
    """
    positions = np.stack([snap.X.v, snap.Y.v, snap.Z.v], axis=-1)

    print(f"Building Voronoi for {len(positions):,} cells ...", flush=True)
    vor = Voronoi(positions)

    adj = defaultdict(list)
    for i, j in vor.ridge_points:
        adj[i].append(j)
        adj[j].append(i)

    n     = len(positions)
    sizes = np.array([len(adj[k]) for k in range(n)], dtype=np.int32)
    neighbor_ptr  = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(sizes, out=neighbor_ptr[1:])
    neighbor_data = np.array([j for k in range(n) for j in adj[k]], dtype=np.int32)

    print(f"Done.  Mean neighbours/cell: {sizes.mean():.1f}", flush=True)
    return SimpleNamespace(
        positions=positions,
        neighbor_data=neighbor_data,
        neighbor_ptr=neighbor_ptr,
    )


# ---------------------------------------------------------------------------- #
#                          Internal – numba JIT kernels                        #
# ---------------------------------------------------------------------------- #

@njit(cache=True)
def _next_cell(positions, neighbor_data, neighbor_ptr, idx, dx, dy, dz):
    """Exact ray–face crossing on the Voronoi graph.

    The face between generators xᵢ and xⱼ is the perpendicular bisector plane
    with normal n = xⱼ−xᵢ.  A ray from xᵢ in direction d̂ hits it at
        t = |n|² / (2 · n·d̂)
    We return the neighbour with the smallest positive t, or -1 at a boundary.

    Source: Springel (2010) MNRAS 401, §2.
    """
    xi = positions[idx, 0];  yi = positions[idx, 1];  zi = positions[idx, 2]
    norm = _math.sqrt(dx*dx + dy*dy + dz*dz)
    if norm == 0.0:
        return -1
    dx /= norm;  dy /= norm;  dz /= norm
    best_t, best_j = _math.inf, -1
    for k in range(neighbor_ptr[idx], neighbor_ptr[idx + 1]):
        j  = neighbor_data[k]
        nx = positions[j, 0] - xi
        ny = positions[j, 1] - yi
        nz = positions[j, 2] - zi
        nd = nx*dx + ny*dy + nz*dz
        if nd <= 0.0:
            continue
        t = (nx*nx + ny*ny + nz*nz) / (2.0 * nd)
        if t < best_t:
            best_t = t;  best_j = j
    return best_j


@njit(cache=True)
def _condition3_kernel(positions, neighbor_data, neighbor_ptr,
                       candidates, T_np, P_np, ds_np, DlogT_min, DlogP_min):
    """Check Schaal+14 condition 3 for every candidate cell (numba kernel).

    ds_np : (N, 3) pre-normalised shock directions for each candidate.
    Returns bool array of length N.
    """
    result = np.zeros(len(candidates), dtype=np.bool_)
    for k in range(len(candidates)):
        idx = candidates[k]
        dx = ds_np[k, 0];  dy = ds_np[k, 1];  dz = ds_np[k, 2]
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            continue
        i_pre  = _next_cell(positions, neighbor_data, neighbor_ptr, idx,  dx,  dy,  dz)
        i_post = _next_cell(positions, neighbor_data, neighbor_ptr, idx, -dx, -dy, -dz)
        if i_pre < 0 or i_post < 0:
            continue
        T_pre = T_np[i_pre];  T_post = T_np[i_post]
        P_pre = P_np[i_pre];  P_post = P_np[i_post]
        if T_pre <= 0.0 or T_post <= 0.0 or P_pre <= 0.0 or P_post <= 0.0:
            continue
        if (_math.log10(T_post/T_pre) >= DlogT_min and
                _math.log10(P_post/P_pre) >= DlogP_min):
            result[k] = True
    return result


@njit(cache=True)
def _ray_tracer(positions, neighbor_data, neighbor_ptr, shock_mask,
                idx_shock, idx_cell, divV_shock, ds_shock, sign, max_steps):
    """Walk along ±ds until exiting the shock zone.

    sign = +1 → pre-shock (along ds),  sign = -1 → post-shock.
    Returns global cell index or -1 (rejected).
    """
    current = idx_shock[idx_cell]
    dx = sign * ds_shock[idx_cell, 0]
    dy = sign * ds_shock[idx_cell, 1]
    dz = sign * ds_shock[idx_cell, 2]
    norm = _math.sqrt(dx*dx + dy*dy + dz*dz)
    if norm == 0.0:
        return -1
    dx /= norm;  dy /= norm;  dz /= norm
    divV_curr = divV_shock[idx_cell]

    for _ in range(max_steps):
        nxt = _next_cell(positions, neighbor_data, neighbor_ptr, current, dx, dy, dz)
        if nxt < 0:
            return -1
        if not shock_mask[nxt]:
            return nxt                                # exited → found pre/post cell

        pos = np.searchsorted(idx_shock, nxt)
        if pos < len(idx_shock) and idx_shock[pos] == nxt:
            divV_nxt = divV_shock[pos]
            if divV_nxt < divV_curr:
                return -1                             # unphysical divergence
            if dx*ds_shock[pos,0] + dy*ds_shock[pos,1] + dz*ds_shock[pos,2] < 0.0:
                return nxt                            # shock direction reversed
            divV_curr = divV_nxt
        current = nxt
    return -1


@njit(cache=True)
def _shock_surface_kernel(positions, neighbor_data, neighbor_ptr,
                          T_np, P_np, rho_np,
                          shock_mask, idx_shock, divV_shock, ds_shock,
                          gamma, max_steps=500):
    """Full shock-surface finder — fully JIT-compiled (Schaal+14 §2.2).

    Returns (M_T, M_P, M_rho, surf_idx, pre_idx, post_idx).
    """
    n = len(idx_shock)
    M_T    = np.empty(n, np.float64);  M_P   = np.empty(n, np.float64)
    M_rho  = np.empty(n, np.float64)
    i_surf = np.empty(n, np.int64);    i_pre = np.empty(n, np.int64)
    i_post = np.empty(n, np.int64)
    count  = 0

    for i in range(n):
        if ds_shock[i,0] == 0.0 and ds_shock[i,1] == 0.0 and ds_shock[i,2] == 0.0:
            continue

        post = _ray_tracer(positions, neighbor_data, neighbor_ptr, shock_mask,
                           idx_shock, i, divV_shock, ds_shock, -1, max_steps)
        if post < 0:
            continue
        pre = _ray_tracer(positions, neighbor_data, neighbor_ptr, shock_mask,
                          idx_shock, i, divV_shock, ds_shock, +1, max_steps)
        if pre < 0:
            continue

        T_jump = T_np[post] / T_np[pre]
        if T_jump < 1.0 or T_np[pre] <= 0.0:
            continue

        P_jump   = P_np[post]   / P_np[pre]
        rho_jump = rho_np[post] / rho_np[pre]

        # T2T1M
        a  = 2.0*gamma*(gamma-1.0)
        mb = 2.0*gamma - 6.0*gamma + T_jump*(gamma+1.0)**2 + 1.0
        M_T[count] = _math.sqrt(max((mb + _math.sqrt(max(mb*mb + 8.0*a*(gamma-1.0), 0.0))) / (2.0*a), 0.0))

        # P2P1M
        M_P[count] = _math.sqrt(max((P_jump*(gamma+1.0) + gamma-1.0) / (2.0*gamma), 0.0))

        # rho2rho1M
        dR = gamma + 1.0 - rho_jump*(gamma-1.0)
        M_rho[count] = _math.sqrt(2.0*rho_jump/dR) if dR > 0.0 else 0.0

        i_surf[count] = i;  i_pre[count] = pre;  i_post[count] = post
        count += 1

    return (M_T[:count], M_P[:count], M_rho[:count],
            i_surf[:count], i_pre[:count], i_post[:count])


# ---------------------------------------------------------------------------- #
#                                  Shock Zone                                  #
# ---------------------------------------------------------------------------- #

def find_shock_zone(snap, vor, gamma=5/3):
    """Identify shock-zone cells via three Schaal+14 criteria.

    Condition 1: converging flow (∇·v < 0)
    Condition 2: temperature gradient aligned with pressure gradient
    Condition 3: measurable T and P jump across the face

    Parameters
    ----------
    snap  : richio Snapshot (must provide DrhoDx/y/z for grad_T)
    vor   : result of build_voronoi(snap)
    gamma : adiabatic index

    Returns
    -------
    shock_zone : bool ndarray, shape (N,)
    """
    # Condition 1
    cond1 = snap.divV.v < 0

    # Temperature gradient  ∇T ∝ ∇P/ρ − P ∇ρ/ρ²
    P_v   = snap.P.v
    rho_v = snap.rho.v
    grad_P = np.stack([snap.DpDx.v, snap.DpDy.v, snap.DpDz.v], axis=-1)
    grad_rho = np.stack([snap.DrhoDx.v, snap.DrhoDy.v, snap.DrhoDz.v], axis=-1)
    grad_T = (grad_P.T / rho_v - P_v * grad_rho.T / rho_v**2).T

    # Condition 2
    cond2 = np.einsum('ij,ij->i', grad_T, grad_P) > 0

    # Shock direction: −∇T / |∇T|  (points from post-shock → pre-shock)
    norm_gT = np.linalg.norm(grad_T, axis=-1, keepdims=True)
    ds = np.where(norm_gT > 0, -grad_T / norm_gT, 0.0)

    # Condition 3 — Voronoi + numba
    # T ∝ P/ρ: we use the ratio directly (constant μmH/kB cancels)
    T_np   = np.ascontiguousarray(P_v / rho_v,  dtype=np.float64)
    P_np   = np.ascontiguousarray(P_v,           dtype=np.float64)

    candidates = np.ascontiguousarray(np.where(cond1 & cond2)[0], dtype=np.int64)
    ds_cand    = np.ascontiguousarray(ds[candidates], dtype=np.float64)

    cond3_mask = _condition3_kernel(
        vor.positions, vor.neighbor_data, vor.neighbor_ptr,
        candidates, T_np, P_np, ds_cand, 0.11, 0.27)

    shock_zone = np.zeros(len(snap), dtype=bool)
    shock_zone[candidates[cond3_mask]] = True
    return shock_zone


# ---------------------------------------------------------------------------- #
#                                 Shock Surface                                #
# ---------------------------------------------------------------------------- #

def find_shock_surface(snap, vor, shock_zone, gamma=5/3):
    """Find shock-surface cells and estimate Mach numbers (Schaal+14 §2.2).

    For each shock-zone cell, traces rays along ±ds until exiting the zone,
    then computes the Rankine-Hugoniot Mach number from the T, P, ρ jumps.

    Parameters
    ----------
    snap       : richio Snapshot
    vor        : result of build_voronoi(snap)
    shock_zone : bool array from find_shock_zone()
    gamma      : adiabatic index

    Returns
    -------
    SimpleNamespace with:
        surface_mask : bool (N,)  — global mask for shock surface cells
        pre_mask     : bool (N,)  — mask for pre-shock cells
        post_mask    : bool (N,)  — mask for post-shock cells
        mach_T       : float (M,) — Mach from temperature jump
        mach_P       : float (M,) — Mach from pressure jump
        mach_rho     : float (M,) — Mach from density jump
    """
    P_v   = snap.P.v
    rho_v = snap.rho.v
    grad_P   = np.stack([snap.DpDx.v, snap.DpDy.v, snap.DpDz.v], axis=-1)
    grad_rho = np.stack([snap.DrhoDx.v, snap.DrhoDy.v, snap.DrhoDz.v], axis=-1)
    grad_T   = (grad_P.T / rho_v - P_v * grad_rho.T / rho_v**2).T

    norm_gT = np.linalg.norm(grad_T, axis=-1, keepdims=True)
    ds = np.where(norm_gT > 0, -grad_T / norm_gT, 0.0)

    T_np   = np.ascontiguousarray(P_v / rho_v, dtype=np.float64)
    P_np   = np.ascontiguousarray(P_v,          dtype=np.float64)
    rho_np = np.ascontiguousarray(rho_v,         dtype=np.float64)

    shock_mask = np.asarray(shock_zone, dtype=np.bool_)
    idx_shock  = np.ascontiguousarray(np.where(shock_zone)[0], dtype=np.int64)

    divV_shock = np.ascontiguousarray(snap.divV.v[idx_shock], dtype=np.float64)
    ds_shock   = np.ascontiguousarray(ds[idx_shock],           dtype=np.float64)

    M_T, M_P, M_rho, i_surf, i_pre, i_post = _shock_surface_kernel(
        vor.positions, vor.neighbor_data, vor.neighbor_ptr,
        T_np, P_np, rho_np,
        shock_mask, idx_shock, divV_shock, ds_shock, gamma)

    surface_mask = np.zeros(len(snap), dtype=bool)
    pre_mask     = np.zeros(len(snap), dtype=bool)
    post_mask    = np.zeros(len(snap), dtype=bool)

    surface_mask[idx_shock[i_surf]] = True
    pre_mask[i_pre]                  = True
    post_mask[i_post]                = True

    return SimpleNamespace(
        surface_mask=surface_mask,
        pre_mask=pre_mask,
        post_mask=post_mask,
        mach_T=M_T,
        mach_P=M_P,
        mach_rho=M_rho,
    )
