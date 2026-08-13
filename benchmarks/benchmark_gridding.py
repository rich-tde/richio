"""Benchmark streamed projection timing and peak memory on a RICH snapshot.

This is intentionally not part of the test suite. Example::

    python benchmarks/benchmark_gridding.py snap_0042.h5 --res 256 --res 512
"""

import argparse
import json
import multiprocessing as mp
import resource
from time import perf_counter

import numpy as np
from scipy.spatial import KDTree
import unyt as u

import richio
from richio.data import _iter_3d_nearest_slabs


def _run_one(path, field, res, workers, queue):
    started = perf_counter()
    snap = richio.load(path)

    tick = perf_counter()
    x, y, z = snap.X, snap.Y, snap.Z
    data = snap._get_data(field)
    field_seconds = perf_counter() - tick

    tick = perf_counter()
    coords, source_indices, xspace, yspace, zspace = snap._prepare_3d_grid(
        res=res, X=x, Y=y, Z=z
    )
    preparation_seconds = perf_counter() - tick

    tick = perf_counter()
    tree = KDTree(coords)
    tree_seconds = perf_counter() - tick

    tick = perf_counter()
    dz = np.asarray(zspace[1:] - zspace[:-1])
    values = np.asarray(data)
    projected = np.empty((res - 1, res - 1), dtype="float64")
    for a, local_indices in _iter_3d_nearest_slabs(
        tree, xspace[:-1], yspace[:-1], zspace[:-1], workers=workers
    ):
        indices = (
            source_indices[local_indices]
            if source_indices is not None
            else local_indices
        )
        projected[a : a + len(indices)] = np.sum(values[indices] * dz, axis=-1)
    query_integration_seconds = perf_counter() - tick

    result = u.unyt_array(projected, data.units * zspace.units).in_base("cgs")
    queue.put(
        {
            "resolution": res,
            "workers": workers,
            "cells": len(coords),
            "field_loading_s": field_seconds,
            "grid_preparation_s": preparation_seconds,
            "tree_build_s": tree_seconds,
            "query_integration_s": query_integration_seconds,
            "total_s": perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "checksum": float(np.sum(np.asarray(result))),
            "unit": str(result.units),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="HDF5 snapshot file or NPY snapshot directory")
    parser.add_argument("--field", default="density", help="field to project")
    parser.add_argument(
        "--res",
        type=int,
        action="append",
        dest="resolutions",
        help="cubic grid resolution; repeat to benchmark several (default: 256)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, -1],
        help="worker counts to compare (default: 1 2 4 8 16 -1)",
    )
    args = parser.parse_args()

    context = mp.get_context("spawn")
    for res in args.resolutions or [256]:
        for workers in args.workers:
            queue = context.Queue()
            process = context.Process(
                target=_run_one,
                args=(args.snapshot, args.field, res, workers, queue),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise SystemExit(
                    f"benchmark failed for res={res}, workers={workers} "
                    f"with exit code {process.exitcode}"
                )
            print(json.dumps(queue.get(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
