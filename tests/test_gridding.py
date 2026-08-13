import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial import KDTree
import unyt as u

from richio.data import Snapshot, _iter_3d_nearest_slabs
from richio.plots import SnapshotPlotter


class ArraySnapshot(Snapshot):
    """Small in-memory snapshot used to exercise the public gridding API."""

    _field_aliases = {
        "X": [],
        "Y": [],
        "Z": [],
        "Density": ["density"],
        "Volume": ["volume"],
        "Box": ["box"],
    }
    _alias_to_canonical = {}

    def __init__(self):
        axis = np.linspace(-0.9, 0.9, 7)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        self._fields = {
            "X": x.ravel() * u.cm,
            "Y": y.ravel() * u.cm,
            "Z": z.ravel() * u.cm,
            "Density": (2.0 + x + 2.0 * y + 3.0 * z).ravel() * u.g / u.cm**3,
            # Large enough that volume_selection retains cells for every test plane.
            "Volume": np.full(x.size, 0.5) * u.cm**3,
            "Box": u.unyt_array([-1, -1, -1, 1, 1, 1], "cm"),
        }
        super().__init__("snap_0")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            field, index = key
        else:
            field, index = key, slice(None)
        value = self._fields[self._resolve_field_name(field)]
        return value if np.ndim(value) == 0 else value[index]

    def keys(self):
        return sorted(self._fields)


class GriddingTests(unittest.TestCase):
    def setUp(self):
        self.snap = ArraySnapshot()

    def assert_unyt_equal(self, left, right):
        self.assertEqual(left.units, right.units)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_slice_workers_are_identical_for_all_planes(self):
        for plane in ("xy", "xz", "yz"):
            with self.subTest(plane=plane):
                serial = self.snap.slice(
                    "density",
                    res=(13, 11),
                    plane=plane,
                    slice_coord=0.1 * u.cm,
                    volume_selection=False,
                    workers=1,
                )
                threaded = self.snap.slice(
                    "density",
                    res=(13, 11),
                    plane=plane,
                    slice_coord=0.1 * u.cm,
                    volume_selection=False,
                    workers=8,
                )
                for actual, expected in zip(threaded, serial):
                    self.assert_unyt_equal(actual, expected)

    def test_slice_selection_custom_box_and_all_workers(self):
        selection = np.asarray(self.snap.X) > -0.7
        serial = self.snap.slice(
            "density",
            res=9,
            box_size=[-0.5, -0.4, 0.7, 0.8],
            selection=selection,
            volume_selection=False,
            workers=1,
        )
        threaded = self.snap.slice(
            "density",
            res=9,
            box_size=[-0.5, -0.4, 0.7, 0.8],
            selection=selection,
            volume_selection=False,
            workers=-1,
        )
        for actual, expected in zip(threaded, serial):
            self.assert_unyt_equal(actual, expected)

    def test_streamed_projection_matches_full_cube_reference(self):
        selection = np.asarray(self.snap.Y) < 0.8
        for plane in (None, "xy", "xz", "yz"):
            with self.subTest(plane=plane):
                kwargs = {
                    "res": (10, 9, 8),
                    "box_size": [-0.8, -0.7, -0.6, 0.9, 0.8, 0.7],
                    "selection": selection,
                    "plane": plane,
                }
                indices, xspace, yspace, zspace = self.snap.to_3dgrid(
                    **kwargs, workers=1
                )
                coords, source_indices, _, _, _ = self.snap._prepare_3d_grid(**kwargs)
                streamed_indices = np.empty(
                    (len(xspace) - 1, len(yspace) - 1, len(zspace) - 1),
                    dtype=np.intp,
                )
                for start, local_indices in _iter_3d_nearest_slabs(
                    KDTree(coords),
                    xspace[:-1],
                    yspace[:-1],
                    zspace[:-1],
                    workers=1,
                ):
                    absolute_indices = (
                        source_indices[local_indices]
                        if source_indices is not None
                        else local_indices
                    )
                    streamed_indices[start : start + len(local_indices)] = absolute_indices
                np.testing.assert_array_equal(
                    streamed_indices, indices[:-1, :-1, :-1]
                )
                expected = np.sum(
                    self.snap.density[indices][:-1, :-1, :-1]
                    * (zspace[1:] - zspace[:-1]),
                    axis=-1,
                ).in_base("cgs")
                actual, actual_x, actual_y = self.snap.project(
                    "density", **kwargs, workers=8
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
                self.assertEqual(actual.units, expected.units)
                self.assert_unyt_equal(actual_x, xspace)
                self.assert_unyt_equal(actual_y, yspace)

    def test_scalar_projection_and_worker_counts_match(self):
        serial = self.snap.project("density", res=8, workers=1)
        threaded = self.snap.project("density", res=8, workers=-1)
        self.assertEqual(threaded[0].units, serial[0].units)
        np.testing.assert_allclose(threaded[0], serial[0], rtol=1e-14, atol=0)
        self.assert_unyt_equal(threaded[1], serial[1])
        self.assert_unyt_equal(threaded[2], serial[2])

    def test_invalid_and_empty_selections_are_clear(self):
        with self.assertRaisesRegex(ValueError, "selection must have shape"):
            self.snap.project("density", res=8, selection=np.ones(3, dtype=bool))
        with self.assertRaisesRegex(ValueError, "empty cell selection"):
            self.snap.project(
                "density", res=8, selection=np.zeros(len(self.snap.X), dtype=bool)
            )
        with self.assertRaisesRegex(ValueError, "empty cell selection"):
            self.snap.slice(
                "density",
                res=8,
                selection=np.zeros(len(self.snap.X), dtype=bool),
                volume_selection=False,
            )

    def test_projection_resolution_must_support_integration(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.snap.project("density", res=(8, 8, 1))


class PlotWorkerForwardingTests(unittest.TestCase):
    @patch("richio.plots.scalar_map")
    def test_slice_forwards_workers(self, scalar_map):
        snap = Mock()
        grid = np.ones((2, 2)) * u.g / u.cm**3
        axis = np.arange(2) * u.cm
        snap.slice.return_value = grid, axis, axis
        scalar_map.return_value = Mock(), Mock()

        SnapshotPlotter(snap).slice("density", 2, workers=-1)

        self.assertEqual(snap.slice.call_args.kwargs["workers"], -1)

    @patch("richio.plots.scalar_map")
    def test_projection_forwards_workers(self, scalar_map):
        snap = Mock()
        grid = np.ones((2, 2)) * u.g / u.cm**2
        axis = np.arange(3) * u.cm
        snap.project.return_value = grid, axis, axis
        scalar_map.return_value = Mock(), Mock()

        SnapshotPlotter(snap).projection("density", 3, workers=4)

        self.assertEqual(snap.project.call_args.kwargs["workers"], 4)


if __name__ == "__main__":
    unittest.main()
