# Volume rendering (optional)

!!! info "Optional capability"
    3-D volume rendering lives in the **optional** `richio.render` subpackage. It
    pulls in a heavy stack (`yt`, `imageio`) that a plain `import richio` never
    touches. Install it with:

    ```bash
    pip install "richio[render]"
    ```

`richio.render` turns the unstructured Voronoi cells into depth-cued **static
images** and **rotating-camera movies**. It has two layers:

- **`richio.render.grid`** — backend-neutral resampling onto a uniform Cartesian
  grid (`UniformGrid`). No `yt` dependency.
- **`richio.render.yt_backend`** — turns a grid into a `yt` dataset and renders
  it (`volume_image`, `volume_movie`).

## The one-liners

The simplest entry points are on the plotter (they lazily import `yt`):

```python
import richio as rio
snap = rio.load("snap_18.h5")

snap.plots.volume("density", res=256, filename="vr.png")
snap.plots.volume_movie("density", res=256, n_frames=180, filename="spin.mp4")
```

Equivalently, call the functions directly:

```python
import richio.render as rr

rr.volume_image(snap, "density", res=256, filename="vr.png")
rr.volume_movie(snap, "density", res=256, n_frames=180, filename="spin.mp4")
```

## Static images: `volume_image`

```python
rr.volume_image(
    snap, "density", res=256,
    box_size="auto",      # tight box around the dense region (see tight_box)
    mode="volume",        # emission–absorption render
    cmap="magma",
    elevation=20.0, azimuth=0.0, zoom=1.4,
    filename="vr.png",
)
```

Key knobs:

| Keyword | Meaning |
|---------|---------|
| `res` | resampling grid resolution (cubic, or `(nx,ny,nz)`) |
| `box_size` | `"auto"` (tight box around dense gas), `None` (full `snap.box`), or explicit `[x0,y0,z0,x1,y1,z1]` |
| `mode` | `"volume"` (depth-cued render) or `"projection"` (quantitative line-of-sight integral) |
| `tf_mode` | `"map"` (continuous colour + opacity ramp) or `"layers"` (discrete iso-density shells) |
| `alpha`, `gamma` | opacity scale and steepness for `tf_mode="map"` |
| `vmin`, `vmax` | transfer-function bounds in field units (auto from percentiles when `None`) |
| `cmap`, `sigma_clip` | colormap; brightness clip (higher ⇒ darker) |
| `azimuth`, `elevation`, `zoom`, `rot_axis` | camera placement |
| `colorbar` | composite a log colorbar beside the image |
| `filename`, `return_scene` | save a PNG; also return the `yt` Scene |

### Volume vs projection mode

- `mode="volume"` — emission–absorption render with an opacity transfer function:
  looks solid and 3-D, but brightness depends on viewing side.
- `mode="projection"` — off-axis line-of-sight integral (∫field·dl; column
  density for `density`): additive, no occlusion, **quantitative**. The
  transfer-function knobs (`alpha`/`gamma`/`grey_opacity`/`n_layers`/`sigma_clip`)
  are ignored. Pass `weight="density"` for a density-weighted mean instead of a
  plain integral.

## Movies: `volume_movie`

The camera orbits the domain centre about `rot_axis` through `total_angle`
degrees over `n_frames` frames:

```python
rr.volume_movie(
    snap, "density", res=256,
    n_frames=180, total_angle=360.0, fps=30,
    filename="spin.mp4",
)
```

It returns a dict with `frames_dir`, `frame_paths`, and `filename`.

### Single-node multi-core

Each frame is rendered independently, so on a multi-core node just raise
`n_jobs` — frames are spread across forked worker processes (`yt`'s renderer is
single-threaded, so this is the cheap way to use the cores). The grid is shared
copy-on-write:

```python
rr.volume_movie(snap, "density", res=256, n_frames=180,
                n_jobs=16, filename="spin.mp4")
```

This is enough for a typical compute node and needs nothing beyond `[render]`.

### Multi-node (MPI)

Each frame is computed from an **absolute** angle, so rendering an arbitrary
subset is stateless and safe to split across MPI ranks. The hooks are
`frame_indices` (which frames this call renders) and `encode` (only one rank
stitches the movie):

```python
# pattern, run under mpirun across nodes
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

n_frames = 360
mine = range(rank, n_frames, size)          # round-robin frame assignment

rr.volume_movie(
    snap, "density", res=256, n_frames=n_frames,
    frame_indices=mine,                     # this rank's subset
    frames_dir="/shared/frames",            # all ranks write here
    encode=(rank == 0),                     # rank 0 encodes after a barrier
    keep_frames=True,
)
```

Install the MPI extra with `pip install "richio[mpi]"` (see
[Installation](../getting-started/installation.md)).

!!! note
    A ready-made `scripts/render_movie_mpi.py` driver is referenced in the code
    but is not shipped yet — use the pattern above, or single-node `n_jobs`,
    which already saturates a 48-core node.

## Building the grid yourself

For full control, resample once with
[`to_uniform_grid`](../api/render.md) and reuse the `UniformGrid` across many
renders (it builds the k-d tree only once):

```python
from richio.render import to_uniform_grid, tight_box

box  = tight_box(snap, field="density", pct=98.0)   # cube around the densest ~2%
grid = to_uniform_grid(snap, ["density", "temperature"], res=256, box_size=box)

rr.volume_image(snap, "density", grid=grid, filename="rho.png")
rr.volume_image(snap, "temperature", grid=grid, filename="T.png")
```

[`tight_box`](../api/render.md) returns a bounding box around the top-percentile
cells of a field, so the frame is filled with bright structure instead of empty
domain.

See the [`richio.render` API reference](../api/render.md) for every parameter.
