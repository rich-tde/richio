# Volume rendering (optional)

!!! info "This is an optional extra"
    Volume rendering lives in `richio.render`, which relies on a couple of large
    libraries (`yt` and `imageio`) that a plain `import richio` never loads.
    Install it when you want it:

    ```bash
    pip install "richio[render]"
    ```

A slice shows you one plane. A volume render shows you the whole cloud of gas at
once, shaded so that depth reads as depth: bright dense structure in front, fog
behind. `richio.render` makes still images and rotating movies from a snapshot.

## The one-liners

The quickest way in is through the plotter:

```python
import richio as rio
snap = rio.load("snap_18.h5")

snap.plots.volume("density", res=256, filename="vr.png")
snap.plots.volume_movie("density", res=256, n_frames=180, filename="spin.mp4")
```

These are shortcuts for the two functions in the module, which you can also call
directly:

```python
import richio.render as rr

rr.volume_image(snap, "density", res=256, filename="vr.png")
rr.volume_movie(snap, "density", res=256, n_frames=180, filename="spin.mp4")
```

## A still image

```python
rr.volume_image(
    snap, "density", res=256,
    box_size="auto",      # frame the box tightly around the dense gas
    cmap="magma",
    elevation=20.0, azimuth=0.0, zoom=1.4,
    filename="vr.png",
)
```

The arguments fall into a few groups.

**What and how much to render.** `field` is the field, `res` is the resolution of
the grid the gas is resampled onto before rendering (a single number for a cube,
or `(nx, ny, nz)`). `box_size` chooses the region: `"auto"` draws a tight box
around the dense gas so the frame is full of structure rather than empty space,
`None` uses the whole simulation box, or you can give explicit bounds
`[x0, y0, z0, x1, y1, z1]`.

**Where the camera sits.** `azimuth` and `elevation` (both in degrees) swing the
camera around and above the gas, and `zoom` moves it in or out.

**Colour and brightness.** `cmap` is the colormap. `vmin` and `vmax` set the
range of field values that the colours and opacity span; left unset, they are
chosen from the data so the dense core stays visible. `sigma_clip` controls the
brightest pixels, so turn it up if the image looks blown out. `colorbar` adds a
labelled colorbar beside the image.

**Output.** `filename` saves a PNG. Pass `return_scene=True` to also get back
`yt`'s scene object if you want to keep adjusting it.

### How the gas is coloured: the transfer function

A volume render has to decide, for every value of density, what colour to paint
it and how opaque (how solid) it should be. That mapping is called the *transfer
function*, and `tf_mode` picks between two styles:

- `"map"` (the default) blends a continuous colour ramp with opacity that rises
  toward high density, so the dense core glows through the thinner gas around it.
  `alpha` scales the overall opacity and `gamma` controls how sharply it ramps
  up.
- `"layers"` draws a handful of discrete shells at fixed densities, like contour
  surfaces in 3-D. `n_layers` sets how many.

### A render vs. a measurement

`mode` decides what the image actually means:

- `mode="volume"` (the default) is the shaded look described above. Because
  nearer gas hides gas behind it, the picture is good for seeing structure, but
  the brightness depends on which side you view from, so it is not something you
  would measure off.
- `mode="projection"` instead adds the field up straight along each line of sight
  (for density, that is the column density). Nothing hides anything else and the
  result is quantitative, a real measurement you can put numbers on. In this mode
  the transfer-function knobs do not apply, since there is no shading involved.
  Pass `weight="density"` to get a density-weighted average along the line of
  sight instead of a plain sum.

## Movies

`volume_movie` orbits the camera around the gas and stitches the frames into a
video:

```python
rr.volume_movie(
    snap, "density", res=256,
    n_frames=180, total_angle=360.0, fps=30,
    filename="spin.mp4",
)
```

`n_frames` is how many frames to render, `total_angle` how far the camera travels
(360° for a full turn), and `fps` the playback rate. It returns a small
dictionary with the output filename and where the individual frames were written.

## Rendering faster

A movie is many independent images, so you can speed it up by rendering several at
once.

**On one machine**, raise `n_jobs` to spread the frames across the cores. (The
renderer itself runs on a single core, so handing it whole frames in parallel is
how you use the rest.) This needs nothing beyond the `[render]` extra and is
enough for a typical compute node:

```python
rr.volume_movie(snap, "density", res=256, n_frames=180,
                n_jobs=16, filename="spin.mp4")
```

**Across several machines (MPI)** is rarely necessary, but the option is there.
Each frame is computed from an absolute camera angle, so any machine can render
any subset of frames without coordinating with the others. You tell each one
which frames to do with `frame_indices`, point them all at a shared `frames_dir`,
and let a single one stitch the result with `encode`:

```python
# run under mpirun across nodes
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

n_frames = 360
mine = range(rank, n_frames, size)          # this rank takes every size-th frame

rr.volume_movie(
    snap, "density", res=256, n_frames=n_frames,
    frame_indices=mine,                     # the frames this machine renders
    frames_dir="/shared/frames",            # all machines write here
    encode=(rank == 0),                     # one machine stitches the movie
    keep_frames=True,
)
```

This needs the MPI extra (`pip install "richio[mpi]"`), built against your
cluster's MPI; see [Installation](../getting-started/installation.md). Unless you
are rendering something enormous, single-machine `n_jobs` is simpler and already
fast.

## Resampling once for many renders

Both `volume_image` and `volume_movie` start by resampling the irregular cells
onto a uniform grid, which involves building a k-d tree, the slow step. If you are
going to render the same snapshot several times (different fields, different
camera angles), build that grid once and reuse it:

```python
from richio.render import to_uniform_grid, tight_box

box  = tight_box(snap, field="density", pct=98.0)   # a cube around the densest ~2% of cells
grid = to_uniform_grid(snap, ["density", "temperature"], res=256, box_size=box)

rr.volume_image(snap, "density", grid=grid, filename="rho.png")
rr.volume_image(snap, "temperature", grid=grid, filename="T.png")
```

`tight_box` returns a box drawn around the brightest cells of a field, so the
frame is filled with the structure you care about instead of empty domain.

See the [`richio.render` API reference](../api/render.md) for every parameter.
