# Installation

RICHIO targets **Python ≥ 3.10**.

## Core install

Clone the repository and install in editable mode from its root directory:

```bash
git clone https://github.com/YujieH3/richio.git
cd richio
pip install -e .
```

The core install pulls in `numpy`, `h5py`, and `matplotlib` (plus `unyt` for the
unit system). After installing, import the package:

```python
import richio as rio
```

By convention we alias it to `rio` throughout the docs.

## Optional extras

Some capabilities depend on heavier third-party stacks and are kept optional so a
plain `pip install` stays lean.

=== "3-D volume rendering"

    Depth-cued volume renders and rotating-camera movies
    ([`richio.render`](../guide/volume-rendering.md)) need `yt` and `imageio`:

    ```bash
    pip install "richio[render]"
    ```

=== "Multi-node movie rendering"

    Frame-parallel movie rendering across **multiple nodes** additionally needs
    `mpi4py` built against a system/cluster MPI:

    ```bash
    # On a cluster, load an MPI module first, e.g.:
    module load OpenMPI
    pip install "richio[mpi]"
    # or:  conda install -c conda-forge mpi4py
    ```

    Single-node multi-core rendering (`volume_movie(..., n_jobs=N)`) needs nothing
    beyond the `[render]` extra.

=== "Shock finding"

    The shock finder ([`richio.shockfinder`](../guide/shock-finding.md)) uses
    `scipy` and `numba`. These are not declared as core dependencies yet, so
    install them explicitly:

    ```bash
    pip install scipy numba
    ```

## Verifying the install

```python
import richio as rio
print(rio.__name__)          # 'richio'
```

If you have a RICH snapshot handy, the fastest end-to-end check is:

```python
snap = rio.load("snap_0042.h5")
snap.info()
```

Next: the [Quickstart](quickstart.md).
