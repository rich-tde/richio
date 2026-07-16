# Installation

RICHIO needs Python 3.10 or newer.

## The basic install

There is no PyPI release yet, so install from the source. Clone the repository
and install it in editable mode (the `-e` flag lets you pull updates with
`git pull` without reinstalling):

```bash
git clone https://github.com/YujieH3/richio.git
cd richio
pip install -e .
```

This pulls in everything the core library needs: `numpy`, `h5py`, `matplotlib`,
and `unyt` (the unit system). Once it finishes, import the package:

```python
import richio as rio
```

The documentation shortens `richio` to `rio` everywhere. You don't have to;
`import richio` works just as well, but the examples all use `rio`.

## Optional extras

Two features rely on larger third-party libraries. They are kept separate so a
plain install stays small and quick to import, so add them only if you need them.

3-D volume rendering ([`richio.render`](../guide/volume-rendering.md)) draws
shaded images and movies of the gas. It needs `yt` and `imageio`:

```bash
pip install "richio[render]"
```

Shock finding ([`richio.shockfinder`](../guide/shock-finding.md)) locates shock
fronts. It uses `scipy` and `numba`, which aren't installed automatically, so add
them yourself:

```bash
pip install scipy numba
```

There is also an MPI extra for rendering a single movie across several cluster
nodes at once. Most people never need it, since rendering on the many cores of
one machine is already fast, so its setup lives in the
[volume rendering guide](../guide/volume-rendering.md#rendering-faster).

## Checking it worked

The quickest test that doesn't need any data:

```python
>>> import richio as rio
>>> rio.__name__
'richio'
```

If you have a RICH snapshot on hand, open it and print a summary. This exercises
the whole reading path:

```python
snap = rio.load("snap_0042.h5")
snap.info()
```

Next: the [Quickstart](quickstart.md).
