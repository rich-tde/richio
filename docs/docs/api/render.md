# `richio.render`

!!! info "Optional subpackage"
    Requires `pip install "richio[render]"` (`yt`, `imageio`) at runtime. See the
    [Volume rendering guide](../guide/volume-rendering.md).

## `richio.render.grid`

Backend-neutral resampling onto a uniform Cartesian grid.

::: richio.render.grid

## `richio.render.yt_backend`

`yt`-backed volume images and movies.

::: richio.render.yt_backend
    options:
      members:
        - to_yt
        - volume_image
        - volume_movie
