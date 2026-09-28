from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SpatialGrid(NamedTuple):
    origin: jax.Array  # (3,) lower corner of the grid.
    dims: jax.Array  # (3,) number of cells along each axis.
    cell_size: jax.Array  # Side length of the cubic cells, equal to the search radius.
    sorted_keys: jax.Array  # (N,) flat cell index of each particle, in ascending order.
    order: jax.Array  # (N,) particle indices sorted by cell.
    sorted_particles: jax.Array  # (N, 3) particle positions sorted by cell.


def flat_cell_index(ijk, dims):
    return (ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2]


@jax.jit
def build_spatial_grid(gridpoint_limits, particles, cell_size_aka_search_radius):
    """
    Sorts particles into a dense grid of cubic cells covering the gridpoint limits, padded by one cell on each side.
    CALL SEQUENCE: grid = build_spatial_grid(gridpoint_limits, particles, cell_size_aka_search_radius)
    INPUTS:
        gridpoint_limits: (Xmin, Ymin, Zmin, Xmax, Ymax, Zmax) bounding box of all query points.
        particles: array of particle positions, size N x 3.
        cell_size_aka_search_radius: side length of the cells, and the search radius used by query_spatial_grid.
    OUTPUTS:
        grid: SpatialGrid, a pytree that can be passed to jitted functions.
    """
    c = cell_size_aka_search_radius
    limits = jnp.asarray(gridpoint_limits)
    origin = limits[:3] - c
    dims = jnp.floor((limits[3:] - limits[:3]) / c).astype(jnp.int32) + 3
    ijk = jnp.floor((particles - origin) / c).astype(jnp.int32)
    # Particles outside the grid are further than the search radius from every query point, their key sorts them last.
    inside = jnp.all((ijk >= 0) & (ijk < dims), axis=1)
    keys = jnp.where(inside, flat_cell_index(ijk, dims), jnp.iinfo(jnp.int32).max)
    sorted_keys, order = jax.lax.sort_key_val(keys, jnp.arange(keys.shape[0]))
    return SpatialGrid(origin, dims, c, sorted_keys, order, particles[order])


@partial(jax.jit, static_argnames=("k", "max_candidates", "ord", "batch_size"))
def query_spatial_grid(grid, points, k, max_candidates, ord=2, batch_size=1024):
    """
    Finds all particles within the search radius of each point, the result is exact where overflow is False.
    CALL SEQUENCE: indices, overflow = query_spatial_grid(grid, points, k, max_candidates, ord=2, batch_size=1024)
    INPUTS:
        grid: SpatialGrid from build_spatial_grid.
        points: array of query points inside the gridpoint limits, size nq x 3.
        k: maximum number of neighbours per point.
        max_candidates: number of particles examined per point, must hold all particles in the 27 surrounding cells.
            This is about 27 / (4 pi / 3) ~ 6.5 times the particles within the radius for ord=2, and 27 / 8 ~ 3.4 for ord=inf.
        ord: norm used for distances, 2 or jnp.inf.
        batch_size: number of points processed in parallel, bounds memory use to about batch_size x max_candidates.
    OUTPUTS:
        indices: particle indices in no particular order, size nq x k. Unused entries are set to N.
        overflow: boolean array, size (nq,), True where particles were missed since k or max_candidates was too small.
    """
    # The cells (i, j, l-1), (i, j, l), (i, j, l+1) are consecutive in the sorted keys, so the 27 cells form 9 ranges.
    first_cell_offsets = jnp.array([(di, dj, -1) for di in (-1, 0, 1) for dj in (-1, 0, 1)])
    slot = jnp.arange(max_candidates)

    def query_point(x):
        # Clipping only matters for points on the grid border, and then only drops cells outside the grid that are empty.
        ijk = jnp.clip(jnp.floor((x - grid.origin) / grid.cell_size).astype(jnp.int32), 1, grid.dims - 2)
        first_cell = flat_cell_index(ijk + first_cell_offsets, grid.dims)
        start = jnp.searchsorted(grid.sorted_keys, first_cell, side="left", method="scan_unrolled")
        end = jnp.searchsorted(grid.sorted_keys, first_cell + 2, side="right", method="scan_unrolled")

        # Lay out the 9 ranges back to back in the candidate slots.
        range_end = jnp.cumsum(end - start)
        r = jnp.searchsorted(range_end, slot, side="right", method="compare_all")
        idx = end[r] - range_end[r] + slot
        dist = jnp.linalg.norm(grid.sorted_particles[idx] - x, ord=ord, axis=-1)
        in_range = (slot < range_end[-1]) & (dist <= grid.cell_size)

        # Compact the particles within range to the front, those beyond k are dropped.
        rank = jnp.cumsum(in_range) - 1
        indices = jnp.full(k, grid.order.shape[0]).at[jnp.where(in_range, rank, k)].set(grid.order[idx], mode="drop")
        return indices, (range_end[-1] > max_candidates) | (rank[-1] >= k)

    return jax.lax.map(query_point, points, batch_size=batch_size)
