from functools import partial

import jax
import jax.numpy as jnp
from math import ceil

def flat_cell_index(ijk, dims):
    """
    Flatten spatial grid index to 1-dimensional array index. Order is row major "C".
    INPUTS:
        ijk: array of cell indices, size (n, 3).
        dims: spatil grid dimensions, size (3,).
    """
    return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]


@jax.jit
def build_spatial_grid(gridpoint_limits, particle_positions, cell_size_and_search_radius):
    """
    Builds a spatial grid of cubic cells covering the gridpoint limits, padded by one cell on each side.
    CALL SEQUENCE: grid = build_spatial_grid(gridpoint_limits, particles, cell_size_aka_search_radius)
    INPUTS:
        gridpoint_limits: (Xmin, Ymin, Zmin, Xmax, Ymax, Zmax) bounding box of all query points.
        particle_positions: array of particle positions, size N x 3.
        cell_size_and_search_radius: side length of the cells, and the search radius used by query_spatial_grid.
    OUTPUTS:
        grid: dict with the keys
            "origin": lower corner of the grid, size (3,).
            "dims": number of cells along each axis, size (3,).
            "cell_size": side length of the cubic cells, equal to the search radius.
            "sorted_keys": flat cell index of each particle in ascending order, size (N,).
            "order": particle indices sorted by cell, size (N,).
            "sorted_particles": particle positions sorted by cell, size N x 3.
    """

    # To make a spatial grid, padded with one cell in each dimension.
    limits = jnp.asarray(gridpoint_limits)
    origin = limits[:3] - cell_size_and_search_radius
    grid_dims = jnp.floor((limits[3:] - limits[:3]) / cell_size_and_search_radius).astype(jnp.int32) + 3

    # To sort particles into the grid.
    particle_ijk = jnp.floor((particle_positions - origin) / cell_size_and_search_radius).astype(jnp.int32)

    # Particles outside the grid are further than the search radius from every query point, their key sorts them last.
    inside_grid = jnp.all((particle_ijk >= 0) & (particle_ijk < grid_dims), axis=1)
    keys = jnp.where(inside_grid, flat_cell_index(particle_ijk, grid_dims), jnp.iinfo(jnp.int32).max)
    sorted_keys, order = jax.lax.sort_key_val(keys, jnp.arange(keys.shape[0]))
    return {
        "origin": origin,
        "dims": grid_dims,
        "cell_size": cell_size_and_search_radius,
        "sorted_keys": sorted_keys,
        "order": order,
        "sorted_particles": particle_positions[order],
    }


@partial(jax.jit, static_argnames=("max_candidates", "max_neighbours", "ord", "batch_size"))
def query_spatial_grid(grid, query_points, max_candidates, max_neighbours, ord=2, batch_size=128):
    """
    Finds all particles within the search radius of each point, the result is exact where overflow is False.
    CALL SEQUENCE: indices, overflow = query_spatial_grid(grid, query_points, max_candidates, max_neighbours, ord=2, batch_size=128)
    INPUTS:
        grid: dict from build_spatial_grid.
        query_points: array of query points inside the gridpoint limits, size nq x 3.
        max_candidates: number of particles examined per point, must hold all particles in the 27 surrounding cells.
        max_neighbours: number of particles returned per point, must hold all particles within the search radius.
        ord: norm used for distances, 2 or jnp.inf.
        batch_size: number of points processed in parallel, bounds memory use to about batch_size x max_candidates.
    OUTPUTS:
        indices: particle indices in no particular order, size nq x max_neighbours. Unused entries are set to N.
        overflow: boolean array, size (nq,), True where particles were missed since max_candidates or max_neighbours was too small.
    """

    sorted_keys, order, dims = grid["sorted_keys"], grid["order"], grid["dims"]
    
    # The cells (i, j, l-1), (i, j, l), (i, j, l+1) are consecutive sprted buffers (keys, particles), so the 27 cells form 9 ranges of candidates to gather.
    first_cell_offsets = jnp.array([(di, dj, -1) for di in (-1, 0, 1) for dj in (-1, 0, 1)])
    max_particles_per_range = ceil(max_candidates / 9)
    slot_indices = jnp.arange(max_particles_per_range)

    def query_point(x):
        query_ijk = jnp.floor((x - grid["origin"]) / grid["cell_size"]).astype(jnp.int32)
        first_cells = flat_cell_index(query_ijk + first_cell_offsets, dims)
        
        # To find the start and end indices of the 9 ranges of candidates to gather.
        range_start_idx = jnp.searchsorted(sorted_keys, first_cells, side="left", method="scan_unrolled")
        range_end_idx = jnp.searchsorted(sorted_keys, first_cells + 2, side="right", method="scan_unrolled")

        # To construct the indices of the candidates to gather.
        range_length = range_end_idx - range_start_idx
        pick = jnp.ravel(range_start_idx[:, jnp.newaxis] + slot_indices)
        is_valid = jnp.ravel(slot_indices < range_length[:, jnp.newaxis])

        # To keep the candidates within the search radius, moved to the front of max_neighbours slots. Unused slots get keep = -1.
        dist = jnp.linalg.norm(grid["sorted_particles"][pick] - x, ord=ord, axis=-1)
        in_range = (is_valid) & (dist <= grid["cell_size"])
        keep = jnp.nonzero(in_range, size=max_neighbours, fill_value=-1)[0]
        indices = jnp.where(keep >= 0, order[pick[keep]], order.shape[0])

        overflow = jnp.any(range_length > max_particles_per_range) | (jnp.sum(in_range) > max_neighbours)
        return indices, overflow

    return jax.lax.map(query_point, query_points, batch_size=batch_size)
