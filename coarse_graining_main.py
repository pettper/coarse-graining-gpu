# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a listener class that computes the physical fields from particle data using the
#              coarse graining method described in the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library JAX for GPU-acceleration.
#
# 2025-12-04: First public version, programmed by Petter Persson.
import os
from functools import partial
from math import ceil, pi

# AGX imports
import jax
import jax.numpy as jnp

from coarse_graining_gpu import (
    C_FORCE_KEY,
    C_NORMAL_KEY,
    C_POS_KEY,
    C_TANGENT_U_KEY,
    C_TANGENT_V_KEY,
    P_DISP_KEY,
    P_MASS_KEY,
    P_POS_KEY,
    P_VEL_KEY,
    coarseGrainingFields,
)

# Constants
INT_32_MAX = jnp.iinfo(jnp.int32).max


class CoarseGrainingMain:
    """
    A coarse graining main class.
    """

    REQUIRED_KEYS = {
        P_POS_KEY,
        P_MASS_KEY,
        P_VEL_KEY,
        P_DISP_KEY,
        C_FORCE_KEY,
        C_NORMAL_KEY,
        C_TANGENT_U_KEY,
        C_TANGENT_V_KEY,
        C_POS_KEY,
    }
    MARGIN = 5  # Margin to determine how large buffer to use for the number of contacts
    PARTICLE_PACKING_DENSITY = 0.75

    def __init__(
        self,
        gridpoints,
        smoothing_length,
        particle_diameter,
        cg_batch_size=100,
        max_num_particles=None,
        debug_prints_on=False,
    ):
        """
        CALL SEQUENCE: cg_listener = CoarseGrainingListenerJAX(
            gridpoints,
            smoothing_length,
            particle_diameter,
            max_num_particles=10000
        )
        INPUTS:
            grid: array of size np x 3, assumed to be regular grid.
            smoothing_length: the smoothing length parameter to use.
            particle_diameter: the mean particle diameter for the granular particles.
            max_num_particles: Optional parameter, to manually put an upper limit
                on the number of particles that fits inside the provided grid.
        """

        self.gridpoints = gridpoints
        self.cg_batch_size = cg_batch_size
        self.debug_prints_on = debug_prints_on

        # Parameters
        self.params = {
            "particleDiameter": particle_diameter,
            "smoothingLength": smoothing_length,
        }

        # Determine the number of particles to include when approximating the cutoff |x| > 3*R. Passed as an environment variable to the function 'coarseGrainingAtPosition' for performance reasons.
        os.environ["NUM_CUTOFF_PARTICLES"] = str(
            ceil(
                0.75 * ((3 * smoothing_length) ** 3) / ((0.5 * particle_diameter) ** 3)
            )
        )
        if self.debug_prints_on:
            print(
                f"NUM_CUTOFF_PARTICLES={int(os.environ.get('NUM_CUTOFF_PARTICLES', '1500'))}"
            )

        # maxNumParticles = rough estimate of number of spheres that fit inside the grid domain limits + 3*smoothingLengths
        if max_num_particles:
            self.maxNumParticles = max_num_particles
        else:
            self.maxNumParticles = self._estimateMaxNumParticles()
        self.inputBuffers = {}
        if self.debug_prints_on:
            print(f"self.maxNumParticles={self.maxNumParticles}")

        return

    def calculate(self, input_buffers: dict):
        """
        Calculates the coarse graining fields for the provided dictionary of
        input buffers.

        INPUTS:
            input_buffers: dictionary that contains all the keys defined "coarse_graining_constants.py".
                            In addition, the particle buffers are assumed to be of equal length, and same
                            for the contact buffers.
        OUTPUTS:
            fields: a dictionary of buffers with the coarse grained fields calculated. The keys are defined
                    in "coarse_graining_constants.py". In addition, the gridpoints are included.
        """
        input_buffers = self._validate_input_buffers(input_buffers)
        input_buffers = self._domainCutoff(input_buffers)
        args = {**input_buffers, **self.params}

        if self.debug_prints_on:
            print("Input buffer sizes")
            for k, v in input_buffers.items():
                print(f"{k}: {v.shape}")

        fields = coarseGrainingFields(
            jnp.array(self.gridpoints, dtype=jnp.float32),
            args,
            batch_size=self.cg_batch_size,
        )
        return fields

    def update_gridpoints(self, gridpoints, max_num_particles=None):
        """
        This function is used to update the gridpoints. In addition, the max_num_particles must also be updated.

        INPUTS:
            gridpoints: New gridpoints stored in an nx3 array. The grid is assumed to be regular.
            max_num_particles: Optional parameter, to manually put an upper limit
                                on the number of particles that fits inside the provided grid.
        """
        self.gridpoints = gridpoints
        if max_num_particles:
            self.maxNumParticles = max_num_particles
        else:
            self.maxNumParticles = self._estimateMaxNumParticles()
        if self.debug_prints_on:
            print(f"self.maxNumParticles={self.maxNumParticles}")

    def set_particle_diameter(self, particle_diameter):
        self.params["particleDiameter"] = particle_diameter

    def set_smoothing_length(self, smoothing_length):
        self.params["smoothingLength"] = smoothing_length

    def _estimateMaxNumParticles(self):
        """
        Function that estimates the maximum number of particles that needs to be allocated for the grid supplied by the user.
        The estimate is based on sphere packing density, the grid volume, and the particle volume.
        INPUTS:
        OUTPUTS:
            n: The recommended number of particles to allocate for the current grid.
        """
        smoothing_length = self.params["smoothingLength"]
        mins = self.gridpoints.min(axis=0)
        maxs = self.gridpoints.max(axis=0)
        limits = {
            "xmin": mins[0],
            "xmax": maxs[0],
            "ymin": mins[1],
            "ymax": maxs[1],
            "zmin": mins[2],
            "zmax": maxs[2],
        }
        sizeX = (limits["xmax"] + 3.0 * smoothing_length) - (
            limits["xmin"] - 3.0 * smoothing_length
        )
        sizeY = (limits["ymax"] + 3.0 * smoothing_length) - (
            limits["ymin"] - 3.0 * smoothing_length
        )
        sizeZ = (limits["zmax"] + 3.0 * smoothing_length) - (
            limits["zmin"] - 3.0 * smoothing_length
        )
        gridVolume = sizeX * sizeY * sizeZ
        particleVolume = (
            (4.0 / 3.0) * pi * ((0.5 * self.params["particleDiameter"]) ** 3)
        )
        return int(self.PARTICLE_PACKING_DENSITY * (gridVolume / particleVolume))

    def _domainCutoff(self, input_buffers):
        """
        Removes particles that are outside the current grid.
        """
        smoothing_length = self.params["smoothingLength"]
        mins = self.gridpoints.min(axis=0)
        maxs = self.gridpoints.max(axis=0)
        limits = {
            "xmin": mins[0] - 3.0 * smoothing_length,
            "xmax": maxs[0] + 3.0 * smoothing_length,
            "ymin": mins[1] - 3.0 * smoothing_length,
            "ymax": maxs[1] + 3.0 * smoothing_length,
            "zmin": mins[2] - 3.0 * smoothing_length,
            "zmax": maxs[2] + 3.0 * smoothing_length,
        }

        particleIndices = jittedCutoffIndices(
            input_buffers[P_POS_KEY], limits, size=self.maxNumParticles
        )
        contactIndices = jittedCutoffIndices(
            input_buffers[C_POS_KEY],
            limits,
            size=self.MARGIN * self.maxNumParticles,
        )

        for key, buffer in input_buffers.items():
            if key in [P_POS_KEY, P_VEL_KEY, P_DISP_KEY, P_MASS_KEY]:
                input_buffers[key] = jittedTake(buffer, particleIndices)
            elif key in [
                C_FORCE_KEY,
                C_POS_KEY,
                C_NORMAL_KEY,
                C_TANGENT_U_KEY,
                C_TANGENT_V_KEY,
            ]:
                input_buffers[key] = jittedTake(buffer, contactIndices)
        return input_buffers

    def _validate_input_buffers(self, input_buffers):
        missing_keys = self.REQUIRED_KEYS - input_buffers.keys()
        input_buffers = {
            k: input_buffers[k] for k in self.REQUIRED_KEYS if k in input_buffers
        }
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        return input_buffers


@partial(jax.jit, static_argnames=["size"])
def jittedCutoffIndices(particlePos, limits, size):
    """
    Finds indexes of particle inside the grid limits.

    Returns: A buffer of length 'size' containing the indices of particles/contacts inside 'limits'
    """
    mask = (
        (particlePos[:, 0] >= limits["xmin"])
        & (particlePos[:, 0] <= limits["xmax"])
        & (particlePos[:, 1] >= limits["ymin"])
        & (particlePos[:, 1] <= limits["ymax"])
        & (particlePos[:, 2] >= limits["zmin"])
        & (particlePos[:, 2] <= limits["zmax"])
    )
    # Fill with an out of bounds index
    return jnp.nonzero(mask, size=size, fill_value=INT_32_MAX)[0]


@jax.jit
def jittedTake(buffer, indices):
    """
    Efficient array indexing along axis=0, fills out of bounds indices with floating point zeros.
    This ensures that these values won't affect coarse graining calculations downstream.

    Returns: A buffer of the same length as indices
    """
    return jnp.take(
        buffer,
        indices,
        axis=0,
        mode="fill",
        fill_value=0.0,
        indices_are_sorted=False,
        unique_indices=True,
    )
