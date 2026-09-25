# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a class that computes the physical fields from particle data using the
#              coarse graining method described in the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library JAX for GPU-acceleration.
#
# 2025-12-04: First public version, programmed by Petter Persson.
import os
from math import ceil
from enum import Enum

# AGX imports
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree

from .src.coarse_graining_constants import (
    C_FORCE_KEY,
    C_NORMAL_KEY,
    C_POS_KEY,
    C_TANGENT_U_KEY,
    C_TANGENT_V_KEY,
    P_DISP_KEY,
    P_MASS_KEY,
    P_POS_KEY,
    P_VEL_KEY,
)
from .src.coarse_graining_warp import CoarseGrainingWarp
from .src.coarse_graining_jax import coarseGrainingFields as cg_fields_jax


class GPUBackend(Enum):
    WARP = 0
    JAX = 1


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
    STANDARD_PARTICLE_BUFFER_SIZE = 1024
    STANDARD_GRIDPOINTS_BUFFER_SIZE = 256

    def __init__(
        self,
        gridpoints,
        smoothing_length,
        particle_diameter,
        cg_batch_size=100,
        debug_prints_on=False,
        backend=GPUBackend.WARP,
    ):
        """
        CALL SEQUENCE: cg_listener = CoarseGrainingListenerJAX(
            gridpoints,
            smoothing_length,
            particle_diameter,
            cg_batch_size=100,
            debug_prints_on=False,
            backend="jax",
        )
        INPUTS:
            grid: array of size np x 3, assumed to be regular grid.
            smoothing_length: the smoothing length parameter to use.
            particle_diameter: the mean particle diameter for the granular particles.
            cg_batch_size: the batch size to use for the coarse graining calculation on the GPU.
            debug_prints_on: Boolean, whether to print debug information.
            backend: String, either "jax" or "warp". The coarse graining calculation is performed on the GPU using the
                     coarseGrainingFields function from the coarse_graining_jax or coarse_graining_warp modules.
        """

        self._warp_cg = None
        self.backend = self.set_backend(backend)
        self.cg_batch_size = cg_batch_size
        self.debug_prints_on = debug_prints_on

        self.gridpoints = np.zeros([10, 3])  # Just set something, it is immidiately updated with correct input.
        self.number_of_gridpoints = self.gridpoints.shape[0]
        self.update_gridpoints(gridpoints)

        # Parameters
        self.params = {
            "particleDiameter": particle_diameter,
            "smoothingLength": smoothing_length,
        }

        if self.debug_prints_on:
            print(f"NUM_CUTOFF_PARTICLES={int(os.environ.get('NUM_CUTOFF_PARTICLES', '1500'))}")

        # To handle adaptive buffer sizes
        self.particle_buffer_size = self.STANDARD_PARTICLE_BUFFER_SIZE
        self.contact_buffer_size = self.STANDARD_PARTICLE_BUFFER_SIZE
        if self.debug_prints_on:
            print(f"Particle buffer size: {self.particle_buffer_size}")
            print(f"Contact buffer size: {self.contact_buffer_size}")

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
        if self.backend == GPUBackend.JAX:
            # This required for the JAX-implementation to logically work correctly.
            input_buffers = self._domainCutoff(input_buffers)
        else:
            pass  # Warp does not care about sizes, and code runs faster in tests if we just send our input data.

        args = {**input_buffers, **self.params}
        fields = self.coarseGrainingFields(
            self.gridpoints,
            args,
            batch_size=self.cg_batch_size,
        )
        fields = {k: v[: self.number_of_gridpoints] for k, v in fields.items()}
        return fields

    def update_gridpoints(self, gridpoints):
        """
        This function is used to update the gridpoints.

        INPUTS:
            gridpoints: New gridpoints stored in an nx3 array.
        """

        assert gridpoints.ndim == 2 and gridpoints.shape[1] == 3
        self.number_of_gridpoints = gridpoints.shape[0]

        if self.backend == GPUBackend.JAX:
            size = int(
                ceil(self.number_of_gridpoints / self.STANDARD_GRIDPOINTS_BUFFER_SIZE)
                * self.STANDARD_GRIDPOINTS_BUFFER_SIZE
            )
            if not size == self.gridpoints.shape[0]:
                self.gridpoints = self._buffer_pad(np.asarray(gridpoints), size, pad_value=0.0)
                if self.debug_prints_on:
                    print(f"Gridpoints buffer size changed to {size}")
            else:
                self.gridpoints[: self.number_of_gridpoints] = np.asarray(gridpoints)
        else:
            self.gridpoints = np.asarray(gridpoints)

    def set_particle_diameter(self, particle_diameter):
        self.params["particleDiameter"] = particle_diameter

    def set_smoothing_length(self, smoothing_length):
        self.params["smoothingLength"] = smoothing_length

    def set_backend(self, backend):
        assert isinstance(backend, GPUBackend)
        self.backend = backend
        if self.backend == GPUBackend.WARP:
            self._warp_cg = CoarseGrainingWarp() if self._warp_cg is None else self._warp_cg
            self.coarseGrainingFields = lambda gp, args, batch_size: self._warp_cg.coarseGrainingFields(gp, args)
        elif self.backend == GPUBackend.JAX:
            self.coarseGrainingFields = cg_fields_jax
        else:
            raise NotImplementedError(f"Backend {backend} not implemented.")
        return self.backend

    def _domainCutoff(self, input_buffers):
        """
        Removes particles that are outside the current grid.
        """

        def pick_indices(pos, mins, maxs):
            """
            Finds indexes of particle inside the grid limits.
            Returns: A buffer of length 'size' containing the indices of particles/contacts inside 'limits'
            """
            return np.flatnonzero(np.all((pos >= mins) & (pos <= maxs), axis=1))

        smoothing_length = self.params["smoothingLength"]
        mins = self.gridpoints[: self.number_of_gridpoints].min(axis=0) - 3.0 * smoothing_length
        maxs = self.gridpoints[: self.number_of_gridpoints].max(axis=0) + 3.0 * smoothing_length
        particleIndices = pick_indices(input_buffers[P_POS_KEY], mins, maxs)
        contactIndices = pick_indices(input_buffers[C_POS_KEY], mins, maxs)
        n_particles, n_contacts = self._set_particle_buffer_sizes(particleIndices.size, contactIndices.size)

        position_pad_value = maxs.max() + 100 * smoothing_length  # Important for to pad outside grid for correctness.
        for key, buffer in input_buffers.items():
            if key in [P_POS_KEY, P_VEL_KEY, P_DISP_KEY, P_MASS_KEY]:
                input_buffers[key] = self._buffer_pad(
                    buffer[particleIndices], n_particles, pad_value=position_pad_value if key == P_POS_KEY else 0.0
                )
            elif key in [
                C_FORCE_KEY,
                C_POS_KEY,
                C_NORMAL_KEY,
                C_TANGENT_U_KEY,
                C_TANGENT_V_KEY,
            ]:
                input_buffers[key] = self._buffer_pad(
                    buffer[contactIndices], n_contacts, pad_value=position_pad_value if key == C_POS_KEY else 0.0
                )
        return input_buffers

    def _validate_input_buffers(self, input_buffers):
        missing_keys = self.REQUIRED_KEYS - input_buffers.keys()
        input_buffers = {k: input_buffers[k] for k in self.REQUIRED_KEYS if k in input_buffers}
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        return input_buffers

    def _set_particle_buffer_sizes(self, num_particles, num_contacts):
        """
        Responsible for setting the number of particles and contacts that are stored in the buffer. Strictly increasing to avoid recompilation / cache problems.
        """

        def update_size(count, current_size, print_label=""):
            new_size = int(ceil(count / self.STANDARD_PARTICLE_BUFFER_SIZE) * self.STANDARD_PARTICLE_BUFFER_SIZE)
            if new_size > current_size:
                size = new_size
                if self.debug_prints_on:
                    print(f"Increasing {print_label} buffer size to {new_size}")
            else:
                size = current_size

            return size

        self.particle_buffer_size = update_size(num_particles, self.particle_buffer_size, "particle")
        self.contact_buffer_size = update_size(num_contacts, self.contact_buffer_size, "contact")

        return self.particle_buffer_size, self.contact_buffer_size

    @staticmethod
    def _buffer_pad(buffer, size, pad_value=0.0):
        """
        Adds a zero padding along axis=0, result is (size, -1). Using pad_value=None does not assign any value to the padded entries, useful for performance.
        """
        n = buffer.shape[0]
        dtype = jax.dtypes.canonicalize_dtype(buffer.dtype)
        out = np.empty((size, *buffer.shape[1:]), dtype=dtype)
        out[:n] = buffer
        if pad_value is not None:
            out[n:] = pad_value
        return out
