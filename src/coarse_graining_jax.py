# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a GPU-accelerated implementation of the coarse graining method described in
#              the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library JAX.
#
# 2025-12-04: First public version, programmed by Petter Persson.

from functools import partial
from math import pi, sqrt, ceil

import jax
import jax.numpy as jnp
import numpy as np

from .coarse_graining_constants import (
    C_FORCE_KEY,
    C_NORMAL_KEY,
    C_POS_KEY,
    C_TANGENT_U_KEY,
    C_TANGENT_V_KEY,
    F_DISP_KEY,
    F_GRANULAR_TEMP_KEY,
    F_MASS_DENSITY_KEY,
    F_MOM_DENSITY_KEY,
    F_PRESSURE_KEY,
    F_RATE_OF_STRAIN_KEY,
    F_STRAIN_KEY,
    F_STRESS_KEY,
    F_VEL_KEY,
    F_VON_MISES_KEY,
    P_DISP_KEY,
    P_MASS_KEY,
    P_POS_KEY,
    P_VEL_KEY,
)
from .utils.spatial_grid_jax import build_spatial_grid, query_spatial_grid

PARTICLE_PACKING_DENSITY = 0.80
CONTACTS_PER_PARTICLE = 8


@jax.jit
def computeGaussianKernel(factor, scale, r):
    """
    Computes the gaussian kernel phi(x, p), returns phi(x,p) = scale*exp(factor*(abs(x-p)^2)).
    CALL SEQUENCE: phi = computeGaussianKernel(factor, scale, r)
    INPUTS:
        factor: Pre-computed factor in the exponential.
        scale: Pre computed normalization constant.
        r: array of separations x - p between a gridpoint x and the particle positions p, size np x 3.
    OUTPUTS:
        phi: array of size (np,) containing the results for each particle.
    """
    return scale * jnp.exp(factor * jnp.sum(r * r, axis=1))


@jax.jit
def computeGranularTemperature(dv, kernel):
    """
    Computes the granular temperature field from the particle velocities and
    the coarse graining velocity field, at a single gridpoint. See eq. 4 in "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: T = computeGranularTemperature(dv, kernel)
    INPUTS:
        dv: array of particle velocities minus the coarse graining velocity at the gridpoint, size np x 3.
        kernel: array of gaussian kernel values, size (np,)
    OUTPUTS:
        T: The granular temperature field at a single gridpoint, scalar value.
    """
    return jnp.dot(kernel, jnp.sum(dv * dv, axis=1))


@jax.jit
def computeKineticStress(M, particleVelocities):
    """
    Computes the kinetic stress tensor according to eq. 5 in "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: sigma = computeKineticStress(M, particleVelocities)
    INPUTS:
        M: array containing particle mass * phi(x,p), size (np,).
        particleVelocities: array of particle velocities, size np x 3
    OUTPUTS:
        sigma: The kinetic stress tensor at a single gridpoint, size 3 x 3.
    """
    return jnp.multiply(-1.0, jnp.einsum("i,ij,ik->jk", M, particleVelocities, particleVelocities))


@jax.jit
def computeGlobalContactForce(cf, cn, ctu, ctv):
    """
    Rotates the contact forces from the local contact frame (cn, ctu, ctv) to the global frame.
    CALL SEQUENCE: cf_global = computeGlobalContactForce(cf, cn, ctu, ctv)
    INPUTS:
        cf: array of contact forces in a local frame, size nc x 3.
        cn: array of contact normal vectors, size nc x 3.
        ctu: array of contact tangent vectors, size nc x 3.
        ctv: array of contact tangent vectors, size nc x 3.
    OUTPUTS:
        cf_global: array of contact forces in the global frame, size nc x 3.
    """
    return cf[:, 0:1] * cn + cf[:, 1:2] * ctu + cf[:, 2:3] * ctv


@jax.jit
def computeContactStress(heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn):
    """
    Computes the contact stress tensor according to eq. 22 in "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: sigma = computeContactStress(heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn):
    INPUTS:
        heavisideScale: Pre-computed scaling constant for the Heaviside kernel.
        particleDiameter: Mean particle diameter for all particles.
        smoothingLength: Kernel smoothing length.
        x: a gridpoint coordinate (x,y,z), size (3,).
        cf: array of contact forces in the global frame, size nc x 3.
        cp: array of contact positions, size nc x 3.
        cn: array of contact normal vectors, size nc x 3.
    OUTPUTS:
        sigma: The contact stress tensor at x, size 3 x 3.
    """
    inside = jnp.all(jnp.abs(x - cp) <= smoothingLength, axis=1)  # (nc,)
    return -heavisideScale * particleDiameter * jnp.einsum("ij,ik->jk", cf * inside[:, None], cn)


@jax.jit
def computeDeformationGradient(M, r, dw, massDensity, gaussianKernelFactor):
    """
    Computes the deformation gradient according to eq. 12 in "Stress and strain in pseudo-particle solids.pdf".
    Eq. 12 is rewritten in the same centred form as in the warp implementation,
    F = sum_i (2 * gaussianKernelFactor / rho) * M_i * dw_i r_i^T, which avoids the double sum over particles
    and the cancellation between its two terms. The minus sign is hidden in gaussianKernelFactor = -0.5 / R^2.
    CALL_SEQUENCE: F = computeDeformationGradient(M, r, dw, massDensity, gaussianKernelFactor)
    INPUTS:
        M: array containing particle mass * phi(x,p), size (np,).
        r: array of separations x - p between a gridpoint x and the particle positions p, size np x 3.
        dw: array of a particle vector field (velocity or displacement) minus its coarse graining field at x, size np x 3.
        massDensity: mass density field at gridpoint x, scalar.
        gaussianKernelFactor: Pre-computed scaling constant in the gaussian kernel exponential.
    OUTPUTS:
        F: The deformation gradient tensor, size 3 x 3.
    """
    return (2.0 * gaussianKernelFactor / massDensity) * jnp.einsum("i,ik,il->kl", M, dw, r)

@jax.jit
def coarseGrainingFieldsAtPosition(
    x: jnp.ndarray,
    particle_data: dict,
    contact_data: dict,
    precomputed_params: dict,
):
    """
    Computes the coarse graining fields at a single gridpoint x. For general documentation,
    consult "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: fields = coarseGrainingFieldsAtPosition(x, particle_data, contact_data, precomputed_params)
    INPUTS:
        x: a gridpoint coordinate (x,y,z), size (3,).
        particle_data: dict with particle data buffers.
        contact_data: dict with contact data buffers.
        precomputed_params: dict with precomputed parameters.
    OUTPUTS:
        fields: tuple of arrays containing the computed coarse graining fields.
    """

    p = particle_data[P_POS_KEY]
    v = particle_data[P_VEL_KEY]
    u = particle_data[P_DISP_KEY]
    m = particle_data[P_MASS_KEY]
    validParticles = particle_data["validParticles"]

    cf_global = contact_data[C_FORCE_KEY]  # in the global frame
    cp = contact_data[C_POS_KEY]
    cn = contact_data[C_NORMAL_KEY]
    validContacts = contact_data["validContacts"]

    gaussianKernelFactor = precomputed_params["gaussianKernelFactor"]
    gaussianScale = precomputed_params["gaussianScale"]
    heavisideScale = precomputed_params["heavisideScale"]
    smoothingLength = precomputed_params["smoothingLength"]
    particleDiameter = precomputed_params["particleDiameter"]

    # To mask out invalid particles and contacts, those are set to zero, and this propagates thorough all calculations.
    r = x - p  # (np, 3), shared by the kernel and the gradients
    kernel = computeGaussianKernel(gaussianKernelFactor, gaussianScale, r) * validParticles  # (np,)
    cf_global = cf_global * validContacts[:, jnp.newaxis]

    # Contract over particles to obtain mass density, momentum density, velocity and displacement fields
    M = m * kernel  # (np,)
    massDensity = jnp.sum(M)
    momentumDensity = M @ v
    velocity = momentumDensity / massDensity
    displacement = (M @ u) / massDensity

    # stress tensor
    stressTensor = computeKineticStress(M, v) + computeContactStress(heavisideScale, particleDiameter, smoothingLength, x, cf_global, cp, cn)
    pressure = -jnp.trace(stressTensor) / 3.0

    # von mises stress = sqrt(3.0/2.0*(stress_ij * stress_ij - 3*pressure ** 2)), clamped against round-off below zero
    vonMisesStress = jnp.sqrt(jnp.maximum(1.5 * (jnp.sum(stressTensor * stressTensor) - 3.0 * pressure * pressure), 0.0))

    # Deviations from the coarse graining velocity and displacement, shared by the granular temperature and the gradients.
    # If the mass density is zero, velocity and displacement are NaN, and so are the fields computed from dv and du.
    dv = v - velocity
    du = u - displacement
    granularTemperature = computeGranularTemperature(dv, kernel)

    gradV = computeDeformationGradient(M, r, dv, massDensity, gaussianKernelFactor)
    rateOfStrainTensor = 0.5 * (gradV + gradV.T)

    gradU = computeDeformationGradient(M, r, du, massDensity, gaussianKernelFactor)
    strainTensor = 0.5 * (gradU + gradU.T)

    return {
        F_MASS_DENSITY_KEY: massDensity,
        F_MOM_DENSITY_KEY: momentumDensity,
        F_VEL_KEY: velocity,
        F_DISP_KEY: displacement,
        F_GRANULAR_TEMP_KEY: granularTemperature,
        F_PRESSURE_KEY: pressure,
        F_VON_MISES_KEY: vonMisesStress,
        F_STRESS_KEY: stressTensor,
        F_STRAIN_KEY: strainTensor,
        F_RATE_OF_STRAIN_KEY: rateOfStrainTensor,
    }


@partial(jax.jit, static_argnames=("batch_size",))
def coarse_graining_mapped(gridpoints, pidx, cidx, particle_data, contact_data, precomputed_params, batch_size):
    """
    Computes the coarse graining fields at every gridpoint with jax.lax.map, which splits the
    gridpoints into batches of batch_size, vmaps over each batch and handles the remainder.
    Each gridpoint gathers its own neighbour data on the device.
    INPUTS:
        gridpoints: gridpoints, size (ng, 3).
        pidx: particle neighbour indices, size (ng, kp).
        cidx: contact neighbour indices, size (ng, kc).
        particle_data, contact_data: dicts with the full (unbatched) buffers.
        precomputed_params: dict with precomputed parameters.
        batch_size: number of gridpoints computed in parallel.
    OUTPUTS:
        fields: dict of fields, each of size (ng, ...). Keys are defined
            in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """
    num_particles = particle_data[P_POS_KEY].shape[0]
    num_contacts = contact_data[C_POS_KEY].shape[0]

    def fields_at_point(args):
        x, ip, ic = args  # (3,), (kp,), (kc,)
        input_particles_data = {key: buf[ip] for key, buf in particle_data.items()} | {"validParticles": ip < num_particles}
        input_contact_data = {key: buf[ic] for key, buf in contact_data.items()} | {"validContacts": ic < num_contacts}
        fields = coarseGrainingFieldsAtPosition(x, input_particles_data, input_contact_data, precomputed_params)
        return fields
    
    return jax.lax.map(fields_at_point, (gridpoints, pidx, cidx), batch_size=batch_size)


def coarseGrainingFields(gridpoints, gridlimits, args, batch_size=1000, debug_prints_on=False):
    """
    Computes the coarse graining fields at all gridpoints. For general documentation,
    consult "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: fields = coarseGrainingFields(gridPoints, args, batch_size=1000)
    INPUTS:
        gridPoints: array of gridpoint coordinates (x, y, z), size ng x 3.
        gridlimits: array=[xmin, ymin, zmin, xmax, ymax, zmax]. User is responsible for ensuring that the gridlimits are correct.
        args: dictionary containing all required particle buffers and the parameters, smoothing length and particle diameter.
            expected dictionary keys are defined "src/listeners/coarse_graining_calculation/coarse_graining_constants". The two parameters
            are expected to be "smoothingLength", and "particleDiamter".
        batch_size: Optional argument specifying how larges batches that passed through vmap. Too large value leads to out of memory error,
            and too small limits computational speed (at least on GPU).
    OUTPUTS:
        fields: a dictionary containing all the computed fields. Keys are defined
            in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """

    def get_max_count(V_region, V_particle):
        """Maximum number of particles in a region of volume V_region, given by the packing density."""
        return ceil(PARTICLE_PACKING_DENSITY * V_region / V_particle)

    cell_size_particles = 3*args["smoothingLength"]
    cell_size_contacts = args["smoothingLength"]
    V_particle = (4/3)*pi*((0.5 * args["particleDiameter"]) ** 3)

    # Particles: the candidates are the particles in the 27 cells around a gridpoint, and the neighbours are those within the cutoff |x - p| <= 3*R (2-norm).
    max_candidates_particles = get_max_count(27 * cell_size_particles**3, V_particle)
    max_neighbours_particles = get_max_count((4/3) * pi * cell_size_particles**3, V_particle)

    # Contacts: the candidates are the contacts in the 27 cells around a gridpoint, and the neighbours are those within the cutoff |x - c| <= R (inf-norm).
    max_candidates_contacts = CONTACTS_PER_PARTICLE * get_max_count(27 * cell_size_contacts**3, V_particle)
    max_neighbours_contacts = CONTACTS_PER_PARTICLE * get_max_count((2 * cell_size_contacts)**3, V_particle)

    # Pre-compute some constants
    R = args["smoothingLength"]
    constants = {
        "gaussianScale": 1.0 / ((sqrt(2.0 * pi) * R) ** 3),
        "gaussianKernelFactor": -0.5 / (R * R),
        "heavisideScale": 1.0 / ((2.0 * R) ** 3),
    }

    # To prepare the input for the coarse graining calculation
    particle_data = {
        P_POS_KEY: args[P_POS_KEY],
        P_VEL_KEY: args[P_VEL_KEY],
        P_DISP_KEY: args[P_DISP_KEY],
        P_MASS_KEY: args[P_MASS_KEY].flatten(),
    }
    particle_data = {k: jnp.asarray(v) for k, v in particle_data.items()}
    contact_data = {
        C_FORCE_KEY: args[C_FORCE_KEY],
        C_POS_KEY: args[C_POS_KEY],
        C_NORMAL_KEY: args[C_NORMAL_KEY],
        C_TANGENT_U_KEY: args[C_TANGENT_U_KEY],
        C_TANGENT_V_KEY: args[C_TANGENT_V_KEY],
    }
    contact_data = {k: jnp.asarray(v) for k, v in contact_data.items()}
    contact_data[C_FORCE_KEY] = computeGlobalContactForce(contact_data[C_FORCE_KEY], contact_data[C_NORMAL_KEY], contact_data[C_TANGENT_U_KEY], contact_data[C_TANGENT_V_KEY])
    precomputed_params = {
        **constants,
        "smoothingLength": args["smoothingLength"],
        "particleDiameter": args["particleDiameter"],
    }

    pidx, is_overflow_p = query_spatial_grid(build_spatial_grid(gridlimits, particle_data[P_POS_KEY], cell_size_particles), gridpoints, max_candidates=max_candidates_particles, max_neighbours=max_neighbours_particles, ord=2, batch_size=batch_size)
    cidx, is_overflow_c = query_spatial_grid(build_spatial_grid(gridlimits, contact_data[C_POS_KEY], cell_size_contacts), gridpoints, max_candidates=max_candidates_contacts, max_neighbours=max_neighbours_contacts, ord=jnp.inf, batch_size=batch_size)
    assert not (is_overflow_p.any() or is_overflow_c.any())

    gridpoints = jnp.asarray(gridpoints)
    pidx = jnp.asarray(pidx, dtype=jnp.int32)
    cidx = jnp.asarray(cidx, dtype=jnp.int32)

    # To perform a batched map of coarse graining calculations over gridpoints.
    fields = coarse_graining_mapped(gridpoints, pidx, cidx, particle_data, contact_data, precomputed_params, batch_size=batch_size)
    return {key: np.asarray(arr) for key, arr in fields.items()}
