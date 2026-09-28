# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a GPU-accelerated implementation of the coarse graining method described in
#              the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library JAX.
#
# 2025-12-04: First public version, programmed by Petter Persson.

from functools import partial
from math import pi, sqrt, ceil
from time import perf_counter

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

PARTICLE_PACKING_DENSITY = 0.75
CONTACTS_PER_PARTICLE = 8


@jax.jit
def computeGaussianKernel(factor, scale, x, p):
    """
    Computes the gaussian kernel phi(x, p), returns phi(x,p) = scale*exp(factor*(abs(x-p)^2)).
    CALL SEQUENCE: phi = computeGaussianKernel(factor, scale, x, p)
    INPUTS:
        factor: Pre-computed factor in the exponential.
        scale: Pre computed normalization constant.
        x: a gridpoint coordinate (x,y,z), size (3,).
        p: array of particle positions, size np x 3
    OUTPUTS:
        phi: array of size (np,) containing the results for each particle.
    """
    return jnp.multiply(
        scale,
        jnp.exp(jnp.multiply(factor, jnp.sum(jnp.square(jnp.subtract(x, p)), axis=1))),
    )


@jax.jit
def computeGranularTemperature(particleVelocity, velocity, kernel):
    """
    Computes the granular temperature field from the particle velocities and
    the coarse graining velocity field, at a single gridpoint. See eq. 4 in "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: T = computeGranularTemperature(particleVelocity, velocity, kernel)
    INPUTS:
        particleVelocity: array of particle velocities, size np x 3.
        velocity: coarse graining velocity field at a single gridpoint, size (3,)
        kernel: array of gaussian kernel values, size (np,)
    OUTPUTS:
        T: The granular temperature field at a single gridpoint, scalar value.
    """
    return jnp.dot(
        jnp.sum(
            jnp.square(jnp.subtract(particleVelocity, velocity)),
            axis=1,
        ),
        kernel,
    )


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
def computeContactStress(heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn, ctu, ctv):
    """
    Computes the contact stress tensor according to eq. 22 in "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: sigma = computeContactStress(heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn, ctu, ctv):
    INPUTS:
        heavisideScale: Pre-computed scaling constant for the Heaviside kernel.
        particleDiameter: Mean particle diameter for all particles.
        smoothingLength: Kernel smoothing length.
        x: a gridpoint coordinate (x,y,z), size (3,).
        cf: array of contact forces in a local frame, size nc x 3.
        cp: array of contact positions, size nc x 3.
        cn: array of contact normal vectors, size nc x 3.
        ctu: array of contact tangent vectors, size nc x 3.
        ctv: array of contact tangent vectors, size nc x 3.
    OUTPUTS:
        sigma: The contact stress tensor at x, size 3 x 3.
    """
    return jnp.multiply(
        -heavisideScale,
        jnp.einsum(
            "ij,ik->jk",
            jnp.multiply(
                jnp.add(
                    jnp.add(
                        jnp.einsum("i,ij->ij", cf[:, 0], cn),
                        jnp.einsum("i,ij->ij", cf[:, 1], ctu),
                    ),
                    jnp.einsum("i,ij->ij", cf[:, 2], ctv),
                ),
                jnp.all(jnp.absolute(x - cp) <= smoothingLength, axis=1)[:, None],
            ),
            jnp.multiply(particleDiameter, cn),
        ),
    )


@jax.jit  # genericVectorField is either displacement or velocity
def computeDeformationGradient(M, x, p, massDensity, genericVectorField, gaussianKernelFactor):
    """
    Computes the deformation gradient according to eq. 12 in "Stress and strain in pseudo-particle solids.pdf".
    CALL_SEQUENCE: F = computeDeformationGradient(M, x, p, massDensity, genericVectorField, gaussianKernelFactor)
    INPUTS:
        M: array containing particle mass * phi(x,p), size (np,).
        x: a gridpoint coordinate (x,y,z), size (3,).
        p: array of particle positions, size np x 3.
        massDensity: mass density field at gridpoint x, scalar.
        genericVectorField: array of a generic particle vector field, size np x 3.
        gaussianKernelFactor: Pre-computed scaling constant in the gaussian kernel exponential.
    OUTPUTS:
        F: The deformation gradient tensor, size 3 x 3.
    """
    d = jnp.multiply(2.0 * gaussianKernelFactor, x - p)  # (np, 3)
    return jnp.divide(
        jnp.subtract(
            jnp.einsum("i,j,ik,il->kl", M, M, genericVectorField, d),
            jnp.einsum("i,j,jk,il->kl", M, M, genericVectorField, d),
        ),
        jnp.multiply(massDensity, massDensity),
    )


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

    cf = contact_data[C_FORCE_KEY]
    cp = contact_data[C_POS_KEY]
    cn = contact_data[C_NORMAL_KEY]
    ctu = contact_data[C_TANGENT_U_KEY]
    ctv = contact_data[C_TANGENT_V_KEY]
    validContacts = contact_data["validContacts"]

    gaussianKernelFactor = precomputed_params["gaussianKernelFactor"]
    gaussianScale = precomputed_params["gaussianScale"]
    heavisideScale = precomputed_params["heavisideScale"]
    smoothingLength = precomputed_params["smoothingLength"]
    particleDiameter = precomputed_params["particleDiameter"]

    # To mask out invalid particles and contacts, those are set to zero, and this propagates thorough all calculations.
    kernel = computeGaussianKernel(gaussianKernelFactor, gaussianScale, x, p) * validParticles  # (np,)
    cf = cf * validContacts[:, jnp.newaxis]

    # Contract over particles to obtain mass density and momentum density fields
    M = jnp.multiply(m, kernel)  # (np,)
    massDensity = jnp.sum(M)  # jnp.einsum("i->", M)
    momentumDensity = jnp.dot(M, v)  # jnp.einsum("i,ij->j", M, v)
    velocity = jnp.divide(momentumDensity, massDensity)
    granularTemperature = computeGranularTemperature(v, velocity, kernel)

    # stress tensor
    stressTensor = jnp.add(
        computeKineticStress(M, v),
        computeContactStress(heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn, ctu, ctv),
    )

    pressure = jnp.multiply(-0.333333333333, jnp.trace(stressTensor))

    # von mises stress = sqrt(3.0/2.0*(stress_ij * stress_ij - 3*pressure ** 2)), this expression has been simplified
    vonMisesStress = jnp.sqrt(
        jnp.multiply(
            1.5,
            jnp.subtract(
                jnp.einsum("jk,jk->", stressTensor, stressTensor),
                jnp.multiply(3.0, jnp.square(pressure)),
            ),
        )
    )

    # displacement field
    # jnp.einsum("i,ij->j", M, u)
    displacement = jnp.divide(jnp.dot(M, u), massDensity)

    # Rate of strain tensor is a double contraction over particles
    deform_grad = computeDeformationGradient(M, x, p, massDensity, v, gaussianKernelFactor)
    rateOfStrainTensor = jnp.multiply(0.5, jnp.add(deform_grad, deform_grad.T))

    # Strain tensor
    deform_grad = computeDeformationGradient(M, x, p, massDensity, u, gaussianKernelFactor)
    strainTensor = jnp.multiply(0.5, jnp.add(deform_grad, deform_grad.T))

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


def coarseGrainingFields(gridpoints, gridlimits, args, batch_size=1000):
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

    def get_num_cutoff(V_cell, V_particle):
        V_cutoff_grid = 27*V_cell
        return ceil(PARTICLE_PACKING_DENSITY * V_cutoff_grid / V_particle)

    cell_size_particles = 3*args["smoothingLength"]
    cell_size_contacts = args["smoothingLength"]

    # To determine the number of particles to include when approximating the cutoff |x| > 3*R (2-norm). rho * (box_volume / small sphere) -> rho * (3R)^3 / (r^3).
    V_cell = (3*args["smoothingLength"])**3
    V_particle = (4/3)*pi*((0.5 * args["particleDiameter"]) ** 3)
    num_cutoff_particles = get_num_cutoff(V_cell, V_particle)

    # To determine the number of contacts to include when approximating the cutoff |x| > R (1-norm). rho * (box_volume / small_sphere_volume) -> rho * (8R)^3 / ((4/3)*pi*r^3).
    V_cell = args["smoothingLength"]**3
    num_cutoff_contacts = CONTACTS_PER_PARTICLE * get_num_cutoff(V_cell, V_particle)

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
    contact_data = {key: args[key] for key in (C_FORCE_KEY, C_POS_KEY, C_NORMAL_KEY, C_TANGENT_U_KEY, C_TANGENT_V_KEY)}
    precomputed_params = {
        **constants,
        "smoothingLength": args["smoothingLength"],
        "particleDiameter": args["particleDiameter"],
    }

    start = perf_counter()
    pidx, is_overflow_p = query_spatial_grid(build_spatial_grid(gridlimits, args[P_POS_KEY], cell_size_particles), gridpoints, max_candidates=num_cutoff_particles, ord=2, batch_size=batch_size)
    cidx, is_overflow_c = query_spatial_grid(build_spatial_grid(gridlimits, args[C_POS_KEY], cell_size_contacts), gridpoints, max_candidates=num_cutoff_contacts, ord=jnp.inf, batch_size=batch_size)
    assert not (is_overflow_p.any() or is_overflow_c.any())
    print("Spatial grid build and query time:", perf_counter() - start)


    gridpoints = jnp.asarray(gridpoints)
    pidx = jnp.asarray(pidx, dtype=jnp.int32)
    cidx = jnp.asarray(cidx, dtype=jnp.int32)

    # To perform a batched map of coarse graining calculations over gridpoints.
    fields = coarse_graining_mapped(gridpoints, pidx, cidx, particle_data, contact_data, precomputed_params, batch_size=batch_size)
    return {key: np.asarray(arr) for key, arr in fields.items()}
