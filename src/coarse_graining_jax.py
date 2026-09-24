# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a GPU-accelerated implementation of the coarse graining method described in
#              the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library JAX.
#
# 2025-12-04: First public version, programmed by Petter Persson.

import os
from functools import partial
from math import pi, sqrt, ceil
from time import perf_counter

import jax
import jax.numpy as jnp
from scipy.spatial import KDTree

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

PARTICLE_PACKING_DENSITY = 0.75
CONTACTS_PER_PARTICLE = 8


@partial(jax.jit, static_argnames=["N"])
def filterParticles(x, p, v, u, m, smoothingLength, N):
    """
    Apply a filter to keep the 'N' particles closest to x, that are within 3 * smoothingLength of x.
    CALL SEQUENCE: pos, vel, disp, mass = filterParticles(x, p, v, u, m, smoothingLength, N)
    INPUTS:
        x: array of gridpoints, size ng x 3, ng = number of gridpoints.
        p: array of particle positions, size np x 3, np = number of particles.
        v: array of particle velocities, size np x 3.
        u: array of particle displacements, size np x 3.
        m: particle masses, size np.
        smoothingLength: kernel smoothing length.
        N: number of particles to include in the result.
    OUTPUTS:
        pos: filtered particle positions.
        vel: filtered particle velocities.
        disp: filtered particle displacements.
        mass: filtered particle masses.
    """
    negDistance2, idx = jax.lax.top_k(jnp.negative(jnp.sum(jnp.square(jnp.subtract(x, p)), axis=1)), N)
    isValid = jnp.absolute(negDistance2) < 9 * smoothingLength * smoothingLength
    isValid = isValid.astype(jnp.uint8)
    return (
        jnp.multiply(isValid[:, None], p[idx, :]),
        jnp.multiply(isValid[:, None], v[idx, :]),
        jnp.multiply(isValid[:, None], u[idx, :]),
        jnp.multiply(isValid, m[idx]),
        isValid,
    )


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

    return (
        massDensity,
        momentumDensity,
        velocity,
        displacement,
        granularTemperature,
        pressure,
        vonMisesStress,
        stressTensor,
        strainTensor,
        rateOfStrainTensor,
    )


# Scan body compatible with jax.lax.scan
@jax.jit
def coarse_graining_body(carry, x):
    """
    Body function for jax.lax.scan function. See official documentation for jax.lax.scan.
    INPUTS:
        carry: Expects a tuple of dictionaries (particle_data, contact_data, precomputed_params) containing all particle data and pre-computed constants. Some of the
            keys are defined in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
        x: array of gridpoints, size ng x 3.
    OUTPUTS:
        carry: same as input.
        fields: a dictionary containing all the computed fields. Keys are defined
                in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """

    particle_data, contact_data, precomputed_params = carry

    # Create a vectorized map over the input for the gridpoints x.
    cg_vmap = jax.vmap(
        coarseGrainingFieldsAtPosition,
        in_axes=(0, 0, 0, None),
    )

    # Call coarse graining calculation to get all fields at x.
    fields = cg_vmap(x, particle_data, contact_data, precomputed_params)

    return carry, {
        F_MASS_DENSITY_KEY: fields[0],
        F_MOM_DENSITY_KEY: fields[1],
        F_VEL_KEY: fields[2],
        F_DISP_KEY: fields[3],
        F_GRANULAR_TEMP_KEY: fields[4],
        F_PRESSURE_KEY: fields[5],
        F_VON_MISES_KEY: fields[6],
        F_STRESS_KEY: fields[7],
        F_STRAIN_KEY: fields[8],
        F_RATE_OF_STRAIN_KEY: fields[9],
    }


@jax.jit
def coarse_graining_batched(points_b, pidx_b, cidx_b, particle_data, contact_data, precomputed_params):
    """
    Runs coarse_graining_body over batches of gridpoints with jax.lax.scan, gathering the
    neighbour data of each batch on the device.
    INPUTS:
        points_b: gridpoints, size (num_batches, batch_size, 3).
        pidx_b: particle neighbour indices, size (num_batches, batch_size, kp).
        cidx_b: contact neighbour indices, size (num_batches, batch_size, kc).
        particle_data, contact_data: dicts with the full (unbatched) buffers.
        precomputed_params: dict with precomputed parameters.
    OUTPUTS:
        fields: dict of fields, each of size (num_batches, batch_size, ...).
    """

    def scan_body(carry, batch):
        x, bp, bc = batch
        batch_particle_data = {key: buf[bp] for key, buf in particle_data.items()} | {
            "validParticles": bp < particle_data[P_POS_KEY].shape[0]
        }
        batch_contact_data = {key: buf[bc] for key, buf in contact_data.items()} | {
            "validContacts": bc < contact_data[C_POS_KEY].shape[0]
        }
        batch_carry = (batch_particle_data, batch_contact_data, precomputed_params)
        _, fields = coarse_graining_body(batch_carry, x)
        return carry, fields

    _, fields = jax.lax.scan(scan_body, None, (points_b, pidx_b, cidx_b))
    return fields


def coarseGrainingFields(gridPoints, args, batch_size=1000):
    """
    Computes the coarse graining fields at all gridpoints. For general documentation,
    consult "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: fields = coarseGrainingFields(gridPoints, args, batch_size=1000)
    INPUTS:
        gridPoints: array of gridpoint coordinates (x, y, z), size ng x 3.
        args: dictionary containing all required particle buffers and the parameters, smoothing length and particle diameter.
            expected dictionary keys are defined "src/listeners/coarse_graining_calculation/coarse_graining_constants". The two parameters
            are expected to be "smoothingLength", and "particleDiamter".
        batch_size: Optional argument specifying how larges batches that passed through vmap. Too large value leads to out of memory error,
            and too small limits computational speed (at least on GPU).
    OUTPUTS:
        fields: a dictionary containing all the computed fields. Keys are defined
            in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """

    # To determine the number of particles to include when approximating the cutoff |x| > 3*R (2-norm). rho * (large sphere / small sphere) -> rho * (3R)^3 / (r^3).
    num_cutoff_particles = ceil(
        PARTICLE_PACKING_DENSITY * ((3 * args["smoothingLength"]) ** 3) / ((0.5 * args["particleDiameter"]) ** 3)
    )

    # To determine the number of contacts to include when approximating the cutoff |x| > R (1-norm). rho * (box_volume / small_sphere_volume) -> rho * (8R)^3 / ((4/3)*pi*r^3).
    num_cutoff_contacts = CONTACTS_PER_PARTICLE * ceil(
        PARTICLE_PACKING_DENSITY
        * (8 * (args["smoothingLength"] ** 3))
        / ((4 / 3) * pi * ((0.5 * args["particleDiameter"]) ** 3))
    )

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
    particle_tree = KDTree(args[P_POS_KEY])
    contact_tree = KDTree(args[C_POS_KEY])
    print("Build Trees CPU time:", perf_counter() - start)
    start = perf_counter()
    _, pidx = particle_tree.query(
        gridPoints, k=num_cutoff_particles, distance_upper_bound=3 * args["smoothingLength"], p=2, workers=-1
    )
    _, cidx = contact_tree.query(
        gridPoints, k=num_cutoff_contacts, distance_upper_bound=args["smoothingLength"], p=jnp.inf, workers=-1
    )
    print("Tree query CPU time:", perf_counter() - start)

    gridPoints = jnp.asarray(gridPoints)
    pidx = jnp.asarray(pidx, dtype=jnp.int32)
    cidx = jnp.asarray(cidx, dtype=jnp.int32)

    def run_batches(start_idx, end_idx, nb):
        """Scans over nb equally sized batches of the gridpoints start_idx:end_idx."""
        return coarse_graining_batched(
            gridPoints[start_idx:end_idx].reshape(nb, -1, 3),
            pidx[start_idx:end_idx].reshape(nb, -1, pidx.shape[1]),
            cidx[start_idx:end_idx].reshape(nb, -1, cidx.shape[1]),
            particle_data,
            contact_data,
            precomputed_params,
        )

    # To perform a batched scan of coarse graining calculations over gridpoints.
    ng = gridPoints.shape[0]
    nb = ng // batch_size
    ng_batched = nb * batch_size
    if nb > 0:
        result_dict = run_batches(0, ng_batched, nb)
    else:
        result_dict = run_batches(0, ng, 1)

    if nb == 0 or ng_batched == ng:
        result_dict_rem = {k: jnp.array([]) for k in result_dict}
    else:  # To handle remainder batch
        result_dict_rem = run_batches(ng_batched, ng, 1)

    cg_result = {}
    for key, arr in result_dict.items():
        arr_rem = result_dict_rem[key]
        if key in [
            F_MASS_DENSITY_KEY,
            F_GRANULAR_TEMP_KEY,
            F_PRESSURE_KEY,
            F_VON_MISES_KEY,
        ]:
            cg_result[key] = jnp.concatenate((arr.ravel(), arr_rem.ravel()), axis=0).reshape(-1)
        elif key in [F_MOM_DENSITY_KEY, F_VEL_KEY, F_DISP_KEY]:
            cg_result[key] = jnp.concatenate((arr.ravel(), arr_rem.ravel()), axis=0).reshape(-1, 3)
        elif key in [F_STRESS_KEY, F_STRAIN_KEY, F_RATE_OF_STRAIN_KEY]:
            cg_result[key] = jnp.concatenate((arr.ravel(), arr_rem.ravel()), axis=0).reshape(-1, 3, 3)

    return cg_result
