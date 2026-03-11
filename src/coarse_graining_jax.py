# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a GPU-accelerated implementation of the coarse graining method described in
#              the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library JAX.
#
# 2025-12-04: First public version, programmed by Petter Persson.

import os
from functools import partial
from math import pi, sqrt
from time import perf_counter

import jax
import jax.numpy as jnp

from coarse_graining_gpu import (
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
    negDistance2, idx = jax.lax.top_k(
        jnp.negative(jnp.sum(jnp.square(jnp.subtract(x, p)), axis=1)), N
    )
    isValid = jnp.absolute(negDistance2) < 9 * smoothingLength * smoothingLength
    isValid = isValid.astype(jnp.uint8)
    return (
        jnp.multiply(isValid[:, None], p[idx, :]),
        jnp.multiply(isValid[:, None], v[idx, :]),
        jnp.multiply(isValid[:, None], u[idx, :]),
        jnp.multiply(isValid, m[idx]),
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
    return jnp.multiply(
        -1.0, jnp.einsum("i,ij,ik->jk", M, particleVelocities, particleVelocities)
    )


@jax.jit
def computeContactStress(
    heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn, ctu, ctv
):
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
def computeDeformationGradient(M, x, p, massDensity, genericVectorField, gaussianScale):
    """
    Computes the deformation gradient according to eq. 12 in "Stress and strain in pseudo-particle solids.pdf".
    CALL_SEQUENCE: F = computeDeformationGradient(M, x, p, massDensity, genericVectorField, gaussianScale)
    INPUTS:
        M: array containing particle mass * phi(x,p), size (np,).
        x: a gridpoint coordinate (x,y,z), size (3,).
        p: array of particle positions, size np x 3.
        massDensity: mass density field at gridpoint x, scalar.
        genericVectorField: array of a generic particle vector field, size np x 3.
        gaussianScale: Pre-computed scaling constant for the gaussian kernel.
    OUTPUTS:
        F: The deformation gradient tensor, size 3 x 3.
    """
    d = jnp.multiply(2.0 * gaussianScale, x - p)  # (np, 3)
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
    p: jnp.ndarray,
    v: jnp.ndarray,
    u: jnp.ndarray,
    m: jnp.ndarray,
    cf: jnp.ndarray,
    cp: jnp.ndarray,
    cn: jnp.ndarray,
    ctu: jnp.ndarray,
    ctv: jnp.ndarray,
    gaussianKernelFactor,
    gaussianScale,
    heavisideScale,
    smoothingLength,
    particleDiameter,
):
    """
    Computes the coarse graining fields at a single gridpoint x. For general documentation,
    consult "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: fields = computeGrainingFieldsAtPosition(x, p, v, u, m, cf, cp, cn, ctu, ctv,
        gaussianKernelFactor, gaussianScale, heavisideScale, smoothingLength, particleDiameter)
    INPUTS:
        x: a gridpoint coordinate (x,y,z), size (3,).
        p: array of particle positions, size np x 3.
        v: array of particle velocities, size np x 3.
        u: array of particle displacements, size np x 3.
        m: particle masses, size np.
        cf: array of contact forces in a local frame, size nc x 3.
        cp: array of contact positions, size nc x 3.
        cn: array of contact normal vectors, size nc x 3.
        ctu: array of contact tangent vectors, size nc x 3.
        ctv: array of contact tangent vectors, size nc x 3.
        gaussianKernelFactor: Pre-computed factor in the gaussian kernel exponential.
        gaussianScale: Pre computed normalization constant.
        heavisideScale: Pre computed normalization constant for heaviside kernel.
        smoothingLength: The smoothing length parameter.
        particleDiameter: The mean particle diameter.
    OUTPUTS:
        fields: tuple of arrays containing the computed coarse graining fields.
    """
    # Apply a rough filter to approximate |x| > 3*smoothingLength cutoff for particles
    p, v, u, m = filterParticles(
        x,
        p,
        v,
        u,
        m,
        smoothingLength,
        int(os.environ.get("NUM_CUTOFF_PARTICLES", "1500")),
    )

    kernel = computeGaussianKernel(gaussianKernelFactor, gaussianScale, x, p)  # (np,)

    # Contract over particles to obtain mass density and momentum density fields
    M = jnp.multiply(m, kernel)  # (np,)
    massDensity = jnp.sum(M)  # jnp.einsum("i->", M)
    momentumDensity = jnp.dot(M, v)  # jnp.einsum("i,ij->j", M, v)

    velocity = jnp.divide(momentumDensity, massDensity)
    granularTemperature = computeGranularTemperature(v, velocity, kernel)

    # stress tensor
    stressTensor = jnp.add(
        computeKineticStress(M, v),
        computeContactStress(
            heavisideScale, particleDiameter, smoothingLength, x, cf, cp, cn, ctu, ctv
        ),
    )

    pressure = jnp.multiply(-0.333333333333, jnp.trace(stressTensor))

    # von mises stress = sqrt(3.0/2.0*(stress_ij * stress_ij - 3*pressure ** 2))
    vonMisesStress = jnp.sqrt(
        jnp.multiply(
            1.666666666667,
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
    deform_grad = computeDeformationGradient(M, x, p, massDensity, v, gaussianScale)
    rateOfStrainTensor = jnp.multiply(0.5, jnp.add(deform_grad, deform_grad.T))

    # Strain tensor
    deform_grad = computeDeformationGradient(M, x, p, massDensity, u, gaussianScale)
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
        carry: Expects a dictionary containing all particle data and pre-computed constants. Some of the
            keys are defined in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
        x: array of gridpoints, size ng x 3.
    OUTPUTS:
        carry: same as input.
        fields: a dictionary containing all the computed fields. Keys are defined
                in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """
    # Parse input data.
    p = carry[P_POS_KEY]
    v = carry[P_VEL_KEY]
    u = carry[P_DISP_KEY]
    m = carry[P_MASS_KEY]
    cf = carry[C_FORCE_KEY]
    cp = carry[C_POS_KEY]
    cn = carry[C_NORMAL_KEY]
    ctu = carry[C_TANGENT_U_KEY]
    ctv = carry[C_TANGENT_V_KEY]
    gaussianKernelFactor = carry["gaussianKernelFactor"]
    gaussianScale = carry["gaussianScale"]
    heavisideScale = carry["heavisideScale"]
    smoothingLength = carry["smoothingLength"]
    particleDiameter = carry["particleDiameter"]

    # Create a vectorized map over the input for the gridpoints x.
    cg_vmap = jax.vmap(
        coarseGrainingFieldsAtPosition,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )

    # Call coarse graining calculation to get all fields at x.
    fields = cg_vmap(
        x,
        p,
        v,
        u,
        m,
        cf,
        cp,
        cn,
        ctu,
        ctv,
        gaussianKernelFactor,
        gaussianScale,
        heavisideScale,
        smoothingLength,
        particleDiameter,
    )

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

    assert not int(os.environ.get("NUM_CUTOFF_PARTICLES", "1500")) < (
        0.75
        * ((3 * args["smoothingLength"]) ** 3)
        / ((0.5 * args["particleDiameter"]) ** 3)
    ), (
        f"clipped number of particles {int(os.environ.get('NUM_CUTOFF_PARTICLES', '1500'))}, is likely smaller than actual number included in |x| < 3*smoothingLength. Increase NUM_CUTOFF_PARTICLES or decrease the smoothingLength."
    )

    # Pre-compute some constants
    R = args["smoothingLength"]
    constants = {
        "gaussianScale": 1.0 / ((sqrt(2.0 * pi) * R) ** 3),
        "gaussianKernelFactor": -0.5 / (R * R),
        "heavisideScale": 1.0 / ((2.0 * R) ** 3),
    }
    args = {**args, **constants}
    args[P_MASS_KEY] = args[P_MASS_KEY].flatten()

    # Batches over gridpoints using jax.lax.scan
    x_size = gridPoints.shape[0]
    num_splits = x_size // batch_size
    if num_splits <= 0:
        x_splits = [gridPoints]  # No splitting required
    else:
        x_splits = jnp.split(gridPoints[: (num_splits) * batch_size], num_splits)
        x_splits.append(gridPoints[(num_splits) * batch_size :])

    if len(x_splits) > 1:
        # handle splits of equal size
        _, result_dict = jax.lax.scan(
            coarse_graining_body,
            args,
            jnp.stack(x_splits[:-1]),
        )
        # handle remainder
        _, result_dict_rem = jax.lax.scan(
            coarse_graining_body,
            args,
            jnp.stack(x_splits[-1][None, :, :]),
        )
    else:
        _, result_dict = jax.lax.scan(
            coarse_graining_body,
            args,
            jnp.stack(x_splits[-1][None, :, :]),
        )
        result_dict_rem = {k: jnp.array([]) for k in result_dict.keys()}

    cg_result = {}
    for key, arr in result_dict.items():
        arr_rem = result_dict_rem[key]
        arr_rem.block_until_ready()
        arr.block_until_ready()
        if key in [
            F_MASS_DENSITY_KEY,
            F_GRANULAR_TEMP_KEY,
            F_PRESSURE_KEY,
            F_VON_MISES_KEY,
        ]:
            cg_result[key] = jnp.concatenate(
                (arr.ravel(), arr_rem.ravel()), axis=0
            ).reshape(-1)
        elif key in [F_MOM_DENSITY_KEY, F_VEL_KEY, F_DISP_KEY]:
            cg_result[key] = jnp.concatenate(
                (arr.ravel(), arr_rem.ravel()), axis=0
            ).reshape(-1, 3)
        elif key in [F_STRESS_KEY, F_STRAIN_KEY, F_RATE_OF_STRAIN_KEY]:
            cg_result[key] = jnp.concatenate(
                (arr.ravel(), arr_rem.ravel()), axis=0
            ).reshape(-1, 3, 3)

    return cg_result


def measure(f, repetitions=5):
    # call once to avoid measuring jit-compilation
    f()
    t = []
    for _ in jnp.arange(0, repetitions):
        s = perf_counter()
        f()
        e = perf_counter()
        t.append(e - s)
    t = jnp.array(t)
    return t, jnp.mean(t), jnp.std(t)


# If the file runs as main script, this small test suite is executed.
if __name__ == "__main__":
    np = 500000
    ng = 13000
    nc = 3 * np
    smoothingLength = 0.1

    args = {}
    args["particleMass"] = m = jnp.linspace(0.0, 1000.0, np, dtype=jnp.float32)
    args["particlePosition"] = jnp.linspace(
        0.0, 1000.0, 3 * np, dtype=jnp.float32
    ).reshape(np, 3)
    args["particleVelocity"] = jnp.linspace(
        0.0, 1000.0, 3 * np, dtype=jnp.float32
    ).reshape(np, 3)
    args["particleDisplacement"] = jnp.linspace(
        0.0, 1000.0, 3 * np, dtype=jnp.float32
    ).reshape(np, 3)
    args["localContactForce"] = jnp.linspace(
        0.0, 1000.0, 3 * nc, dtype=jnp.float32
    ).reshape(nc, 3)
    args["contactPosition"] = jnp.linspace(
        0.0, 1000.0, 3 * nc, dtype=jnp.float32
    ).reshape(nc, 3)
    args["contactNormal"] = jnp.linspace(
        0.0, 1000.0, 3 * nc, dtype=jnp.float32
    ).reshape(nc, 3)
    args["contactTangentU"] = jnp.linspace(
        0.0, 1000.0, 3 * nc, dtype=jnp.float32
    ).reshape(nc, 3)
    args["contactTangentV"] = jnp.linspace(
        0.0, 1000.0, 3 * nc, dtype=jnp.float32
    ).reshape(nc, 3)
    args["gaussianScale"] = 1.0 / ((sqrt(2.0 * pi) * smoothingLength) ** 3)
    args["gaussianKernelFactor"] = -0.5 / (smoothingLength * smoothingLength)
    args["heavisideScale"] = 8.0 * smoothingLength**3
    args["smoothingLength"] = smoothingLength
    args["particleDiameter"] = 0.06

    gridPoints = jnp.zeros((ng, 3), dtype=jnp.float32)

    t, mean, std = measure(
        lambda: coarseGrainingFields(gridPoints, args, batch_size=100)
    )
    print(t)
    print(mean)
