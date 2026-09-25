# Author: Petter Persson
# Email: petter.p.persson@gmail.com
# Description: This file contains a GPU-accelerated implementation of the coarse graining method described in
#              the document "./coarse_graining_documentation/Stress and strain in pseudo-particle solids.pdf".
#              The implementation uses the python library NVIDIA Warp. It is a drop-in alternative to
#              coarseGrainingFields in "coarse_graining_jax.py": same inputs, same output keys, JAX arrays out.
#
#              One thread per gridpoint walks its actual neighbours through on-device hash grids, so there is
#              no fixed neighbour count k, no padding and no CPU KDTree. The particle sums are done in two passes:
#              pass 1 gives mass density, momentum, displacement and kinetic stress; pass 2 uses the resulting
#              mean velocity/displacement for the granular temperature and the deformation gradients, using
#                  F = sum_i M_i (u_i - u_mean) (x) d_i / rho,
#              which equals the double sum in eq. 12 but avoids cancellation in float32.

from math import pi, sqrt

import jax
import numpy as np
import warp as wp

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

# Hash table size per axis. Points are hashed into it modulo the dimension, so it need not cover the domain;
# collisions only add candidates that are rejected by the distance test.
HASH_GRID_DIM = 128

_hash_grids = {}  # (device, name) -> wp.HashGrid, reused between calls


@wp.struct
class ParticleData:
    mass: wp.array(dtype=float)
    pos: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    disp: wp.array(dtype=wp.vec3)


@wp.struct
class ContactData:
    force: wp.array(dtype=wp.vec3)
    pos: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    tangent_u: wp.array(dtype=wp.vec3)
    tangent_v: wp.array(dtype=wp.vec3)


@wp.struct
class PrecomputedParams:
    gaussian_kernel_factor: float
    gaussian_scale: float
    heaviside_scale: float
    smoothing_length: float
    particle_diameter: float


@wp.kernel
def contact_force_to_global_frame(
    cf: wp.array(dtype=wp.vec3),
    cn: wp.array(dtype=wp.vec3),
    ctu: wp.array(dtype=wp.vec3),
    ctv: wp.array(dtype=wp.vec3),
    cf_global: wp.array(dtype=wp.vec3),
):
    """
    Helper to get the global frame contact forces, f = f0*n + f1*tu + f2*tv.
    INPUTS:
        cf: array of local contact forces, size nc x 3.
        cn: array of contact normals, size nc x 3.
        ctu: array of contact tangent vectors, size nc x 3.
        ctv: array of contact tangent vectors, size nc x 3.
        cf_global: array to be filled with global frame contact forces, size nc x 3.
    """
    tid = wp.tid()
    f = cf[tid]
    cf_global[tid] = f[0] * cn[tid] + f[1] * ctu[tid] + f[2] * ctv[tid]


@wp.kernel
def coarseGrainingKernel(
    particle_grid: wp.uint64,
    contact_grid: wp.uint64,
    gridpoints: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    cp: wp.array(dtype=wp.vec3),
    cf_global: wp.array(dtype=wp.vec3),
    cn: wp.array(dtype=wp.vec3),
    gaussianKernelFactor: float,
    gaussianScale: float,
    heavisideScale: float,
    smoothingLength: float,
    particleDiameter: float,
    massDensity: wp.array(dtype=float),
    momentumDensity: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    displacement: wp.array(dtype=wp.vec3),
    granularTemperature: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
    vonMisesStress: wp.array(dtype=float),
    stressTensor: wp.array(dtype=wp.mat33),
    strainTensor: wp.array(dtype=wp.mat33),
    rateOfStrainTensor: wp.array(dtype=wp.mat33),
):
    """
    Computes all coarse graining fields at gridpoint gridpoints[tid]. For general documentation,
    consult "Stress and strain in pseudo-particle solids.pdf".
    """
    tid = wp.tid()
    x = gridpoints[tid]

    particleCutoff = 3.0 * smoothingLength
    particleCutoff2 = particleCutoff * particleCutoff

    # Pass 1 over particles: mass density, momentum, mass weighted displacement and kinetic stress.
    rho = float(0.0)
    mom = wp.vec3(0.0, 0.0, 0.0)
    mu = wp.vec3(0.0, 0.0, 0.0)
    kineticStress = wp.mat33(0.0)
    query = wp.hash_grid_query(particle_grid, x, particleCutoff)
    i = int(0)
    while wp.hash_grid_query_next(query, i):
        r = x - p[i]
        r2 = wp.dot(r, r)
        if r2 < particleCutoff2:
            M = m[i] * gaussianScale * wp.exp(gaussianKernelFactor * r2)
            rho += M
            mom += M * v[i]
            mu += M * u[i]
            kineticStress -= M * wp.outer(v[i], v[i])

    # Contacts inside the box |x - cp|_inf <= R. The hash query is spherical, so query the circumscribed sphere.
    contactSum = wp.mat33(0.0)
    query = wp.hash_grid_query(contact_grid, x, 1.7320508 * smoothingLength)
    c = int(0)
    while wp.hash_grid_query_next(query, c):
        r = x - cp[c]
        if wp.abs(r[0]) <= smoothingLength and wp.abs(r[1]) <= smoothingLength and wp.abs(r[2]) <= smoothingLength:
            contactSum += wp.outer(cf_global[c], cn[c])

    stress = kineticStress - (heavisideScale * particleDiameter) * contactSum
    press = -wp.trace(stress) / 3.0

    # von mises stress = sqrt(3.0/2.0*(stress_ij * stress_ij - 3*pressure ** 2)), clamped against round-off below zero
    vonMises = wp.sqrt(wp.max(1.5 * (wp.ddot(stress, stress) - 3.0 * press * press), 0.0))

    massDensity[tid] = rho
    momentumDensity[tid] = mom
    pressure[tid] = press
    vonMisesStress[tid] = vonMises
    stressTensor[tid] = stress

    # No particles within the cutoff: the mass weighted fields are undefined (NaN, as in the JAX implementation),
    # and pass 2 is skipped.
    if rho == 0.0:
        velocity[tid] = wp.vec3(wp.nan)
        displacement[tid] = wp.vec3(wp.nan)
        granularTemperature[tid] = wp.nan
        strainTensor[tid] = wp.mat33(wp.nan)
        rateOfStrainTensor[tid] = wp.mat33(wp.nan)
        return

    vel = mom / rho
    disp = mu / rho

    # Pass 2 over particles: granular temperature and deformation gradients, relative to the mean fields.
    # The kernel gradient is d_i = 2 * gaussianKernelFactor * (x - p_i); the constant factor is applied after the loop.
    temperature = float(0.0)
    gradV = wp.mat33(0.0)
    gradU = wp.mat33(0.0)
    query = wp.hash_grid_query(particle_grid, x, particleCutoff)
    i = int(0)
    while wp.hash_grid_query_next(query, i):
        r = x - p[i]
        r2 = wp.dot(r, r)
        if r2 < particleCutoff2:
            phi = gaussianScale * wp.exp(gaussianKernelFactor * r2)
            M = m[i] * phi
            dv = v[i] - vel
            temperature += phi * wp.dot(dv, dv)
            gradV += M * wp.outer(dv, r)
            gradU += M * wp.outer(u[i] - disp, r)
    gradScale = 2.0 * gaussianKernelFactor / rho
    gradV = gradScale * gradV
    gradU = gradScale * gradU

    velocity[tid] = vel
    displacement[tid] = disp
    granularTemperature[tid] = temperature
    strainTensor[tid] = 0.5 * (gradU + wp.transpose(gradU))
    rateOfStrainTensor[tid] = 0.5 * (gradV + wp.transpose(gradV))


def _to_warp(a, dtype, device):
    """Wraps a JAX array without copying, or copies a numpy array to the device, as float32."""
    if isinstance(a, jax.Array):
        return wp.from_jax(a.astype(np.float32), dtype=dtype)
    return wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=dtype, device=device)


def _hash_grid(device, name):
    key = (str(device), name)
    if key not in _hash_grids:
        _hash_grids[key] = wp.HashGrid(HASH_GRID_DIM, HASH_GRID_DIM, HASH_GRID_DIM, device=device)
    return _hash_grids[key]


def coarseGrainingFields(gridPoints, args):
    """
    Computes the coarse graining fields at all gridpoints. For general documentation,
    consult "Stress and strain in pseudo-particle solids.pdf".
    CALL SEQUENCE: fields = coarseGrainingFields(gridPoints, args)
    INPUTS:
        gridPoints: array of gridpoint coordinates (x, y, z), size ng x 3.
        args: dictionary containing all required particle buffers and the parameters, smoothing length and particle diameter.
            expected dictionary keys are defined "src/listeners/coarse_graining_calculation/coarse_graining_constants". The two parameters
            are expected to be "smoothingLength", and "particleDiameter".
        batch_size: Ignored, kept for call compatibility with the JAX implementation. No batching is needed since
            memory use is only the input and output buffers.
        device: Optional Warp device, e.g. "cuda:0". Defaults to the device of a JAX input array, else Warp's preferred device.
    OUTPUTS:
        fields: a dictionary containing all the computed fields as JAX arrays (float32). Keys are defined
            in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """

    device = wp.get_device(wp.get_preferred_device())

    # Pre-compute some constants
    R = args["smoothingLength"]
    gaussianScale = 1.0 / ((sqrt(2.0 * pi) * R) ** 3)
    gaussianKernelFactor = -0.5 / (R * R)
    heavisideScale = 1.0 / ((2.0 * R) ** 3)

    x = _to_warp(gridPoints, wp.vec3, device)
    p = _to_warp(args[P_POS_KEY], wp.vec3, device)
    v = _to_warp(args[P_VEL_KEY], wp.vec3, device)
    u = _to_warp(args[P_DISP_KEY], wp.vec3, device)
    m = _to_warp(args[P_MASS_KEY].reshape(-1), float, device)
    cp = _to_warp(args[C_POS_KEY], wp.vec3, device)
    cf = _to_warp(args[C_FORCE_KEY], wp.vec3, device)
    cn = _to_warp(args[C_NORMAL_KEY], wp.vec3, device)
    ctu = _to_warp(args[C_TANGENT_U_KEY], wp.vec3, device)
    ctv = _to_warp(args[C_TANGENT_V_KEY], wp.vec3, device)

    nc = cp.shape[0]
    cf_global = wp.empty(nc, dtype=wp.vec3, device=device)
    wp.launch(contact_force_to_global_frame, dim=nc, inputs=[cf, cn, ctu, ctv], outputs=[cf_global], device=device)

    # Cell sizes match the query radii, so each query visits the 3x3x3 block of cells around x.
    particle_grid = _hash_grid(device, "particles")
    particle_grid.build(p, 3.0 * R)
    contact_grid = _hash_grid(device, "contacts")
    contact_grid.build(cp, sqrt(3.0) * R)

    ng = x.shape[0]
    outputs = {
        F_MASS_DENSITY_KEY: wp.empty(ng, dtype=float, device=device),
        F_MOM_DENSITY_KEY: wp.empty(ng, dtype=wp.vec3, device=device),
        F_VEL_KEY: wp.empty(ng, dtype=wp.vec3, device=device),
        F_DISP_KEY: wp.empty(ng, dtype=wp.vec3, device=device),
        F_GRANULAR_TEMP_KEY: wp.empty(ng, dtype=float, device=device),
        F_PRESSURE_KEY: wp.empty(ng, dtype=float, device=device),
        F_VON_MISES_KEY: wp.empty(ng, dtype=float, device=device),
        F_STRESS_KEY: wp.empty(ng, dtype=wp.mat33, device=device),
        F_STRAIN_KEY: wp.empty(ng, dtype=wp.mat33, device=device),
        F_RATE_OF_STRAIN_KEY: wp.empty(ng, dtype=wp.mat33, device=device),
    }

    wp.launch(
        coarseGrainingKernel,
        dim=ng,
        inputs=[
            particle_grid.id,
            contact_grid.id,
            x,
            p,
            v,
            u,
            m,
            cp,
            cf_global,
            cn,
            gaussianKernelFactor,
            gaussianScale,
            heavisideScale,
            R,
            args["particleDiameter"],
        ],
        outputs=list(outputs.values()),
        device=device,
    )

    # to_jax is zero-copy on CUDA; vec3/mat33 arrays come out as (ng, 3) and (ng, 3, 3).
    if device.is_cuda:
        return {key: wp.to_jax(arr) for key, arr in outputs.items()}
    return {key: jax.numpy.asarray(arr.numpy()) for key, arr in outputs.items()}
