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
    global_force: wp.array(dtype=wp.vec3)
    pos: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)


@wp.struct
class PrecomputedParams:
    gaussian_kernel_factor: float
    gaussian_scale: float
    heaviside_scale: float
    smoothing_length: float
    particle_diameter: float
    particle_cutoff: float
    particle_cutoff_sq: float


@wp.struct
class CGFields:
    mass_density: wp.array(dtype=float)
    momentum_density: wp.array(dtype=wp.vec3)
    velocity: wp.array(dtype=wp.vec3)
    displacement: wp.array(dtype=wp.vec3)
    granular_temperature: wp.array(dtype=float)
    pressure: wp.array(dtype=float)
    von_mises_stress: wp.array(dtype=float)
    stress_tensor: wp.array(dtype=wp.mat33)
    strain_tensor: wp.array(dtype=wp.mat33)
    rate_of_strain_tensor: wp.array(dtype=wp.mat33)


@wp.func
def gaussian_kernel(r2: float, kernel_factor: float, scale: float):
    return scale * wp.exp(kernel_factor * r2)


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
    particle_hash_grid: wp.uint64,
    contact_hash_grid: wp.uint64,
    gridpoints: wp.array(dtype=wp.vec3),
    particle_data: ParticleData,
    contact_data: ContactData,
    precomputed_params: PrecomputedParams,
    cg_fields: CGFields,
):
    """
    Computes all coarse graining fields at gridpoint gridpoints[tid].
    INPUTS:
        particle_hash_grid: Warp hash grid for particles.
        contact_hash_grid: Warp hash grid for contacts.
        gridpoints: Warp array of gridpoints, size ng x 3.
        particle_data: ParticleData struct.
        contact_data: ContactData struct.
        precomputed_params: PrecomputedParams struct.
        cg_fields: CGFields struct. Outputs are written to this struct.
    """

    tid = wp.tid()
    x = gridpoints[tid]

    # To parse input data before calculations start
    R = precomputed_params.smoothing_length
    particleCutoff = precomputed_params.particle_cutoff
    particleCutoff2 = precomputed_params.particle_cutoff_sq
    gaussianScale = precomputed_params.gaussian_scale
    gaussianKernelFactor = precomputed_params.gaussian_kernel_factor
    pos = particle_data.pos
    vel = particle_data.vel
    u = particle_data.disp
    mass = particle_data.mass
    cpos = contact_data.pos
    cf = contact_data.global_force
    cn = contact_data.normal

    # To calculate the mass density, momentum density, mass weighted displacement and kinetic stress.
    # Requires one pass over neighbour particles
    rho = float(0.0)
    mom = wp.vec3(0.0, 0.0, 0.0)
    mu = wp.vec3(0.0, 0.0, 0.0)
    kineticStress = wp.mat33(0.0)
    query = wp.hash_grid_query(particle_hash_grid, x, particleCutoff)
    p = int(0)
    while wp.hash_grid_query_next(query, p):
        r = x - pos[p]
        r2 = wp.dot(r, r)
        if r2 < particleCutoff2:
            kernel = gaussian_kernel(r2, gaussianKernelFactor, gaussianScale)
            M = mass[p] * kernel
            rho += mass[p] * kernel
            mom += M * vel[p]
            mu += M * u[p]
            kineticStress -= M * wp.outer(vel[p], vel[p])

    # To calculate the contact stress from contacts inside the box |x - cp|_inf <= R.
    # The hash query is spherical, so query the circumscribed sphere and filter out contacts outside the box.
    # Box side is L = 2R => radius of circumscribed sphere is 0.5*sqrt(3.0)*L = sqrt(3.0)*R
    # sqrt(3.0) = 1.7320508075688772
    contact_stress = wp.mat33(0.0)
    query = wp.hash_grid_query(contact_hash_grid, x, float(1.7320508075688772) * R)
    c = int(0)
    while wp.hash_grid_query_next(query, c):
        r = x - cpos[c]
        if wp.abs(r[0]) <= R and wp.abs(r[1]) <= R and wp.abs(r[2]) <= R:
            contact_stress += wp.outer(cf[c], cn[c])
    contact_stress *= -precomputed_params.heaviside_scale * precomputed_params.particle_diameter

    stress = kineticStress + contact_stress
    pressure = -wp.trace(stress) / 3.0

    # von mises stress is clamped against round-off below zero
    vonMises = wp.sqrt(wp.max(1.5 * (wp.ddot(stress, stress) - 3.0 * pressure * pressure), 0.0))

    cg_fields.mass_density[tid] = rho
    cg_fields.momentum_density[tid] = mom
    cg_fields.pressure[tid] = pressure
    cg_fields.von_mises_stress[tid] = vonMises
    cg_fields.stress_tensor[tid] = stress

    # To break early if there are no particles within the cutoff, i.e mass density is zero.
    if rho == 0.0:
        cg_fields.velocity[tid] = wp.vec3(wp.nan)
        cg_fields.displacement[tid] = wp.vec3(wp.nan)
        cg_fields.granular_temperature[tid] = wp.nan
        cg_fields.strain_tensor[tid] = wp.mat33(wp.nan)
        cg_fields.rate_of_strain_tensor[tid] = wp.mat33(wp.nan)
        return

    velocity = mom / rho
    displacement = mu / rho

    # To calculate remaining fields now that we have all required data.
    temperature = float(0.0)
    gradV = wp.mat33(0.0)
    gradU = wp.mat33(0.0)
    query = wp.hash_grid_query(particle_hash_grid, x, particleCutoff)
    p = int(0)
    while wp.hash_grid_query_next(query, p):
        r = x - pos[p]
        r2 = wp.dot(r, r)
        if r2 < particleCutoff2:
            kernel = gaussian_kernel(r2, gaussianKernelFactor, gaussianScale)
            M = mass[p] * kernel
            dv = vel[p] - velocity
            temperature += kernel * wp.dot(dv, dv)
            gradV += M * wp.outer(dv, r)
            gradU += M * wp.outer(u[p] - displacement, r)
    gradScale = 2.0 * gaussianKernelFactor / rho
    gradV = gradScale * gradV
    gradU = gradScale * gradU

    cg_fields.velocity[tid] = velocity
    cg_fields.displacement[tid] = displacement
    cg_fields.granular_temperature[tid] = temperature
    cg_fields.strain_tensor[tid] = 0.5 * (gradU + wp.transpose(gradU))
    cg_fields.rate_of_strain_tensor[tid] = 0.5 * (gradV + wp.transpose(gradV))


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
    OUTPUTS:
        fields: a dictionary containing all the computed fields as JAX arrays (float32). Keys are defined
            in "src/listeners/coarse_graining_calculation/coarse_graining_constants".
    """

    device = wp.get_device(wp.get_preferred_device())

    # Pre-compute some constants
    R = args["smoothingLength"]
    precomputed_params = PrecomputedParams()
    precomputed_params.gaussian_kernel_factor = -0.5 / (R * R)
    precomputed_params.gaussian_scale = 1.0 / ((sqrt(2.0 * pi) * R) ** 3)
    precomputed_params.heaviside_scale = 1.0 / ((2.0 * R) ** 3)
    precomputed_params.smoothing_length = R
    precomputed_params.particle_diameter = args["particleDiameter"]
    precomputed_params.particle_cutoff = 3.0 * R
    precomputed_params.particle_cutoff_sq = 9.0 * R * R

    x = _to_warp(gridPoints, wp.vec3, device)

    particle_data = ParticleData()
    particle_data.pos = _to_warp(args[P_POS_KEY], wp.vec3, device)
    particle_data.vel = _to_warp(args[P_VEL_KEY], wp.vec3, device)
    particle_data.disp = _to_warp(args[P_DISP_KEY], wp.vec3, device)
    particle_data.mass = _to_warp(args[P_MASS_KEY].reshape(-1), float, device)

    contact_data = ContactData()
    contact_data.pos = _to_warp(args[C_POS_KEY], wp.vec3, device)
    contact_data.normal = _to_warp(args[C_NORMAL_KEY], wp.vec3, device)
    nc = contact_data.pos.shape[0]
    contact_data.global_force = wp.empty(nc, dtype=wp.vec3, device=device)
    wp.launch(
        contact_force_to_global_frame,
        dim=nc,
        inputs=[
            _to_warp(args[C_FORCE_KEY], wp.vec3, device),
            contact_data.normal,
            _to_warp(args[C_TANGENT_U_KEY], wp.vec3, device),
            _to_warp(args[C_TANGENT_V_KEY], wp.vec3, device),
        ],
        outputs=[contact_data.global_force],
        device=device,
    )

    # Cell sizes match the query radii, so each query visits the 3x3x3 block of cells around x.
    particle_hash_grid = _hash_grid(device, "particles")
    particle_hash_grid.build(particle_data.pos, precomputed_params.particle_cutoff)
    contact_hash_grid = _hash_grid(device, "contacts")
    contact_hash_grid.build(contact_data.pos, sqrt(3.0) * R)

    ng = x.shape[0]
    cg_fields = CGFields()
    cg_fields.mass_density = wp.empty(ng, dtype=float, device=device)
    cg_fields.momentum_density = wp.empty(ng, dtype=wp.vec3, device=device)
    cg_fields.velocity = wp.empty(ng, dtype=wp.vec3, device=device)
    cg_fields.displacement = wp.empty(ng, dtype=wp.vec3, device=device)
    cg_fields.granular_temperature = wp.empty(ng, dtype=float, device=device)
    cg_fields.pressure = wp.empty(ng, dtype=float, device=device)
    cg_fields.von_mises_stress = wp.empty(ng, dtype=float, device=device)
    cg_fields.stress_tensor = wp.empty(ng, dtype=wp.mat33, device=device)
    cg_fields.strain_tensor = wp.empty(ng, dtype=wp.mat33, device=device)
    cg_fields.rate_of_strain_tensor = wp.empty(ng, dtype=wp.mat33, device=device)

    wp.launch(
        coarseGrainingKernel,
        dim=ng,
        inputs=[
            particle_hash_grid.id,
            contact_hash_grid.id,
            x,
            particle_data,
            contact_data,
            precomputed_params,
            cg_fields,
        ],
        device=device,
    )

    outputs = {
        F_MASS_DENSITY_KEY: cg_fields.mass_density,
        F_MOM_DENSITY_KEY: cg_fields.momentum_density,
        F_VEL_KEY: cg_fields.velocity,
        F_DISP_KEY: cg_fields.displacement,
        F_GRANULAR_TEMP_KEY: cg_fields.granular_temperature,
        F_PRESSURE_KEY: cg_fields.pressure,
        F_VON_MISES_KEY: cg_fields.von_mises_stress,
        F_STRESS_KEY: cg_fields.stress_tensor,
        F_STRAIN_KEY: cg_fields.strain_tensor,
        F_RATE_OF_STRAIN_KEY: cg_fields.rate_of_strain_tensor,
    }

    # to_jax is zero-copy on CUDA; vec3/mat33 arrays come out as (ng, 3) and (ng, 3, 3).
    if device.is_cuda:
        return {key: wp.to_jax(arr) for key, arr in outputs.items()}
    return {key: jax.numpy.asarray(arr.numpy()) for key, arr in outputs.items()}
