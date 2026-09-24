import time
import numpy as np
import jax
from scipy import stats

from coarse_graining_gpu.coarse_graining_main import CoarseGrainingMain
from coarse_graining_gpu.src.coarse_graining_constants import (
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

rng = np.random.default_rng(seed=42)


def make_test_data(num_particles, num_contacts, num_gridpoints, particle_diameter=0.01, packing_density=0.75):
    Vp = (4 / 3) * np.pi * (0.5 * particle_diameter) ** 3
    L = ((num_particles * Vp) / packing_density) ** (1 / 3)

    p = rng.uniform(low=-0.5 * L, high=0.5 * L, size=(num_particles, 3))
    v = rng.standard_normal(size=(num_particles, 3))
    u = rng.standard_normal(size=(num_particles, 3))
    m = np.ones(num_particles)
    cp = rng.uniform(low=-0.5 * L, high=0.5 * L, size=(num_contacts, 3))
    cf = rng.standard_normal(size=(num_contacts, 3))
    cn = rng.standard_normal(size=(num_contacts, 3))
    ctu = rng.standard_normal(size=(num_contacts, 3))
    ctv = rng.standard_normal(size=(num_contacts, 3))

    x = np.linspace(-0.5 * L, 0.5 * L, round(num_gridpoints ** (1 / 3)))
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    gridpoints = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    buffers = {
        P_POS_KEY: p,
        P_VEL_KEY: v,
        P_DISP_KEY: u,
        P_MASS_KEY: m,
        C_POS_KEY: cp,
        C_FORCE_KEY: cf,
        C_NORMAL_KEY: cn,
        C_TANGENT_U_KEY: ctu,
        C_TANGENT_V_KEY: ctv,
    }
    return gridpoints, buffers


def time_coarse_graining(cg: CoarseGrainingMain, buffers: dict, runs=5):
    fields = cg.calculate(buffers)  # Do not measure, just to compile.
    jax.block_until_ready(fields)

    times = np.empty(runs)
    for j in range(runs):
        start = time.perf_counter()
        fields = cg.calculate(buffers)
        jax.block_until_ready(fields)
        times[j] = time.perf_counter() - start

    sem = times.std(ddof=1) / np.sqrt(runs)
    ci_half_width = stats.t.ppf(0.975, df=runs - 1) * sem
    return np.mean(times), ci_half_width
