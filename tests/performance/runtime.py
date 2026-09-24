import time
import numpy as np
import jax
from scipy import stats

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
)
from coarse_graining_gpu.coarse_graining_main import CoarseGrainingMain

NUM_TEST_RUNS = 10
PACKING_DENSITY = 0.75
PARTICLE_DIAMETER = 0.01
SMOOTHING_LENGTH = 1.5 * PARTICLE_DIAMETER
rng = np.random.default_rng(seed=42)


def make_test_data(num_particles, num_contacts, num_gridpoints):
    Vp = (4 / 3) * np.pi * (0.5 * PARTICLE_DIAMETER) ** 3
    L = ((num_particles * Vp) / PACKING_DENSITY) ** (1 / 3)

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


def run_test(num_particles, num_contacts, num_gridpoints, batch_sizes, title_print=""):

    print(title_print)
    for npart, nc, ng, bs in zip(num_particles, num_contacts, num_gridpoints, batch_sizes):
        gridpoints, buffers = make_test_data(npart, nc, ng)
        cg = CoarseGrainingMain(
            gridpoints,
            SMOOTHING_LENGTH,
            PARTICLE_DIAMETER,
            cg_batch_size=bs,
            debug_prints_on=False,
        )
        t, ci = time_coarse_graining(cg, buffers, runs=NUM_TEST_RUNS)
        print(
            f"Tests with batch size {bs}, {npart} particles, {nc} contacts, and {ng} gridpoints. Time per step: {t:.6f} +- {ci:.6f} s"
        )


num_particles = [10000, 100000, 1000000]
num_contacts = [5 * x for x in num_particles]
num_gridpoints = [1000] * len(num_particles)
batch_sizes = [512] * len(num_particles)
run_test(
    num_particles, num_contacts, num_gridpoints, batch_sizes, title_print="TESTING DEPENDENCY ON NUMBER OF PARTICLES"
)

num_gridpoints = [100, 1000, 10000, 100000]
num_particles = [100000] * len(num_gridpoints)
num_contacts = [5 * x for x in num_particles]
batch_sizes = [512] * len(num_gridpoints)
run_test(
    num_particles, num_contacts, num_gridpoints, batch_sizes, title_print="\nTESTING DEPENDENCY ON NUMBER OF GRIDPOINTS"
)

batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 100000]
num_particles = [100000] * len(batch_sizes)
num_contacts = [5 * x for x in num_particles]
num_gridpoints = [100000] * len(batch_sizes)
run_test(num_particles, num_contacts, num_gridpoints, batch_sizes, title_print="\nTESTING DEPENDENCY ON BATCH SIZE")
