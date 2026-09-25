import argparse

from coarse_graining_gpu.coarse_graining_main import CoarseGrainingMain, GPUBackend
from coarse_graining_gpu.tests.setup_utils import make_test_data, time_coarse_graining

NUM_TEST_RUNS = 10
PACKING_DENSITY = 0.75
PARTICLE_DIAMETER = 0.01
SMOOTHING_LENGTH = 1.5 * PARTICLE_DIAMETER

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, required=False, default="warp")
args = parser.parse_args()

if args.backend == "warp":
    BACKEND = GPUBackend.WARP
elif args.backend == "jax":
    BACKEND = GPUBackend.JAX
else:
    raise NotImplementedError(f"Backend {args.backend} not implemented.")


def run_test(num_particles, num_contacts, num_gridpoints, batch_sizes, title_print=""):
    print(title_print)
    for npart, nc, ng, bs in zip(num_particles, num_contacts, num_gridpoints, batch_sizes):
        gridpoints, buffers = make_test_data(
            npart, nc, ng, particle_diameter=PARTICLE_DIAMETER, packing_density=PACKING_DENSITY
        )
        cg = CoarseGrainingMain(
            gridpoints,
            SMOOTHING_LENGTH,
            PARTICLE_DIAMETER,
            cg_batch_size=bs,
            debug_prints_on=False,
            backend=BACKEND,
        )
        t, ci = time_coarse_graining(cg, buffers, runs=NUM_TEST_RUNS)
        print(
            f"Tests with batch size {bs}, {npart} particles, {nc} contacts, and {ng} gridpoints. Time per step: {t:.6f} +- {ci:.6f} s"
        )


if __name__ == "__main__":
    num_particles = [10000, 100000, 1000000]
    num_contacts = [5 * x for x in num_particles]
    num_gridpoints = [1000] * len(num_particles)
    batch_sizes = [512] * len(num_particles)
    run_test(
        num_particles,
        num_contacts,
        num_gridpoints,
        batch_sizes,
        title_print="TESTING DEPENDENCY ON NUMBER OF PARTICLES",
    )

    num_gridpoints = [100, 1000, 10000, 100000]
    num_particles = [100000] * len(num_gridpoints)
    num_contacts = [5 * x for x in num_particles]
    batch_sizes = [512] * len(num_gridpoints)
    run_test(
        num_particles,
        num_contacts,
        num_gridpoints,
        batch_sizes,
        title_print="\nTESTING DEPENDENCY ON NUMBER OF GRIDPOINTS",
    )

    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 100000]
    num_particles = [100000] * len(batch_sizes)
    num_contacts = [5 * x for x in num_particles]
    num_gridpoints = [100000] * len(batch_sizes)
    run_test(num_particles, num_contacts, num_gridpoints, batch_sizes, title_print="\nTESTING DEPENDENCY ON BATCH SIZE")
