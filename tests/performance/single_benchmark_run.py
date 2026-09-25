import argparse

from coarse_graining_gpu.tests.setup_utils import make_test_data, make_test_data2, time_coarse_graining
from coarse_graining_gpu.coarse_graining_main import CoarseGrainingMain, GPUBackend

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, required=False, default="warp")
parser.add_argument("--testdata", type=str, required=False, default="case1")
args = parser.parse_args()

if args.backend == "warp":
    backend = GPUBackend.WARP
elif args.backend == "jax":
    backend = GPUBackend.JAX
else:
    raise NotImplementedError(f"Backend {args.backend} not implemented.")

if args.testdata == "case1":
    make_data_fn = make_test_data
elif args.testdata == "case2":
    make_data_fn = make_test_data2
else:
    raise NotImplementedError(f"Test data {args.testdata} not implemented.")

PARTICLE_DIAMETER = 0.01
SMOOTHING_LENGTH = 1.5 * PARTICLE_DIAMETER

NUM_PARTICLES = 1000000
NUM_CONTACTS = 5 * NUM_PARTICLES
NUM_GRIDPOINTS = 100000
BATCH_SIZE = 1024
gridpoints, buffers = make_data_fn(NUM_PARTICLES, NUM_CONTACTS, NUM_GRIDPOINTS)
cg = CoarseGrainingMain(
    gridpoints,
    smoothing_length=SMOOTHING_LENGTH,
    particle_diameter=PARTICLE_DIAMETER,
    cg_batch_size=BATCH_SIZE,
    debug_prints_on=False,
    backend=backend,
)

t, ci = time_coarse_graining(cg, buffers, runs=5)
print(
    f"Tests with batch size {BATCH_SIZE}, {NUM_PARTICLES} particles, {NUM_CONTACTS} contacts, and {NUM_GRIDPOINTS} gridpoints. Time per step: {t:.6f} +- {ci:.6f} s"
)
