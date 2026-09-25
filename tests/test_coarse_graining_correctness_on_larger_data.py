import unittest
import numpy as np

from coarse_graining_gpu.coarse_graining_main import CoarseGrainingMain, GPUBackend
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
    F_MASS_DENSITY_KEY,
    F_MOM_DENSITY_KEY,
    F_VEL_KEY,
    F_DISP_KEY,
    F_GRANULAR_TEMP_KEY,
    F_STRESS_KEY,
    F_PRESSURE_KEY,
    F_VON_MISES_KEY,
    F_STRAIN_KEY,
    F_RATE_OF_STRAIN_KEY,
)
from coarse_graining_gpu.tests.setup_utils import make_test_data2

PARTICLE_DIAMETER = 0.01
SMOOTHING_LENGTH = 1.5 * PARTICLE_DIAMETER
RTOL = 1e-4
ATOL = 1e-3


def gaussian_kernel(x, p):
    dtype = np.result_type(x, p)

    def helper(x, p):
        nonlocal dtype
        scale = dtype.type((1 / (np.sqrt(2 * np.pi) * SMOOTHING_LENGTH)) ** 3)
        exp_factor = dtype.type(-0.5 / (SMOOTHING_LENGTH * SMOOTHING_LENGTH))
        norm = np.linalg.norm(x - p, ord=2, axis=1)
        isValid = norm <= 3 * SMOOTHING_LENGTH
        return scale * np.exp(exp_factor * (norm**2)) * isValid

    kernel = []
    for j in range(x.shape[0]):
        kernel.append(helper(x[j], p))
    return np.array(kernel)


def gaussian_kernel_gradient(x, p):
    dtype = np.result_type(x, p)
    kernel = gaussian_kernel(x, p)
    scale = -1 / (SMOOTHING_LENGTH * SMOOTHING_LENGTH)
    return dtype.type(scale * (np.einsum("nj,ni->nij", x, kernel) - np.einsum("ij,ni->nij", p, kernel)))


def heaviside_kernel(x, p):
    dtype = np.result_type(x, p)

    def helper(x, p):
        nonlocal dtype
        return ((1 / (2 * SMOOTHING_LENGTH)) ** 3) * (np.all(np.abs(x - p) <= SMOOTHING_LENGTH, axis=1)).astype(dtype)

    kernel = []
    for j in range(x.shape[0]):
        kernel.append(helper(x[j], p))
    return np.array(kernel)


def deformation_gradient(x, p, m, v, mass_density):
    kernel = gaussian_kernel(x, p)
    d_kernel = gaussian_kernel_gradient(x, p)
    term1 = np.einsum("i,j,ik,nil,nj->nkl", m, m, v, d_kernel, kernel)
    term2 = np.einsum("i,j,jk,nil,nj->nkl", m, m, v, d_kernel, kernel)
    return (term1 - term2) / np.square(mass_density)[:, np.newaxis, np.newaxis]


class TestCoarseGrainingCorrectnessLargerData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        dtype = np.float32
        cls.gridpoints, cls.buffers = make_test_data2(500, 2500, 100, PARTICLE_DIAMETER)
        cls.gridpoints = cls.gridpoints.astype(dtype)
        cls.buffers = {k: v.astype(dtype) for k, v in cls.buffers.items()}

        cls.cg_fields = {}
        for backend in GPUBackend:
            cg = CoarseGrainingMain(
                cls.gridpoints,
                smoothing_length=SMOOTHING_LENGTH,
                particle_diameter=PARTICLE_DIAMETER,
                cg_batch_size=32,
                debug_prints_on=False,
                backend=backend,
            )
            cls.cg_fields[backend.name] = cg.calculate(cls.buffers)

        # Correct fields from manual calculation
        kernel = gaussian_kernel(cls.gridpoints, cls.buffers[P_POS_KEY])
        kernel_h = heaviside_kernel(cls.gridpoints, cls.buffers[C_POS_KEY])
        cls.correct_mass_density = np.einsum("i,ni->n", cls.buffers[P_MASS_KEY], kernel)
        p = np.einsum("i,ij->ij", cls.buffers[P_MASS_KEY], cls.buffers[P_VEL_KEY])
        cls.correct_momentum_density = np.einsum("ni,ij->nj", kernel, p)
        cls.correct_velocity = cls.correct_momentum_density / cls.correct_mass_density[:, np.newaxis]

        cls.correct_granular_temperature = np.empty(cls.correct_velocity.shape[0], dtype=dtype)
        for j in range(cls.correct_velocity.shape[0]):
            cls.correct_granular_temperature[j] = np.sum(
                np.square(cls.buffers[P_VEL_KEY] - cls.correct_velocity[j, :]).sum(axis=1) * kernel[j, :]
            )
        mu = np.einsum("i,ij->ij", cls.buffers[P_MASS_KEY], cls.buffers[P_DISP_KEY])
        cls.correct_displacement = (
            np.einsum("ni,ij->nj", kernel, mu).reshape(-1, 3) / cls.correct_mass_density[:, np.newaxis]
        )
        stress_k = -np.einsum(
            "i,ij,ik,ni->njk", cls.buffers[P_MASS_KEY], cls.buffers[P_VEL_KEY], cls.buffers[P_VEL_KEY], kernel
        )
        cartesian_force = (
            cls.buffers[C_FORCE_KEY][:, 0, np.newaxis] * cls.buffers[C_NORMAL_KEY]
            + cls.buffers[C_FORCE_KEY][:, 1, np.newaxis] * cls.buffers[C_TANGENT_U_KEY]
            + cls.buffers[C_FORCE_KEY][:, 2, np.newaxis] * cls.buffers[C_TANGENT_V_KEY]
        )
        branch_vector = PARTICLE_DIAMETER * cls.buffers[C_NORMAL_KEY]
        stress_c = -np.einsum("ij,ik,ni->njk", cartesian_force, branch_vector, kernel_h)
        cls.correct_stress_tensor = stress_k + stress_c
        cls.correct_pressure = -(1 / 3) * np.trace(cls.correct_stress_tensor, axis1=1, axis2=2)
        stress_dev = cls.correct_stress_tensor + np.einsum("n,jk->njk", cls.correct_pressure, np.eye(3, dtype=dtype))
        cls.correct_von_mises_stress = np.sqrt((3 / 2) * np.einsum("njk,njk->n", stress_dev, stress_dev))

        tensors = []
        for field in [cls.buffers[P_DISP_KEY], cls.buffers[P_VEL_KEY]]:
            def_grad = deformation_gradient(
                cls.gridpoints, cls.buffers[P_POS_KEY], cls.buffers[P_MASS_KEY], field, cls.correct_mass_density
            )
            tensors.append(0.5 * (def_grad + def_grad.transpose((0, 2, 1))))
        cls.correct_strain_tensor = tensors[0]
        cls.correct_rate_of_strain_tensor = tensors[1]

    def assert_correct_for_all_backends(self, key, correct):
        for backend, fields in self.cg_fields.items():
            with self.subTest(backend=backend):
                np.testing.assert_allclose(fields[key], correct, rtol=RTOL, atol=ATOL, strict=True)

    def test_mass_density_is_correct(self):
        self.assert_correct_for_all_backends(F_MASS_DENSITY_KEY, self.correct_mass_density)

    def test_momentum_density_is_correct(self):
        self.assert_correct_for_all_backends(F_MOM_DENSITY_KEY, self.correct_momentum_density)

    def test_velocity_is_correct(self):
        self.assert_correct_for_all_backends(F_VEL_KEY, self.correct_velocity)

    def test_granular_temperature_is_correct(self):
        self.assert_correct_for_all_backends(F_GRANULAR_TEMP_KEY, self.correct_granular_temperature)

    def test_displacement_is_correct(self):
        self.assert_correct_for_all_backends(F_DISP_KEY, self.correct_displacement)

    def test_stress_tensor_is_correct(self):
        self.assert_correct_for_all_backends(F_STRESS_KEY, self.correct_stress_tensor)

    def test_pressure_is_correct(self):
        self.assert_correct_for_all_backends(F_PRESSURE_KEY, self.correct_pressure)

    def test_von_mises_stress_is_correct(self):
        self.assert_correct_for_all_backends(F_VON_MISES_KEY, self.correct_von_mises_stress)

    def test_strain_tensor_is_correct(self):
        self.assert_correct_for_all_backends(F_STRAIN_KEY, self.correct_strain_tensor)

    def test_rate_of_strain_tensor_is_correct(self):
        self.assert_correct_for_all_backends(F_RATE_OF_STRAIN_KEY, self.correct_rate_of_strain_tensor)


if __name__ == "__main__":
    unittest.main()
