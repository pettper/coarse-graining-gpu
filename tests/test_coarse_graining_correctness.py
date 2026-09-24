import unittest
import numpy as np

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

PARTICLE_DIAMETER = 0.01
SMOOTHING_LENGTH = 1.5 * PARTICLE_DIAMETER
RTOL = 1e-6
ATOL = 1e-5


def gaussian_kernel(x, p):
    dtype = np.result_type(x, p)

    def helper(x, p):
        nonlocal dtype
        scale = dtype.type((1 / (np.sqrt(2 * np.pi) * SMOOTHING_LENGTH)) ** 3)
        exp_factor = dtype.type(-0.5 / (SMOOTHING_LENGTH * SMOOTHING_LENGTH))
        return scale * np.exp(exp_factor * (np.linalg.norm(x - p, ord=2, axis=1) ** 2))

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


class TestCoarseGrainingCorrectness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        dtype = np.float32
        cls.gridpoints = np.array([[0.0, 0.0, 0.0], [0.5 * PARTICLE_DIAMETER for _ in range(3)]], dtype=dtype)
        particle_positions = np.array(
            [
                [PARTICLE_DIAMETER, 0.0, -PARTICLE_DIAMETER],
                [PARTICLE_DIAMETER, 2 * PARTICLE_DIAMETER, 0.0],
                [0.0, 0.0, 0.5 * PARTICLE_DIAMETER],
                [0.0, 3.0 * PARTICLE_DIAMETER, 3.0 * PARTICLE_DIAMETER],
            ],
        )
        contact_positions = np.array(
            [
                [PARTICLE_DIAMETER, 0.0, PARTICLE_DIAMETER],
                [0.0, PARTICLE_DIAMETER, -2 * PARTICLE_DIAMETER],
                [0.5 * PARTICLE_DIAMETER, 0.0, 0.0],
                [3.0 * PARTICLE_DIAMETER, 3.0 * PARTICLE_DIAMETER, 0.0],
            ],
        )

        num_particles = particle_positions.shape[0]
        num_contacts = contact_positions.shape[0]
        data = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        cls.buffers = {
            P_POS_KEY: particle_positions,
            P_VEL_KEY: data,
            P_DISP_KEY: data,
            P_MASS_KEY: np.ones(num_particles),
            C_POS_KEY: contact_positions,
            C_FORCE_KEY: data,
            C_NORMAL_KEY: np.array([[0.0, 0.0, 1.0] for _ in range(num_contacts)]),
            C_TANGENT_U_KEY: np.array([[1.0, 0.0, 0.0] for _ in range(num_contacts)]),
            C_TANGENT_V_KEY: np.array([[0.0, 1.0, 0.0] for _ in range(num_contacts)]),
        }
        cls.buffers = {k: v.astype(dtype) for k, v in cls.buffers.items()}
        cg = CoarseGrainingMain(
            cls.gridpoints,
            smoothing_length=SMOOTHING_LENGTH,
            particle_diameter=PARTICLE_DIAMETER,
            cg_batch_size=10,
            debug_prints_on=False,
        )
        cls.cg_fields = cg.calculate(cls.buffers)

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

    def test_mass_density_is_correct(self):
        # From coarse graining
        mass_density = self.cg_fields[F_MASS_DENSITY_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(mass_density, self.correct_mass_density, rtol=RTOL, atol=ATOL, strict=True)

    def test_momentum_density_is_correct(self):
        # From coarse graining
        momentum_density = self.cg_fields[F_MOM_DENSITY_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(momentum_density, self.correct_momentum_density, rtol=RTOL, atol=ATOL, strict=True)

    def test_velocity_is_correct(self):
        # From coarse graining
        velocity = self.cg_fields[F_VEL_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(velocity, self.correct_velocity, rtol=RTOL, atol=ATOL, strict=True)

    def test_granular_temperature_is_correct(self):
        # From coarse graining
        granular_temperature = self.cg_fields[F_GRANULAR_TEMP_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(
            granular_temperature, self.correct_granular_temperature, rtol=RTOL, atol=ATOL, strict=True
        )

    def test_displacement_is_correct(self):
        # From coarse graining
        displacement = self.cg_fields[F_DISP_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(displacement, self.correct_displacement, rtol=RTOL, atol=ATOL, strict=True)

    def test_stress_tensor_is_correct(self):
        # From coarse graining
        stress_tensor = self.cg_fields[F_STRESS_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(stress_tensor, self.correct_stress_tensor, rtol=RTOL, atol=ATOL, strict=True)

    def test_pressure_is_correct(self):
        # From coarse graining
        pressure = self.cg_fields[F_PRESSURE_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(pressure, self.correct_pressure, rtol=RTOL, atol=ATOL, strict=True)

    def test_von_mises_stress_is_correct(self):
        # From coarse graining
        von_mises_stress = self.cg_fields[F_VON_MISES_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(von_mises_stress, self.correct_von_mises_stress, rtol=RTOL, atol=ATOL, strict=True)

    def test_strain_tensor_is_correct(self):
        # From coarse graining
        strain_tensor = self.cg_fields[F_STRAIN_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(strain_tensor, self.correct_strain_tensor, rtol=RTOL, atol=ATOL, strict=True)

    def test_rate_of_strain_tensor_is_correct(self):
        # From coarse graining
        rate_of_strain_tensor = self.cg_fields[F_RATE_OF_STRAIN_KEY]

        # Compare to manual calculation
        np.testing.assert_allclose(
            rate_of_strain_tensor, self.correct_rate_of_strain_tensor, rtol=RTOL, atol=ATOL, strict=True
        )


if __name__ == "__main__":
    unittest.main()
