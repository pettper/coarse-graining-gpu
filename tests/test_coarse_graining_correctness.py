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


def gaussian_kernel(x, p):
    scale = (1 / (np.sqrt(2 * np.pi) * SMOOTHING_LENGTH)) ** 3
    exp_factor = -0.5 / (SMOOTHING_LENGTH * SMOOTHING_LENGTH)
    return scale * np.exp(exp_factor * (np.linalg.norm(x - p, ord=2, axis=1) ** 2))


class TestCoarseGrainingCorrectness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gridpoints = np.array([[0.0, 0.0, 0.0]])
        particle_positions = np.array(
            [[PARTICLE_DIAMETER, 0.0, -PARTICLE_DIAMETER], [PARTICLE_DIAMETER, 2 * PARTICLE_DIAMETER, 0.0]]
        )
        contact_positions = np.array(
            [[PARTICLE_DIAMETER, 0.0, PARTICLE_DIAMETER], [0.0, PARTICLE_DIAMETER, -2 * PARTICLE_DIAMETER]]
        )
        cls.buffers = {
            P_POS_KEY: particle_positions,
            P_VEL_KEY: np.ones_like(particle_positions),
            P_DISP_KEY: np.ones_like(particle_positions),
            P_MASS_KEY: np.ones(particle_positions.shape[0]),
            C_POS_KEY: contact_positions,
            C_FORCE_KEY: np.ones_like(contact_positions),
            C_NORMAL_KEY: np.array([[0.0, 0.0, 1.0] for x in range(contact_positions.shape[0])]),
            C_TANGENT_U_KEY: np.array([[1.0, 0.0, 0.0] for x in range(contact_positions.shape[0])]),
            C_TANGENT_V_KEY: np.array([[0.0, 1.0, 0.0] for x in range(contact_positions.shape[0])]),
        }
        cls.cg = CoarseGrainingMain(
            cls.gridpoints,
            smoothing_length=SMOOTHING_LENGTH,
            particle_diameter=PARTICLE_DIAMETER,
            cg_batch_size=10,
            debug_prints_on=False,
        )

    def test_mass_density_is_correct(self):
        fields = self.cg.calculate(self.buffers)
        mass_density = fields[F_MASS_DENSITY_KEY]
        correct_mass_density = np.sum(
            self.buffers[P_MASS_KEY] * gaussian_kernel(self.gridpoints, self.buffers[P_POS_KEY]), keepdims=True
        )
        np.testing.assert_array_almost_equal(mass_density, correct_mass_density, decimal=6)


if __name__ == "__main__":
    unittest.main()
