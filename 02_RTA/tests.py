# This file is an executable.
# Test the core code, from the simpest of checks to a full Drude comparison.
#
# If not all tests pass, the code is in all likelihood corrupted.
# Consider if you use compatible packages.
#
# -----------------------
# Part of the SmallAngleScatteringLMO repository
# Subject to the LICENSE terms and conditions
# Written by Roemer Hinlopen, 2022

import core
import unittest
import numpy as np
import time


class TestFermiSurface(unittest.TestCase):
    """ Test that the Fermi surface, energy landscape, velocity are good. """

    def test_energy_value(self):
        """ Simply test this has basic values as programmed initially.

        Also ensure the tb coefficients are periodic the way they should be
        and respond to negative ka / kb values.
        """

        structE = tuple([3, 5])

        # Baseline
        e = core.compute_energy(0, 0, structE)
        self.assertAlmostEqual(e, -2 * structE[0] + -2 * structE[1])

        # BZ size
        e = core.compute_energy(np.pi / core.B, 0, structE)
        self.assertAlmostEqual(e, 2 * structE[0] - 2 * structE[1])

        e = core.compute_energy(np.pi / core.B, np.pi / core.A, structE)
        self.assertAlmostEqual(e, 2 * structE[0] + 2 * structE[1])

        # Negative
        e = core.compute_energy(-np.pi / core.B, -np.pi / core.A, structE)
        self.assertAlmostEqual(e, 2 * structE[0] + 2 * structE[1])

        # Periodic
        e = core.compute_energy(2 * np.pi / core.B, 0, structE)
        self.assertAlmostEqual(e, -2 * structE[0] - 2 * structE[1])

    def test_get_kb(self):
        """ Make sure the Fermi surface can be traced.

        Uses positive and negative tb
        """

        structE = tuple([0.5, 0.05, 0])
        max_ka = np.pi / core.A  # edge BZ
        max_kb = np.pi / core.B  # edge BZ

        # Basic
        # Test that we are really talking Fermi surface
        # close enough that compounding errors won't matter
        kb = core.get_kb(0, structE)
        self.assertLess(kb, max_kb)
        self.assertGreater(kb, 0)  # positive kb branch only.
        e_real = core.compute_energy(kb, 0, structE)
        self.assertAlmostEqual(e_real, 0, delta=1e-10)
        structE2 = tuple([0.5, 0.05, -0.876])
        kb = core.get_kb(0, structE2)
        e_real = core.compute_energy(kb, 0, structE2)
        self.assertAlmostEqual(e_real, -0.876, delta=1e-10)

        # Off center
        kb2 = core.get_kb(max_ka, structE)
        self.assertLess(kb2, max_kb)
        self.assertGreater(kb2, 0)  # positive kb branch only.
        # positive structE[1] means smallest FS at Gamma.
        self.assertGreater(kb2, kb)
        e_real = core.compute_energy(kb2, max_ka, structE)
        self.assertAlmostEqual(e_real, 0, delta=1e-10)
        #   translate to ka=0, towards Gamma from FS
        e_test = core.compute_energy(kb2, 0, structE)
        self.assertLess(e_test, -0.001)

        # Really edge of BZ symmetric
        # And ka==0 symmetric as well
        # No matter energy
        structE3 = tuple([0.5, 0.05, 0.4])
        kb3 = core.get_kb(max_ka * 1.1, structE3)
        kb4 = core.get_kb(max_ka * 0.9, structE3)
        self.assertAlmostEqual(kb3, kb4)

        structE4 = tuple([0.5, 0.05, -0.04])
        kb5 = core.get_kb(max_ka * -0.1, structE4)
        kb6 = core.get_kb(max_ka * 0.1, structE4)
        self.assertAlmostEqual(kb5, kb6)

    def test_kb_time(self):
        """ Benchmark. """

        structE = tuple([0.5, 0.05, 0])
        core.get_kb(0, structE)  # guarantee compiled
        st = time.time()
        iters = 300000
        for _ in range(iters):
            core.get_kb(0, structE)
        print(f'kb calculation takes {(time.time() - st) * 1e6/ iters:.2f}'
              f' micro-s (over {(time.time() - st):.2f} s).')

    def test_velocity(self):
        """ Test the magnitude as well as direction. """

        structE = tuple([1, 0])
        expect_vb = 2 * structE[0] * core.E / core.HBAR * core.B

        # That velocity should exactly match at half filling
        # with no further corrugation.
        kb = core.get_kb(0, structE)
        vb = core.compute_velocity_b(kb, 0, structE)
        self.assertAlmostEqual(vb, expect_vb)

        # And no c component
        va = core.compute_velocity_a(kb, 0, structE)
        self.assertAlmostEqual(va, 0)

        ########
        # Next, test the direction.
        structE = tuple([1, 0.1])

        # Halfway along the BZ vc is maximal
        ka = np.pi / core.A / 2
        kb = core.get_kb(ka, structE)

        # The ratio should be given by t' c sin(ka c)/t b sin(kb b)
        expect_ratio = structE[1] / structE[0]
        expect_ratio *= np.sin(ka * core.A) / np.sin(kb * core.B)
        expect_ratio *= core.A / core.B
        vb = core.compute_velocity_b(kb, ka, structE)
        va = core.compute_velocity_a(kb, ka, structE)
        ratio = va / vb
        self.assertAlmostEqual(ratio, expect_ratio)

    def test_density(self):
        """ Quantitative test for full and fraction filling. """

        # No c corrugation for analytic test.
        # With 1e-10 fractional offset, the density is fraction 1e-5 off
        # because of the quadratic dispersion at the BZ edge.
        #
        # Because this is so accurate, any miss-by-one errors
        # in the Simpson integration may be spotted readily.
        structE = tuple([0.2, 0, 0.4 - 1e-15])
        n = core.compute_density(structE)
        self.assertAlmostEqual(n, 1)

        # No matter the corrugations, these take away as much
        # as they add so zero energy remains half filling.
        structE = tuple([0.2, 0.1, 0])
        n = core.compute_density(structE)
        self.assertAlmostEqual(n, 0.5)

        # A quantitative result in memory away from special points
        # Computed with n=5001
        structE = tuple([0.2, 0.1, -0.03])
        n = core.compute_density(structE)
        self.assertAlmostEqual(n, 0.474337065365204, delta=1e-9)

    def test_time_density(self):
        """ Benchmark """
        structE = tuple([0.2, 0, 0.4 - 1e-15])
        st = time.time()
        iters = 10000
        for _ in range(iters):
            core.compute_density(structE)
        print(f'Density calculation takes {(time.time() - st) * 1e6/ iters:.0f}'
              f' micro-s (over {(time.time() - st):.2f} s).')


class TestCyclotron(unittest.TestCase):
    """ Test the cyclotron orbital motion computations. """

    def test_RK4(self):
        """ Test full BZ time, direction. """

        # No corrugation initially for definite BZ time.
        # This FS has an E minimum at Gamma. Electrons.
        structE = tuple([0.1, 0, 0])
        field = 10

        # Computation
        #   hbar kadot = q vb B  - Newton
        #   BZ = 2pi/C
        #   --> time = 2pi hbar / (q vB B)
        kb = core.get_kb(0, structE)  # same for all ka
        vb = core.compute_velocity_b(kb, 0, structE)  # same for all ka
        self.assertGreater(vb, 0)
        expect_time = 2 * np.pi * core.HBAR / (core.E * vb * field * core.A)

        # Now simulate with RK4 over that period of time
        iters = 1000
        dt = expect_time / iters
        ka = 0
        for _ in range(iters):
            ka = core.RK4(ka, dt, field, structE)
        self.assertAlmostEqual(abs(ka), 2 * np.pi /
                               core.A, delta=1e-8 / core.A)
        self.assertGreater(ka, 0)  # electrons move up

    def test_RK4_convergence(self):
        """ Test convergence for a corrugated FS.
        A more advanced test than straight.
        """

        structE = tuple([-0.2, -0.04, -0.31])
        field = -10

        # So close to the Lifshitz transition at 0.32, kb varies 33% kb
        kb_max = core.get_kb(0, structE)
        kb_min = core.get_kb(np.pi / core.A, structE)
        self.assertAlmostEqual(kb_max / kb_min, 1.336, delta=0.001)

        # Same formula as the uncorrugated, but now average 1/v over the FS.
        kac = np.linspace(-np.pi / core.A, np.pi / core.A, 20000)
        kbb = [core.get_kb(ka, structE) for ka in kac]
        ivb = [1 / core.compute_velocity_b(kb, ka, structE)
               for kb, ka in zip(kbb, kac)]
        expect_time = 2 * np.pi * core.HBAR / (core.E * abs(field) * core.A)
        expect_time *= abs(np.sum(ivb)) / len(ivb)

        # Now simulate with RK4 over that period of time
        iters = 1000
        dt = expect_time / iters
        ka = 0
        for _ in range(iters):
            ka = core.RK4(ka, dt, field, structE)
        self.assertAlmostEqual(abs(ka), 2 * np.pi /
                               core.A, delta=1e-4 / core.A)
        self.assertGreater(ka, 0)  # holes move down, but negative field so up

        # Extra test
        # Should be separate, but all the set-up is here readily available.
        RK4_time = core.time_BZ(field, structE)
        self.assertAlmostEqual(RK4_time, expect_time, delta=expect_time / 1e4)

    def test_timeBZ(self):
        """ The time for ka to move 2pi/c, equivalent of wct=2pi scale. """

        # No corrugation initially for definite BZ time.
        # This FS has an E minimum at Gamma. Electrons.
        structE = tuple([0.1, 0, 0])
        field = 10

        # Computation
        #   hbar kadot = q vb B  - Newton
        #   BZ = 2pi/C
        #   --> time = 2pi hbar / (q vB B)
        kb = core.get_kb(0, structE)  # same for all ka
        vb = core.compute_velocity_b(kb, 0, structE)  # same for all ka
        self.assertGreater(vb, 0)
        expect_time = 2 * np.pi * core.HBAR / (core.E * vb * field * core.A)

        # Now reproduce
        timer = core.time_BZ(field, structE)
        self.assertAlmostEqual(expect_time, timer, delta=expect_time / 1e5)

        # Field
        timer = core.time_BZ(field / 100, structE)
        self.assertAlmostEqual(expect_time * 100, timer,
                               delta=expect_time / 1e3)

        ########
        # Now with corrugation
        # Non-integer fraction to ensure the linear overshoot correction
        # Tested: considerably greater accuracy than 1/100th of dt !
        #
        # Corrugation enough to delay by 52% empirically
        # Although <v> is the same over the FS,
        # what matters is <1/v> and that disproportionally counts
        # slow segments, leading to an overall delay.
        structE = tuple([0.1, 0.04, 0.11])
        timer = core.time_BZ(field, structE)
        self.assertGreater(timer, 1.5 * expect_time)
        self.assertLess(timer, 1.55 * expect_time)

    def test_time_timeBZ(self):
        """ Benchmark """
        structE = tuple([0.2, 0, 0.4 - 0.0001])
        st = time.time()
        iters = 100
        for _ in range(iters):
            core.time_BZ(0.1, structE)
        print(f'timeBZ calculation takes {(time.time() - st) * 1e6/ iters:.0f}'
              f' micro-s (over {(time.time() - st):.2f} s).')

    def test_time_RK4(self):
        """ Benchmark """
        structE = tuple([0.2, 0.04, 0.15])
        field = 10

        kb = core.get_kb(0, structE)  # same for all ka
        vb = core.compute_velocity_b(kb, 0, structE)  # same for all ka
        expect_time = 2 * np.pi * core.HBAR / (core.E * vb * field * core.A)
        iters = 100000
        dt = expect_time / iters

        ka = 0
        core.RK4(ka, dt, field, structE)
        st = time.time()
        for _ in range(iters):
            ka = core.RK4(ka, dt, field, structE)
        print(f'1 RK4 iteration takes {(time.time() - st) * 1e6/ iters:.1f}'
              f' micro-s (over {(time.time() - st):.2f} s).')


class TestConductivity(unittest.TestCase):
    """ Test the conductivity integrals and results. """

    def test_time_integral(self):
        """ Test the intermediate result
        Namely integral_dt_0_infty v(-t) exp(-t/tau)
        """

        structE = tuple([0.5, 0, 0])
        tau = 1e-13
        integral = core.compute_time_integral(
            0, tau / 100, tau, 10, structE, True)
        kb = core.get_kb(0, structE)
        vb = core.compute_velocity_b(kb, 0, structE)
        self.assertAlmostEqual(integral, vb * tau)

        structE = tuple([0.5, 0.1, 0])
        tau = 1e-13
        integral = core.compute_time_integral(
            0, tau / 100, tau, 10, structE, True)
        kb = core.get_kb(0, structE)
        vb = core.compute_velocity_b(kb, 0, structE)
        self.assertGreater(integral, vb * tau)
        self.assertLess(integral, 1.5 * vb * tau)

    def test_Drude(self):
        """ Test all the constants and integrals. To high accuracy.
        This also ensures no miss-by-one errors.
        It also showed the necessity of Simpson rather than Trapezoid for the t-integral.
        """

        # Field is inconsequential, because
        # perfect 1D the velocity is unchanging.
        structE = tuple([0.5, 0])
        field = 10
        tau = 1e-14  # about 15 K is 1e-14

        # This only converges when ntau is high and Simpson integration is used
        # With trapezoidal it takes about 500.
        sbb = core.compute_sigma(True, field, structE,
                                 tau, n_ka=11, dtfrac=31, ntau=25)
        sbb0 = core.compute_sigma(True, 0, structE,
                                  tau, n_ka=11, dtfrac=31, ntau=25)
        self.assertAlmostEqual(sbb, sbb0)

        n = core.compute_total_density(structE)
        kb0 = core.get_kb(0, structE)
        eff_mass = core.HBAR * kb0 / core.compute_velocity_b(kb0, 0, structE)
        drude = n * core.E**2 * tau / eff_mass / 1e8
        self.assertAlmostEqual(sbb, drude, delta=drude / 1e7)


if __name__ == '__main__':
    unittest.main()
