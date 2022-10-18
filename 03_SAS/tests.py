import code as core
import unittest
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize as so


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
        print(f'\nkb calculation takes {(time.time() - st) * 1e6/ iters:.2f}'
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

    def test_time_velocity(self):
        """ Benchmark

        It turns out numba is horrendous in making
        vaa across the FS and I am suspect it can be
        wayyyy faster.
        """

        nr_ka = 100
        merino_small = np.array([0.5 / 8, 0.036 / 8, -0.02])
        st = time.time()
        iters = 0
        while time.time() - st < 0.5:
            core.compute_velocities_bb(nr_ka, merino_small)
            iters += 1
        print(f'\nVb n={nr_ka}, {(time.time() - st) / iters / nr_ka * 1e6:.0f} mu-s.')

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
        print(f'\nDensity calculation takes {(time.time() - st) * 1e6/ iters:.0f}'
              f' micro-s (over {(time.time() - st):.2f} s).')


class TestDiffusionCyclotron(unittest.TestCase):
    """ Test discretising the FS through time evolution with the M matrix. """

    def test_ka_values(self):
        """ Make sure the endpoint is not there. """

        n = 10
        ka = core.get_ka_values(n)
        self.assertEqual(len(ka), n)
        self.assertEqual(ka[0], -np.pi / core.A)
        self.assertEqual(ka[-1], np.pi / core.A * (n - 2) / n)

    def test_state_and_norm(self):
        """ Test the state is well made and normalised. """

        n = 194
        kaa = core.get_ka_values(n)
        state = core.make_state(n, kaa[15], True)
        self.assertEqual(sum(state) - state[15], 0)
        self.assertAlmostEqual(core.compute_norm(state), 1)

        state = core.make_state(n, kaa[15], False)
        self.assertEqual(sum(state) - state[15 + n], 0)
        self.assertAlmostEqual(core.compute_norm(state), 1)

        self.assertRaises(AssertionError, core.make_state, n, kaa[15] * 1.1, 1)

    def test_make_off_diag(self):
        """ Create matrices with an identity off the main diagonal.
        With and without wrapping.
        """

        expect = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])
        result = core._make_off_diag_matrix(5, 0, True)
        self.assertEqual(np.sum(result - expect), 0)

        expect = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])
        result = core._make_off_diag_matrix(5, 2, True)
        self.assertEqual(np.sum(result - expect), 0)

        expect = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
        result = core._make_off_diag_matrix(5, 2, False)
        self.assertEqual(np.sum(result - expect), 0)

        expect = np.array([[0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0]])
        result = core._make_off_diag_matrix(5, -2, True)
        self.assertEqual(np.sum(result - expect), 0)

    def test_make_patterned_off_diag(self):
        """ Same as off diag, but now with variety of elements
        rather than all 1's.
        """

        x = [1, 2, 3, 4, 5]
        expect = np.array([[1, 0, 0, 0, 0],
                           [0, 2, 0, 0, 0],
                           [0, 0, 3, 0, 0],
                           [0, 0, 0, 4, 0],
                           [0, 0, 0, 0, 5]])
        result = core._make_patterned_off_diag_matrix(x, 0)
        self.assertEqual(np.sum(result - expect), 0)

        expect = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 0, 2, 0],
                           [0, 0, 0, 0, 3],
                           [4, 0, 0, 0, 0],
                           [0, 5, 0, 0, 0]])
        result = core._make_patterned_off_diag_matrix(x, 2)
        self.assertEqual(np.sum(result - expect), 0)

        expect = np.array([[0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 2],
                           [3, 0, 0, 0, 0],
                           [0, 4, 0, 0, 0],
                           [0, 0, 5, 0, 0]])
        result = core._make_patterned_off_diag_matrix(x, -2)
        self.assertEqual(np.sum(result - expect), 0)

    def test_second_deriv(self):
        """ Basically two off diagonals and a diagonal. """

        x = [1, 2, 3, 4]
        expect = np.array([[-2, 1, 0, 1],
                           [2, -4, 2, 0],
                           [0, 3, -6, 3],
                           [4, 0, 4, -8]])
        result = core._make_second_derivative(x)
        self.assertEqual(np.sum(np.abs(expect - result)), 0)
        self.assertEqual(np.sum(result), 0)

    def test_diff_matrix(self):
        """ Two second derivs combined. """

        x = np.array([1, 2, 3, 4])
        expect = np.array([[-2, 1, 0, 1, 0, 0, 0, 0],
                           [2, -4, 2, 0, 0, 0, 0, 0],
                           [0, 3, -6, 3, 0, 0, 0, 0],
                           [4, 0, 4, -8, 0, 0, 0, 0],
                           [0, 0, 0, 0, -2, 1, 0, 1],
                           [0, 0, 0, 0, 2, -4, 2, 0],
                           [0, 0, 0, 0, 0, 3, -6, 3],
                           [0, 0, 0, 0, 4, 0, 4, -8]])
        result = core._make_diff_interaction(x)
        self.assertEqual(np.sum(np.abs(expect - result)), 0)
        self.assertEqual(np.sum(result), 0)

    def test_first_deriv(self):
        """ More tricky because of directionality. """

        x = np.array([1, 2, 3, 4])
        forward = True
        expect1 = np.array([[1, 0, 0, -1],
                            [-2, 2, 0, 0],
                            [0, -3, 3, 0],
                            [0, 0, -4, 4]])
        result1 = core._make_first_derivative(x, forward)
        self.assertEqual(np.sum(result1), 0)
        self.assertEqual(np.sum(np.abs(expect1 - result1)), 0)

        forward = False
        result2 = core._make_first_derivative(x, forward)
        expect2 = np.array([[-1, 1, 0, 0],
                            [0, -2, 2, 0],
                            [0, 0, -3, 3],
                            [4, 0, 0, -4]])
        self.assertEqual(np.sum(result2), 0)
        self.assertEqual(np.sum(np.abs(expect2 - result2)), 0)

        # Show the issue
        # Namely positive derivative then you need forward
        #
        # Else, negative elements and elements >1 emerge
        # at low times
        state = [0, 1, 0, 0]
        M = np.identity(4) + 0.01 * result1
        next_state = np.linalg.inv(M) @ state
        self.assertLess(max(next_state), 1)
        self.assertGreater(min(next_state), 0)

        M = np.identity(4) + 0.01 * result2
        next_bad = np.linalg.inv(M) @ state
        self.assertGreater(max(next_bad), 1)
        self.assertLess(min(next_bad), 0)

        # Negative derivative then you need backwards
        state = [0, 1, 0, 0]
        M = np.identity(4) - 0.01 * result1
        next_bad = np.linalg.inv(M) @ state
        self.assertGreater(max(next_bad), 1)
        self.assertLess(min(next_bad), 0)

        M = np.identity(4) - 0.01 * result2
        next_state = np.linalg.inv(M) @ state
        self.assertLess(max(next_state), 1)
        self.assertGreater(min(next_state), 0)

    def test_cyclotron(self):
        """ Two first derivatives moving oppositely. """

        x = np.array([-1, -2, -3, -4])
        expect = -np.array([[-1, 1, 0, 0, 0, 0, 0, 0],
                            [0, -2, 2, 0, 0, 0, 0, 0],
                            [0, 0, -3, 3, 0, 0, 0, 0],
                            [4, 0, 0, -4, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 0, 0, 1],
                            [0, 0, 0, 0, 2, -2, 0, 0],
                            [0, 0, 0, 0, 0, 3, -3, 0],
                            [0, 0, 0, 0, 0, 0, 4, -4]])
        result = core._make_cyclotron_interaction(x)
        self.assertEqual(np.sum(expect - result), 0)
        self.assertEqual(np.sum(np.abs(expect - result)), 0)

        # Ensure the behaviour is correct.
        # This should then lead to NO values <0 or >max
        n = len(x)
        kaa = core.get_ka_values(n)
        state = core.make_state(n, kaa[1], True)
        maxi = max(state)
        M = np.identity(2 * n) + 0.001 * result
        next_state = np.linalg.inv(M) @ state
        self.assertGreater(min(next_state[:4]), 0)
        self.assertEqual(max(next_state[4:]), 0)
        self.assertLess(max(next_state), maxi)
        self.assertGreater(next_state[0], next_state[2])  # truly backwards

        # Also test the other Fermi surface sheet,
        # which is mirrored.
        # And why not, let it go through the border.
        n = len(x)
        kaa = core.get_ka_values(n)
        state = core.make_state(n, kaa[0], False)
        M = np.identity(2 * n) + 0.001 * result
        next_state = np.linalg.inv(M) @ state
        self.assertGreater(min(next_state[4:]), 0)
        self.assertEqual(max(next_state[:4]), 0)
        self.assertLess(max(next_state), maxi)
        self.assertGreater(next_state[5], next_state[-1])  # truly forwards

        # And test that if you then swap the direction of
        # cyclotron motion without updating the matrix,
        # you get negative values.
        state = core.make_state(n, kaa[1], True)
        M = np.identity(2 * n) - 0.001 * result
        next_state = np.linalg.inv(M) @ state
        self.assertLess(min(next_state[:4]), 0)
        self.assertEqual(max(next_state[4:]), 0)
        self.assertGreater(max(next_state), maxi)
        # it *does* go forward, but by making element 0 significantly negative
        # and leaving element 2 ~0
        self.assertLess(next_state[0], next_state[2])

        ##########
        # Repeat for forward motion

        x = np.array([1, 2, 3, 4])
        expect = -np.array([[-1, 0, 0, 1, 0, 0, 0, 0],
                            [2, -2, 0, 0, 0, 0, 0, 0],
                            [0, 3, -3, 0, 0, 0, 0, 0],
                            [0, 0, 4, -4, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1, 1, 0, 0],
                            [0, 0, 0, 0, 0, -2, 2, 0],
                            [0, 0, 0, 0, 0, 0, -3, 3],
                            [0, 0, 0, 0, 4, 0, 0, -4]])
        result = core._make_cyclotron_interaction(x)
        self.assertEqual(np.sum(expect - result), 0)
        self.assertEqual(np.sum(np.abs(expect - result)), 0)

        # Ensure the behaviour is correct.
        # This should then leads to NO values <0 or >max
        n = len(x)
        kaa = core.get_ka_values(n)
        state = core.make_state(n, kaa[1], True)
        maxi = max(state)
        M = np.identity(2 * n) + 0.001 * result
        next_state = np.linalg.inv(M) @ state
        self.assertGreater(min(next_state[:4]), 0)
        self.assertEqual(max(next_state[4:]), 0)
        self.assertLess(max(next_state), maxi)
        self.assertGreater(next_state[2], next_state[0])  # truly forwards

        # Also test the other Fermi surface sheet,
        # which is mirrored.
        # And why not, let it go through the border.
        n = len(x)
        kaa = core.get_ka_values(n)
        state = core.make_state(n, kaa[0], False)
        M = np.identity(2 * n) + 0.001 * result
        next_state = np.linalg.inv(M) @ state
        self.assertGreater(min(next_state[4:]), 0)
        self.assertEqual(max(next_state[:4]), 0)
        self.assertLess(max(next_state), maxi)
        self.assertGreater(next_state[-1], next_state[5])  # truly backwards

    def test_invM_and_diff_lifetime(self):
        """ All the components are tested elsewhere.
        Analytically working out what M or invM looks like is hard.
        But testing that the expected cyclotron time and
        relaxation time come out is very doable. """

        # Constructed in the wake of file 01_show_diffusion
        n = 100
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        tau_aim = 1e-14
        D = 1 / tau_aim / core.A**2
        D2 = core.calculate_D(tau_aim)
        self.assertAlmostEqual(D, D2)
        tau = core.calculate_tau(D2)
        self.assertAlmostEqual(tau, tau_aim)

        t_end = tau_aim * 2
        iterations = 1000

        kaa = core.get_ka_values(n)
        kbb = [core.get_kb(ka, small_merino) for ka in kaa]
        vaa = [core.compute_velocity_a(kb, ka, small_merino)
               for ka, kb, in zip(kaa, kbb)]
        vaa = np.array(vaa * 2)
        state0 = core.make_state(n, kaa[9], True)
        state = np.copy(state0)
        invM = core.make_invM(D, 0, 0, t_end / iterations, n, small_merino)
        for _ in range(iterations):
            state = invM @ state

        va0 = core.compute_norm(state0 * vaa)
        va = core.compute_norm(state * vaa)
        fraction = va / va0
        # The reason this is slightly off is because exponential decay
        # only really sets in on the symmetry point kaa[0]
        # where the left and right cancel eachother and only
        # the size of the central peak matters.
        #
        # However, kaa[0] has va=0, so I cannot test that.
        # Hence stay in the vicinity and show it is approximately this rate.
        # Iteration count or n does not really make much of a difference here
        self.assertAlmostEqual(fraction, np.exp(-2), delta=0.003)

    def test_invM_and_BZ_time_simply(self):
        """ All the components are tested elsewhere.
        Analytically working out what M or invM looks like is hard.
        But testing that the expected cyclotron time and
        relaxation time come out is very doable. """

        # Constructed in the wake of file 02_show_cyclotron
        n = 100
        field = 10
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])

        periods = 2
        tBZ = core.calculate_BZ_time(field, small_merino)
        # This factor 4 follows from the convergence unit test.
        dt = tBZ / n / 4
        t_end = periods * tBZ
        iterations = int(t_end / dt)  # effectively multiple*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[18], True)
        state = np.copy(state0)
        invM = core.make_invM(0, field, 0, dt, n, small_merino)

        st = time.time()
        for _ in range(iterations):
            state = invM @ state

        print(f'\nMatrix multiplication n=100, {(time.time() - st) / iterations * 1e6:.0f} mu-s.')
        self.assertAlmostEqual(np.argmax(state), np.argmax(state0), delta=2)

    def test_relative_dt_n_convergence(self):
        """ Test for fixed n how small dt you need to get
        the most out of this choice of n for a given field """

        return

        # Constructed in the wake of file 02_show_cyclotron
        n = 100
        field = 10
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])

        plt.figure('Convergence by decreasing dt fixed n')
        for multiple in [1, 2, 4, 8, 16, 32]:
            periods = 2
            tBZ = core.calculate_BZ_time(field, small_merino)
            dt = tBZ / n / multiple
            t_end = periods * tBZ
            iterations = int(t_end / dt)  # effectively multiple*n*periods

            kaa = core.get_ka_values(n)
            state0 = core.make_state(n, kaa[18], True)
            state = np.copy(state0)
            invM = core.make_invM(0, field, 0, dt, n, small_merino)
            for _ in range(iterations):
                state = invM @ state

            self.assertAlmostEqual(
                np.argmax(state), np.argmax(state0), delta=2)

            plt.plot(state, label=f'{multiple} dt per dka')
            plt.legend()
            plt.xlabel('ka')
            plt.ylabel('u')

    def test_backscattering(self):
        """ Test that it really has the L provided. """

        # Constructed in the wake of file 02_show_cyclotron
        n = 100
        meanL = 1e-7
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])

        dt = meanL / 1e5 / 1000
        t_end = meanL / 1e5 * 2
        iterations = int(t_end / dt)  # effectively multiple*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[18], True)
        state = np.copy(state0)
        invM = core.make_invM(0, 0, meanL, dt, n, small_merino)
        for _ in range(iterations):
            state = invM @ state

        fraction_vb = (max(state[:n]) - max(state[n:])) / max(state0)
        self.assertAlmostEqual(fraction_vb, np.exp(-2), delta=0.02)



class TestRK4Differentiation(unittest.TestCase):
    """ New in the faster diffusion """

    def test_diffusion_RK4(self):
        """ Test that the diffusion acts as intended. """

        n = 100
        tau = 1e-14
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        # Used for graphs in document '04 diffusion faster'
        show = False

        D = core.calculate_D(tau)
        dt = tau / 1500
        iters = int(5 * tau / dt)

        kaa = core.get_ka_values(n)
        state = core.make_state(n, kaa[10], True)
        deriv = core.deriv_RK4(D, 0, 0, n, small_merino)
        op = core.make_RK4(deriv, dt)

        vaa = core.compute_velocities_aa(n, small_merino, True)
        va_nets = []
        t = 0
        tt = []

        if show:
            plt.figure()
        for i in range(iters):
            if show and i in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, iters-1]:
                plt.plot(kaa, state[:n])
                plt.plot(kaa + 2 * np.pi / core.A, state[n:])
            va_nets.append(core.compute_norm(vaa * state))
            tt.append(t)

            state = op @ state
            t += dt
            # Can now happen because this is an explicit method
            self.assertGreaterEqual(np.min(state), 0)

        def exp(t, a, tau):
            return a * np.exp(-t / tau)
        p, pcov = so.curve_fit(exp, tt, va_nets, p0=[va_nets[0], tau])

        tau_real = p[1]
        # Accuracy here is limited by n
        self.assertAlmostEqual(tau_real, tau, delta=tau / 50)
        if show:
            print('Exponential decay:', p, np.sqrt(np.diag(pcov)))
            xx = np.linspace(0, max(tt) * 1.05, 1000)
            yy = exp(xx, *p)

            plt.xlabel('ka')
            plt.ylabel('u')
            plt.figure()
            plt.plot(tt, va_nets)
            plt.plot(xx, yy)
            plt.xlabel('t')
            plt.ylabel('<va> (m/s)')
            plt.show()

    def test_cyclotron_RK4(self):
        """ Test upward differencing, and timeframe. """

        periods = 2
        field = 10
        n = 100
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        show = False

        tBZ = core.calculate_BZ_time(field, small_merino)
        # This factor 4 follows from the convergence unit test.
        dt = tBZ / n / 4
        t_end = periods * tBZ
        iters = int(t_end / dt)  # effectively 4*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[38], True)
        state = np.copy(state0)
        deriv = core.deriv_RK4(0, field, 0, n, small_merino)
        op = core.make_RK4(deriv, dt)
        state = np.copy(state0)

        tt = []
        maxi = []
        st = time.time()
        if show:
            plt.figure()
        for i in range(iters + 1):
            self.assertGreaterEqual(np.min(state), 0)

            if show and i in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, iters-1]:
                plt.plot(kaa, state[:n])
                plt.plot(kaa + 2 * np.pi / core.A, state[n:])
            maxi.append(np.argmax(state))
            tt.append(i * dt)
            state = op @ state

        # Basically, the maximum may shift 1 position in n compared to expected.
        self.assertLess(abs(kaa[maxi[0] % n] - kaa[maxi[-1] % n]), 2 * np.pi / core.A / n * 1.5)

        if show:
            maxi = np.array(maxi)
            plt.xlabel('ka')
            plt.ylabel('u')
            plt.figure()
            plt.plot(tt, kaa[maxi % n])
            plt.ylabel('particle position')
            plt.xlabel('time (s)')
            plt.show()

        print(f'\nMatrix multiplication n=100 RK4, {(time.time() - st) / iters * 1e6:.0f} mu-s.')
        self.assertAlmostEqual(np.argmax(state), np.argmax(state0), delta=2)

    def test_cyclotron_RK4_order(self):
        """ Test upward differencing, and timeframe. """

        periods = 2
        field = 10
        n = 100
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        show = False

        tBZ = core.calculate_BZ_time(field, small_merino)
        # This factor 4 follows from the convergence unit test.
        dt = tBZ / n / 1.2
        t_end = periods * tBZ
        iters = int(t_end / dt)  # effectively 4*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[3], True)
        state = np.copy(state0)
        deriv = core.deriv_RK4(0, field, 0, n, small_merino)
        deriv2 = -core._make_fast_RK4_cycl(n, field, kaa[1] - kaa[0], small_merino)

        self.assertLess(np.max(np.abs(deriv - deriv2)), np.max(deriv) / 10)
        op = core.make_RK4(deriv, dt)
        op2 = core.make_RK4(deriv2, dt)
        state = np.copy(state0)
        state2 = np.copy(state0)

        tt = []
        maxi = []
        maxi2 = []
        st = time.time()
        if show:
            plt.figure()
        for i in range(iters + 1):
            self.assertGreaterEqual(np.min(state), 0)

            if show and i in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, iters-1]:
                plt.plot(kaa, state[:n], color='tab:red')
                plt.plot(kaa + 2 * np.pi / core.A, state[n:], color='tab:red')
                plt.plot(kaa, state2[:n], color='tab:blue')
                plt.plot(kaa + 2 * np.pi / core.A, state2[n:], color='tab:blue')
            maxi.append(np.argmax(state))
            maxi2.append(np.argmax(state2))
            tt.append(i * dt)
            state = op @ state
            state2 = op2 @ state2

        # Basically, the maximum may shift 1 position in n compared to expected.
        self.assertLess(abs(kaa[maxi[0] % n] - kaa[maxi[-1] % n]), 2 * np.pi / core.A / n * 1.5)

        if show:
            maxi = np.array(maxi)
            maxi2 = np.array(maxi2)
            plt.xlabel('ka')
            plt.ylabel('u')
            plt.figure()
            plt.plot(tt, kaa[maxi % n], color='tab:red')
            plt.plot(tt, kaa[maxi2 % n], color='tab:blue')
            plt.ylabel('particle position')
            plt.xlabel('time (s)')
            plt.title('blue RK4, red ~trapezoid')
            plt.show()

        print(f'\nMatrix multiplication n=100 RK4, {(time.time() - st) / iters * 1e6:.0f} mu-s.')
        self.assertAlmostEqual(np.argmax(state), np.argmax(state0), delta=2)
        self.assertAlmostEqual(np.argmax(state2), np.argmax(state0), delta=2)

    def test_backscattering(self):
        """ Test that it really has the L provided. """

        # Constructed in the wake of file 02_show_cyclotron
        n = 100
        meanL = 1e-7
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        show = False

        dt = meanL / 1e5 / 1000
        t_end = meanL / 1e5 * 2
        iterations = int(t_end / dt)  # effectively multiple*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[18], True)
        state = np.copy(state0)
        deriv = core.deriv_RK4(0, 0, meanL, n, small_merino)
        op = core.make_RK4(deriv, dt)

        if show:
            plt.figure()
        for i in range(iterations):
            state = op @ state

            if i % 100 == 0 and show:
                plt.plot(state)
        if show:
            plt.xlabel('ka')
            plt.ylabel('u')
            plt.show()

        fraction_vb = (max(state[:n]) - max(state[n:])) / max(state0)
        self.assertAlmostEqual(fraction_vb, np.exp(-2), delta=0.02)

    def test_backscattering_isotropic(self):
        """ Test that it really has the L provided. """

        # Constructed in the wake of file 02_show_cyclotron
        n = 100
        meanL = 1e-7
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        show = False

        dt = meanL / 1e5 / 1000
        t_end = meanL / 1e5 * 2
        iterations = int(t_end / dt)  # effectively multiple*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[18], True)
        state = np.copy(state0)
        deriv = core.deriv_RK4(0, 0, meanL, n, small_merino, bs_type=4)
        op = core.make_RK4(deriv, dt)

        if show:
            plt.figure()
        for i in range(iterations):
            state = op @ state

            if i % 100 == 0 and show:
                plt.plot(state)
        if show:
            plt.xlabel('ka')
            plt.ylabel('u')
            plt.show()

        fraction_vb = (max(state[:n]) - max(state[n:])) / max(state0)
        self.assertAlmostEqual(fraction_vb, np.exp(-2), delta=0.02)

    def test_backscattering_RTA(self):
        """ Test exponential decay violating charge conservation with time, like RTA. """

        n = 100
        meanL = 1e-7
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        show = False

        dt = meanL / 1e5 / 1000
        t_end = meanL / 1e5 * 2
        iterations = int(t_end / dt)  # effectively multiple*n*periods

        kaa = core.get_ka_values(n)
        state0 = core.make_state(n, kaa[18], True)
        state = np.copy(state0)
        deriv = core.deriv_RK4(0, 0, meanL, n, small_merino, bs_type=5)
        op = core.make_RK4(deriv, dt)

        if show:
            plt.figure()
        for i in range(iterations):
            state = op @ state

            if i % 100 == 0 and show:
                plt.plot(state)
        if show:
            plt.xlabel('ka')
            plt.ylabel('u')
            plt.show()

        fraction_vb = (max(state[:n]) - max(state[n:])) / max(state0)
        self.assertAlmostEqual(fraction_vb, np.exp(-2), delta=0.02)

    def test_time_make_matrix_RK4(self):
        """ Because there is now no way to adjust the timestep with
        integrity AND no inversion to slow down tremendously,
        aim to remake this matrix many times.
        But then you need an idea about the cost. """

        n = 100
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        D = core.calculate_D(1e-14)
        dt = 1e-17
        iters = 0
        deriv = core.deriv_RK4(D, 10, 1e-9, n, small_merino)
        st = time.time()
        while time.time() - st < 0.3:
            core.make_RK4(deriv, dt)
            iters += 1
        print(f'\nMake RK4 matrix in {(time.time() - st) / iters * 1e6:.0f} mu-s')


class TestConductivity(unittest.TestCase):
    """ Test all the integration steps and Drude/relaxation theory.

    !! All now using explicit RK4 rather than implicit Euler !!
    """

    def test_velocity_corr_back(self):
        """ Test this against semi-analytical results. """

        n = 100
        structE = tuple([0.5, 0, 0])
        D = 0
        meanL = 1e-7
        field = 0

        kaa = core.get_ka_values(n)
        kb0 = core.get_kb(kaa[2], structE)
        vb0 = core.compute_velocity_b(kb0, kaa[2], structE)
        tau = meanL / vb0
        dt = tau / 100
        steps = tau * 10 / dt
        steps += 1 + steps % 2

        state = core.make_state(n, kaa[2], True)
        deriv = core.deriv_RK4(D, field, meanL, n, structE)
        op = core.make_RK4(deriv, dt)

        corr_va, corr_vb, _ = core.compute_vcorr(
            op, state, dt, steps, structE)

        # namely vb(const) exp(-t/tau) integrated
        self.assertAlmostEqual(corr_vb, vb0 * tau, delta=vb0 * tau / 1000)
        self.assertEqual(corr_va, 0)  # va is zero everywhere

    def test_velocity_corr_back2(self):
        """ Test this against semi-analytical results.
        Negative sheet
        """

        n = 100
        structE = tuple([0.5, 0, 0])
        D = 0
        meanL = 1e-7
        field = 0

        kaa = core.get_ka_values(n)
        kb0 = core.get_kb(kaa[2], structE)
        vb0 = -core.compute_velocity_b(kb0, kaa[2], structE)
        tau = meanL / abs(vb0)
        dt = tau / 100
        steps = tau * 10 / dt
        steps += 1 + steps % 2

        state = core.make_state(n, kaa[2], False)
        deriv = core.deriv_RK4(D, field, meanL, n, structE)
        op = core.make_RK4(deriv, dt)

        corr_va, corr_vb, _ = core.compute_vcorr(
            op, state, dt, steps, structE)

        # namely vb(const) exp(-t/tau) integrated
        self.assertAlmostEqual(corr_vb, vb0 * tau, delta=abs(vb0 * tau / 1000))
        self.assertEqual(corr_va, 0)  # va is zero everywhere

    def test_time_velocity_corr_back(self):
        """ Benchmark. """

        n = 100
        structE = tuple([0.5, 0, 0])
        D = 0
        meanL = 1e-7
        field = 0

        kaa = core.get_ka_values(n)
        kb0 = core.get_kb(kaa[2], structE)
        vb0 = core.compute_velocity_b(kb0, kaa[2], structE)
        tau = meanL / abs(vb0)
        # Used to require 1000.
        # Now 100 is a luxury and n-limited.
        # And therefore 10x as fast *at least*
        dt = tau / 100
        steps = tau * 10 / dt
        steps += 1 + steps % 2

        state = core.make_state(n, kaa[2], True)
        deriv = core.deriv_RK4(D, field, meanL, n, structE)
        op = core.make_RK4(deriv, dt)

        core.compute_vcorr(op, state, dt, steps, structE)
        st = time.time()
        iters = 0
        while time.time() - st < 0.5:
            core.compute_vcorr(op, state, dt, steps, structE)
            iters += 1
        print(
            f'\nVelocity correlation in {(time.time() - st) / iters * 1e3:.0f} ms.')

    def test_velocity_corr_back_cycl(self):
        """ Test this against semi-analytical results. """

        n = 100
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        D = 0
        meanL = 1e-7
        field = 10
        tBZ = core.calculate_BZ_time(field, small_merino)

        kaa = core.get_ka_values(n)
        kb0 = core.get_kb(kaa[2], small_merino)
        vb0 = core.compute_velocity_b(kb0, kaa[1], small_merino)
        va_max = core.compute_velocity_a(kb0, kaa[25], small_merino)
        tau_b = meanL / vb0
        tau_a = tBZ / 2
        dt = min(tau_a, tau_b) / 1000
        steps = max(tau_a, tau_b * 5) * 2 / dt
        steps += steps % 2  # make even to test function makes it odd

        state = core.make_state(n, kaa[1], True)
        deriv = core.deriv_RK4(D, field, meanL, n, small_merino)
        op = core.make_RK4(deriv, dt)

        corr_va, corr_vb, _ = core.compute_vcorr(
            op, state, dt, steps, small_merino)

        # namely vb(const) exp(-t/tau) integrated
        self.assertAlmostEqual(corr_vb, vb0 * tau_b, delta=vb0 * tau_b / 50)
        # Do not ask where the sqrt(2) comes from.
        # Otherwise, this is half an oscillation,
        # then average 1/sqrt(2) the velocity
        # Somehow a factor ~2 for gradual decay during the first half-period as well.
        self.assertAlmostEqual(corr_va, va_max / 2.15 * tau_a / np.sqrt(2),
                               delta=abs(corr_va) / 20)

        tau_bs = core.calculate_bs(meanL, small_merino)
        self.assertAlmostEqual(tau_bs, meanL / 1e5, delta=meanL / 2e6)

    def test_sigma_drude(self):
        """ Test an uncorrugated 1D FS """

        n = 15

        small_merino = tuple([0.5 / 8, 0, 0])
        tau_diff = 1e-12
        back_L = 1e-9
        field = 0

        D = core.calculate_D(tau_diff)
        dt = 1e-15
        steps = int(10 * max(back_L / 1e5, tau_diff) / dt)

        deriv = core.deriv_RK4(D, field, back_L, n, small_merino)
        sbb, saa, sab = core.sigma(deriv, dt, steps, small_merino)

        # Same formula as eq (1) in the paper
        # Exception is that this is 1 sheet, but still has a factor 4
        # This is because with the allowance to repeat-backscattering,
        # the mean free path is actually twice as long as stated above.
        expect_bb = 2 * core.E**2 * back_L / \
            (np.pi * core.HBAR * core.A * core.C)
        self.assertEqual(sab, 0)  # va=0 so not even numerical inaccuracy
        self.assertEqual(saa, 0)  # va=0 so not even numerical inaccuracy
        # The remaining deviation is because with repeated backscattering
        # the actual L is not quite 1 nm
        # Really, it is slightly more than 1 nm.
        #
        # This used to be 1/100 accurate because of O(dt) convergence
        # from the implicit method. Now it is O(dt^4) and the constant
        # seems similar. No dk error because it is uniform.
        # So increase dt by 10x (i.e. 1e4 more error) and lower error margin by 1e3
        # because this test was slow.
        self.assertAlmostEqual(expect_bb, sbb, delta=expect_bb / 1e5)

    def test_sigma_zero_field(self):
        """ Test a slightly corrugated 1D FS """

        n = 15
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        tau_diff = 1e-12
        back_L = 1e-9
        field = 0

        D = core.calculate_D(tau_diff)
        dt = 1e-16
        steps = int(10 * max(back_L / 1e5, tau_diff) / dt)

        deriv = core.deriv_RK4(D, field, back_L, n, small_merino)
        sbb, saa, sab = core.sigma(deriv, dt, steps, small_merino)

        # Same formula as eq (1) in the paper
        # Exception is that this is 1 sheet, but still has a factor 4
        # This is because with the allowance to repeat-backscattering,
        # the mean free path is actually twice as long as stated above.
        expect_bb = 2 * core.E**2 * back_L / \
            (np.pi * core.HBAR * core.A * core.C)
        self.assertLess(sab, 0.001)  # no field, within numerical precision 0

        # Estimate saa semi-analytically
        vaa = core.compute_velocities_aa(n, small_merino)
        vbb = core.compute_velocities_bb(n, small_merino)
        estimate_aa = 2 * core.E**2 / (np.pi * core.HBAR * core.A * core.C)
        estimate_aa *= np.mean(vaa**2 / vbb) * tau_diff
        self.assertAlmostEqual(saa, estimate_aa, delta=estimate_aa / 25)
        # The remaining deviation is because with repeated backscattering
        # the actual L is not quite 1 nm
        # Really, it is slightly more than 1 nm.
        self.assertAlmostEqual(expect_bb, sbb, delta=expect_bb / 100)

    def test_sigma_relaxation_field(self):
        """ This is hard to do.

        Two tests at once.
        1) Test that sab is correct, which requires field for a non-zero value.
        2) Test the results against the relaxation time approximation
        3) Verify that diff & meanL both 1e-14 seconds results in tau_a=tau_b~1e-14.
            Despite magnetic field!
        """

        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        tau = 1e-12
        D = core.calculate_D(tau)
        meanL = 1e-9
        field = 100
        dt = meanL / 1e5 / 100
        steps = int(3 * tau / dt)
        n = 30

        # tabulated from file 02/execute
        # only small_merino
        relax_aa = 1.027631352648058e-05  # 1/muOhmcm
        relax_bb = 0.0013327587199451155  # 1/muOhmcm
        relax_ab = -1.4614339173138527e-07  # 1/muOhmcm

        st = time.time()
        iters = 0
        while time.time() - st < 3:
            deriv = core.deriv_RK4(D, field, meanL, n, small_merino, bs_type=5)
            sbb, saa, sab = core.sigma(deriv, dt, steps, small_merino)
            iters += 1
        print(f'\nMatrix + sigma benchmark in {(time.time() - st) / iters * 1e3:.0f} ms')

        deriv = core.deriv_RK4(D, 0, meanL, n, small_merino, bs_type=5)
        sbb0, saa0, sab0 = core.sigma(deriv, dt, steps, small_merino)
        self.assertAlmostEqual(saa / 1e8, relax_aa, delta=relax_aa / 10)
        self.assertAlmostEqual(sbb / 1e8, relax_bb, delta=relax_bb / 10)
        self.assertAlmostEqual(abs(sab) / 1e8, abs(relax_ab), delta=abs(relax_ab) / 3)

        tBZ = core.calculate_BZ_time(100, small_merino)
        print(f'Verified against relaxation method.')
        print(f'Small Merino: At 100 T, you are crossing {tau / tBZ:.2f} sheets in time tau={tau}.')

    def test_adaptive_velocity_corr_back(self):
        """ Test this against semi-analytical results. """

        n = 100
        structE = tuple([0.5, 0, 0])
        D = 0
        meanL = 1e-7
        field = 0

        kaa = core.get_ka_values(n)
        kb0 = core.get_kb(kaa[2], structE)
        vb0 = core.compute_velocity_b(kb0, kaa[2], structE)
        tau = meanL / vb0
        t_end = 10 * tau

        state = core.make_state(n, kaa[2], True)
        deriv = core.deriv_RK4(D, field, meanL, n, structE)

        corr_va, corr_vb, _, _ = core.adaptive_vcorr(
            deriv, state, t_end, 1 / 100, 5, structE)

        # namely vb(const) exp(-t/tau) integrated
        self.assertAlmostEqual(corr_vb, vb0 * tau, delta=vb0 * tau / 1000)
        self.assertEqual(corr_va, 0)  # va is zero everywhere

    def test_adaptive_velocity_corr_back_cycl(self):
        """ Test this against semi-analytical results. """

        n = 100
        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        D = 0
        meanL = 1e-7
        field = 10
        tBZ = core.calculate_BZ_time(field, small_merino)

        kaa = core.get_ka_values(n)
        kb0 = core.get_kb(kaa[2], small_merino)
        vb0 = core.compute_velocity_b(kb0, kaa[1], small_merino)
        va_max = core.compute_velocity_a(kb0, kaa[25], small_merino)
        tau_b = meanL / vb0
        tau_a = tBZ / 2
        t_end = max(tau_a * 3, tau_b * 15)

        state = core.make_state(n, kaa[1], True)
        deriv = core.deriv_RK4(D, field, meanL, n, small_merino)

        corr_va, corr_vb, _, _ = core.adaptive_vcorr(
            deriv, state, t_end, 1 / 100, 5, small_merino)

        # namely vb(const) exp(-t/tau) integrated
        self.assertAlmostEqual(corr_vb, vb0 * tau_b, delta=vb0 * tau_b / 50)
        # Do not ask where the sqrt(2) comes from.
        # Otherwise, this is half an oscillation,
        # then average 1/sqrt(2) the velocity
        # Somehow a factor ~2 for gradual decay during the first half-period as well.
        self.assertAlmostEqual(corr_va, va_max / 2.15 * tau_a / np.sqrt(2),
                               delta=abs(corr_va) / 20)

        tau_bs = core.calculate_bs(meanL, small_merino)
        self.assertAlmostEqual(tau_bs, meanL / 1e5, delta=meanL / 2e6)

    def test_adaptive(self):
        """ Dive right in. Relaxation time approximation full on. """

        small_merino = tuple([0.5 / 8, 0.036 / 8, -0.02])
        tau = 1e-12
        D = core.calculate_D(tau)
        meanL = 1e-9
        field = 100
        t_end = 25 * tau
        fs = 10
        n = fs * 10

        # tabulated from file 02/execute
        # only small_merino
        relax_aa = 1.027631352648058e-05  # 1/muOhmcm
        relax_bb = 0.0013327587199451155  # 1/muOhmcm
        relax_ab = -1.4614339173138527e-07  # 1/muOhmcm

        st = time.time()
        iters = 0
        tevols = 0
        while time.time() - st < 3:
            deriv = core.deriv_RK4(D, field, meanL, n, small_merino, bs_type=5)
            sbb, saa, sab, tevol = core.adaptive(deriv, t_end, 1e-9, 5, fs, small_merino)
            tevols += tevol
            iters += 1
        print(f'\nAdaptive sigma benchmark in {(time.time() - st) / iters * 1e3:.0f} ms')
        print(f'With each t evolution taking {(time.time() - st) / tevols * 1e6:.0f} mu-s')

        self.assertAlmostEqual(saa / 1e8, relax_aa, delta=relax_aa / 10)
        self.assertAlmostEqual(sbb / 1e8, relax_bb, delta=relax_bb / 10)
        self.assertAlmostEqual(sab / 1e8, relax_ab, delta=abs(relax_ab) / 8)
        print(f'Adaptive verified against relaxation method.')

        # Convergence:
        # n=fs=30:      128356.55453644475 1039.1199370167010 -13.962601858355269
        # n=fs=60:      128356.51715450652 1044.2635676490052 -14.16309009222654
        # n=fs=200:     128356.49089370697 1049.0883047199457 -14.29059842962682
        # n=11 x fs=20: 128356.48984066948 1049.2969070662910 -14.295348069638186
        # n=11 x fs=44: 128356.48429189935 1050.4621458200254 -14.320964987502105
        # Compare last one:
        # n=22 x fs=20: 128356.48472717644 1050.3632708710397 -14.318846928652363
        #
        # NOTE: These were all generated with bad cyclotron orbital motion.
        # self.assertAlmostEqual(sbb, 128356.48472717644, delta=100)
        # self.assertAlmostEqual(saa, 1050.3632708710397, delta=100)
        # self.assertAlmostEqual(sab, -14.318846928652363, delta=2)

    def test_check_interval(self):
        """ Quite a peculiar function.
        Could have been right out of Project Euler if the numbers were
        higher. """

        core.check_interval(20 * 10, 10)
        core.check_interval(20 * 10, 20)
        self.assertRaises(ValueError, core.check_interval, 20 * 10, 28)
        self.assertRaises(ValueError, core.check_interval, 20 * 12, 3)
        self.assertRaises(ValueError, core.check_interval, 20 * 10, 400)




if __name__ == '__main__':
    unittest.main(exit=False)
    plt.show()
