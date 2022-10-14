# The actual calculations.
# This code is tested in tests.py
#
# -----------------------
# Part of the SmallAngleScatteringLMO repository
# Subject to the LICENSE terms and conditions
# Written by Roemer Hinlopen, 2022

import numpy as np
import numba
import time

# Lattice parameters, see Popovic2006
A = 9.499e-10
B = 5.523e-10
C = 12.762e-10
E = 1.602176634e-19
KB = 1.380649e-23
HBAR = 1.054571817e-34

#################
# Constants
# Electronic structure
#################


@numba.njit()
def compute_energy(kb, ka, structE):
    """ Use units of eV.
    This function is ambivalent, but velocities are not.

    Note that you are already free to give this any quasi-1D Fermi surface
    with 2 TB coefficients. You are free to expand this, structE is simply
    passed through the code and can have as many parameters as you wish.

    The requirement is that this remains quasi-1D with chains along b,
    smooth and satisfies the periodicity of the Brillouin zone.
    The first requirement is principally to enable FS-integrals
    to be written as ka integrals.

    So yes, this code can be easily adopted. But make sure you satisfy
    the LICENSE terms and conditions, attribution in particular.
    """
    return -2 * structE[0] * np.cos(kb * B) - 2 * structE[1] * np.cos(ka * A)


@numba.njit()
def get_kb(ka, structE):
    """ Get the kb corresponding to the Fermi surface.

    Uses binary search on energy values.
    Positive kb only
    """

    kb_BZ = np.pi / B

    kb_min = 0
    kb_step = kb_BZ
    Eleft = compute_energy(kb_min, ka, structE)
    Eright = compute_energy(kb_min + kb_step, ka, structE)
    increasing = Eright > Eleft

    # Chemical potential under (over) bandwidth
    assert((structE[-1] > Eleft) or (structE[-1] > Eright))
    assert((structE[-1] < Eleft) or (structE[-1] < Eright))

    while kb_step > kb_BZ * 1e-10:
        kb_step /= 2
        Ehalfway = compute_energy(kb_min + kb_step, ka, structE)
        if increasing:
            kb_min += kb_step * (Ehalfway < structE[-1])
        else:
            kb_min += kb_step * (Ehalfway > structE[-1])
    return kb_min + kb_step / 2


@numba.njit()
def compute_velocity_b(kb, ka, structE):
    """ Return the velocity m/s. Your reponsibility to assure this is
    at the Fermi energy!
    """
    return 2 * structE[0] * np.sin(kb * B) * B * E / HBAR


@numba.njit()
def compute_velocity_a(kb, ka, structE):
    return 2 * structE[1] * np.sin(ka * A) * A * E / HBAR


@numba.njit()
def compute_density(structE):
    """ Get the carrier density as fraction of a full BZ of this band. """

    # Formula is
    # n = 1/8pi^3 * 2pi/C * area_encl * 2 * 2
    #   Fourier     c-axis    fs      spin nr_sheets
    # But fraction is just
    # n_frac = area_encl / 4pi^2 * BA

    n = 51  # must be odd for Simpson's rule

    total = 0
    ka = -np.pi / A
    dka = 2 * np.pi / A / (n - 1)
    for i in range(n):
        # 1 4 2 4 ... 4 2 4 1
        weight = 1 if i == 0 or i == n - 1 else 2 * (i % 2) + 2
        total += weight * get_kb(ka, structE)
        ka += dka
    area = total * dka / 3 * 2  # 2 for positive and negative kb

    return area / 4 / np.pi**2 * B * A


@numba.njit()
def compute_total_density(structE):
    """ Get the carrier density as a number m^-3. """
    return 2 * compute_density(structE) / (A * B * C)


#################
# Electronic structure
# Cyclotron motion
#################


@numba.njit()
def RK4(ka0, dt, field, structE):
    """ Progress ka by time step dt. """

    # Newton: hbar * kdot = q cross(v, B)
    # For ka: kadot = q*vb*Ba / hbar
    # Electron or hole is taken care of by the sign of the velocity.

    kb0 = get_kb(ka0, structE)
    vb0 = compute_velocity_b(kb0, ka0, structE)
    ky_dot_0 = E * vb0 * field / HBAR

    ka1 = ka0 + dt / 2 * ky_dot_0
    kb1 = get_kb(ka1, structE)
    vb1 = compute_velocity_b(kb1, ka1, structE)
    ky_dot_1 = E * vb1 * field / HBAR

    ka2 = ka0 + dt / 2 * ky_dot_1
    kb2 = get_kb(ka2, structE)
    vb2 = compute_velocity_b(kb2, ka2, structE)
    ky_dot_2 = E * vb2 * field / HBAR

    ka3 = ka0 + dt * ky_dot_2
    kb3 = get_kb(ka3, structE)
    vb3 = compute_velocity_b(kb3, ka3, structE)
    ky_dot_3 = E * vb3 * field / HBAR

    return ka0 + (ky_dot_0 + 2 * ky_dot_1 + 2 * ky_dot_2 + ky_dot_3) / 6 * dt


@numba.njit()
def _time_BZ(field, structE, dt):
    """ Get the approximate time for a full 2pi/A movement.

    Will be significantly more accurate than dt, but test 2*dt to estimate.
    Tip: Give dt as 1e-14 s for field 10 Tesla.
    """

    assert(field != 0)  # infinite loop
    assert(dt > 0)

    ka = 0
    BZ = 2 * np.pi / A
    timer = 0
    while abs(ka) < BZ:
        ka = RK4(ka, dt, field, structE)
        timer += dt
        if timer > 1e6 * dt:
            assert(False)

    # Do a linear interpolation to get the result more
    # accurately than the timestep given.
    # This is usually a lot more accurate. Orders of magnitude.
    #
    # Especially because corrections come from non-uniformity of vb
    # over this final tiny step, which is located at the BZ edge.
    # At the BZ edge vf-variation is 3rd order since E has no k or k^3 term here.
    # In other words: The FS is flat so vf only varies in sub-sub-leading order.
    last_ka = RK4(ka, -dt, field, structE)
    overshoot = (abs(ka) - BZ) / abs(last_ka - ka) * dt
    return timer - overshoot


@numba.njit()
def time_BZ(field, structE):
    """ Calculate how long it takes to traverse the BZ. """

    dt = 1e-9
    timer = _time_BZ(1, structE, dt)
    while timer / dt < 100:
        dt /= 10
        timer = _time_BZ(1, structE, dt)
    return timer / abs(field)


@numba.njit()
def compute_fieldscale(structE, tau):
    """ Compute at what field you can traverse the FS in expectation.
    Equivalent to wct=1 in higher-D
    """
    return time_BZ(1, structE) / tau / 2 / np.pi


#################
# Cyclotron motion
# Sigma
#################

@numba.njit()
def compute_time_integral(ka, dt, tau, field, structE, is_vb, ntau=15):
    """ Compute integral_dt_0^infty v(-t) exp(-t/tau)
    Including cyclotron motion.
    """

    # If field is present,
    # make sure the BZ is not skipped through too quickly
    # or the whole integration makes no sense at all.
    if field != 0:
        tBZ = time_BZ(field, structE)
        assert(dt < tBZ / 10)

    # Simpson rule
    # Must end with a weight 4.
    # Which means at the while loop it is set to 2.
    timer = 0
    total = 0
    weight = 1

    iteration = 0
    while timer < ntau * tau or weight != 2:
        kb = get_kb(ka, structE)
        if is_vb:
            v = compute_velocity_b(kb, ka, structE)
        else:
            v = compute_velocity_a(kb, ka, structE)
        total += v * np.exp(-timer / tau) * weight
        weight = 4 if weight < 4 else 2
        timer += dt
        ka = RK4(ka, dt, field, structE)

        iteration += 1

    # Last point has weight 1
    # Although it is essentially 0 (or rather exp(-ntau))
    total -= 3 * v * np.exp(-(timer - dt) / tau)
    return total * dt / 3


@numba.njit()
def compute_sigma(is_bb, field_c, structE, tau, *, n_ka=101, dtfrac=101, ntau=15):
    """ Compute sigma_bb or sigma_aa in 1/muOhmcm
    Set is_bb to 2 for sigma_ab

    Convergence behaviour:
    - n_ka is number of Fermi surface starting points. O(n), convergence 1/n^2
    - dtfrac is timesteps in tau/dtfrac. O(n), convergence 1/n^2 (?, faster than linear)
    - ntau is how far in time to go. O(n), conductivity *under*estimate exp(-n).

    Includes spin degeneracy.
    Force-computes all the time integrals. No cutting it short
    at the next ka point and extrapolating from there. Reason: I want
    a proper B->0 limit and so time domain not wc*t domain
    necessary for k-to-k movement.
    """
    assert(n_ka > 5)
    assert(n_ka % 2)
    dt = tau / dtfrac

    total = 0
    kac = np.linspace(-np.pi / A, np.pi / A, n_ka)
    dka = kac[1] - kac[0]
    for i, ka in enumerate(kac):
        kb = get_kb(ka, structE)
        vb = compute_velocity_b(kb, ka, structE)
        va = compute_velocity_a(kb, ka, structE)
        vf = np.sqrt(vb**2 + va**2)
        # indeed +va for sigma_ab.
        v0 = vb if is_bb == 1 else va

        # Because it is integral over Fermi surface not ka,
        # there is a Jacobian term which geometrically works out in a triangle
        # equal to dkf/dka = vf/vb.
        dkf = dka * vf / vb
        t_dep = compute_time_integral(ka, dt, tau, field_c, structE,
                                      is_bb, ntau)
        t_indep = v0 / HBAR / vf
        # 1 4 2 4 2 4 ... 4 1
        weight = 1 if i == 0 or i == n_ka - 1 else (2 + 2 * (i % 2))
        # /3 from Simpson, *2 for negative kf Fermi sheet.
        total += weight * t_indep * t_dep * dkf / 3 * 2

    const = E**2 / 2 / np.pi**2 / C
    return const * total / 1e8


#################
# Sigma
# Interface
#################


def single_error(field, is_bb, structE, tau, ntau, n_ka, dtfrac):
    """ Calculate the conductivity at these settings and with half-accurate settings
    to estimate the error made.

    Note: Many algorithms here are O(1/n^2) converging or faster, so the returned
    error is quite close to the actual error made. Standard technique.
    """

    s0 = compute_sigma(is_bb, 0, structE, tau,
                       n_ka=n_ka, dtfrac=dtfrac, ntau=ntau)
    s1 = compute_sigma(is_bb, 0, structE, tau,
                       n_ka=n_ka * 2 + 1, dtfrac=dtfrac, ntau=ntau)
    nka_err = abs(s1 - s0) / s0
    s2 = compute_sigma(is_bb, 0, structE, tau,
                       n_ka=n_ka, dtfrac=dtfrac * 2, ntau=ntau)
    dt_err = abs(s2 - s0) / s0
    ntau_err = np.exp(-ntau)

    stype = 'sbb' if is_bb else 'saa'
    text = f'  > {stype} B={field:.1f} T'

    if ntau_err > nka_err and ntau_err > dt_err:
        print(f'{text}: nTau limited to {ntau_err:.1e} fractional error')
        return ntau_err
    elif dt_err > nka_err:
        print(f'{text}: dt limited to {dt_err:.1e} fractional error')
        return dt_err
    else:
        print(f'{text}: n_ka limited to {nka_err:.1e} fractional error')
        return nka_err


def analyse_errors(max_field, structE, tau, ntau, n_ka, dtfrac):
    """ Estimate the errors made at these settings for:
    Field=0, max and sigma_bb, sigma_cc
    Do so by half-accuracy settings.

    It is expected the error does not change and therefore
    it is only computed at lowest and highest field.
    This is reported directly to the user, and they can check
    whether this is indeed the case.
    """

    a = single_error(0, True, structE, tau, ntau, n_ka, dtfrac)
    b = single_error(0, False, structE, tau, ntau, n_ka, dtfrac)
    c = single_error(max_field, True, structE, tau, ntau, n_ka, dtfrac)
    d = single_error(max_field, False, structE, tau, ntau, n_ka, dtfrac)
    err = max(a, b, c, d)
    return err


def compute_Bsweep(fields, structE, tau):
    """ Get sigma_bb and sigma_aa for this field *sweep*. muOhmcm.

    Chooses convergence criteria and performes error analysis for you
    """

    # 101 is very high.
    # The reason it is, is because the Nuss fs in particular
    # has an incredibly small MR.
    #
    # Please maintain odd/even if you change this or errors will result.
    dtfrac = 101
    ntau = 30
    n_ka = 51
    st = time.time()
    sbb0 = compute_sigma(True, fields[0], structE, tau, n_ka=n_ka,
                         dtfrac=dtfrac, ntau=ntau)
    print(f'> Expect {(3 * len(fields) + 10) * (time.time() - st):.0f} s compute time')

    err = analyse_errors(max(fields), structE, tau, ntau, n_ka, dtfrac)
    sbb = [sbb0] + [compute_sigma(True, f, structE, tau, n_ka=n_ka,
                                  dtfrac=dtfrac, ntau=ntau) for f in fields[1:]]
    saa = [compute_sigma(False, f, structE, tau, n_ka=n_ka,
                         dtfrac=dtfrac, ntau=ntau) for f in fields]
    sab = [compute_sigma(2, f, structE, tau, n_ka=n_ka,
                         dtfrac=dtfrac, ntau=ntau) for f in fields]

    print(f'> Computation took {(time.time() - st) / len(fields):.3f} s/point'
          f' ({time.time() - st:.1f} s).')
    return np.array(sbb), np.array(saa), np.array(sab), err
