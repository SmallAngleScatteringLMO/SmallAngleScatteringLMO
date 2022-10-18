import numpy as np
import matplotlib.pyplot as plt
import numba

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
def get_ka_values(nr_ka: int):
    """ Returns nr_ka ka values, each will have a
    positive and negative sheet point associated with it. """

    # No endpoint
    # The reason is that the two ends are identical,
    # connected by the BZ periodicity.
    #
    # But numba does not support the endpoint=False keyword
    # so do it yourself.
    kaa = np.linspace(-np.pi / A, np.pi / A, nr_ka + 1)
    return kaa[:-1]


@numba.njit()
def compute_velocities_bb(nr_ka, structE, double=False):
    """ Get vbb and vaa. Each an array of one/both sheets. """

    length = 2 * nr_ka if double else nr_ka
    vbb = np.zeros(length)
    kaa = get_ka_values(nr_ka)
    for i, ka in enumerate(kaa):
        kb = get_kb(ka, structE)
        vb = compute_velocity_b(kb, ka, structE)
        vbb[i] = vb
        if double:
            vbb[i + nr_ka] = -vb
    return vbb


@numba.njit()
def compute_velocities_aa(nr_ka, structE, double=False):
    """ Get vaa. Either one or both (double) sheet. """

    length = 2 * nr_ka if double else nr_ka
    vaa = np.zeros(length)
    kaa = get_ka_values(nr_ka)
    for i, ka in enumerate(kaa):
        kb = get_kb(ka, structE)
        va = compute_velocity_a(kb, ka, structE)
        vaa[i] = va
        if double:
            vaa[i + nr_ka] = va
    return vaa


@numba.njit()
def compute_velocity_a(kb, ka, structE):
    return 2 * structE[1] * np.sin(ka * A) * A * E / HBAR


@numba.njit()
def compute_density(structE):
    """ Get the carrier density as fraction of a full BZ of this band. """

    # Formula is
    # n = 1/8pi^3 * 2pi/C * area_encl * 2(spin) * 2(deg)
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
    """ Get the carrier density as a number per cubic meter. """
    return 2 * compute_density(structE) / (A * B * C)


#################
# Electronic structure
# Cyclotron motion
#################
# This is where the code diverges from the relaxation time version


@numba.njit()
def make_state(nr_ka: int, ka: float, positive: bool):
    """ The only place allowed to make the state.

    nr_ka: int
        number of points on *one* FS sheet.
        odd or even
    ka: float
        where the state is located.
        Must be a part of the 'get_ka_values' set
    positive: bool
        whether this is on the positve or negative kb sheet.

    Each index represents a probability to be there at this time.
    Normalisation follows integrate_kc state = 1
    where integrate runs over both sides of the FS.

    Returns a double-nr_ka length state vector.

    The state follows ka=-pi/A to (almost) pi/A in nr_ka points,
    then goes back to -pi/A to pi/A but with negative kb.
    """
    assert(nr_ka > 3)

    state = np.zeros(2 * nr_ka)
    kaa = get_ka_values(nr_ka)
    closest = np.argmin(np.abs(kaa - ka))

    if abs(kaa[closest] - ka) > (kaa[1] - kaa[0]) / 100:
        assert(False and "Use a ka value from get_ka_values")
    closest += 0 if positive else nr_ka
    state[closest] = 1 / (kaa[1] - kaa[0])
    return state


@numba.njit()
def compute_norm(state):
    """ Calculate the total norm of this state.

    Note that states just made will have norm 1.
    Note that you are free to multiply this by anything,
    it performs a kc integral over both FS sheets.
    Trapezoidal rule.
    """

    # Trapezoidal rule
    # However, these sheets are both periodic
    # that is why kaa lacks the endpoint, because the two are the same.
    # Hence the two coefficients 1 merge into a single coefficient 2 at the start.
    # This leaves all 2's.
    #
    # Even though ka has no endpoint, it is implicitly joined with the first
    # point, therefore the spacing is the full 2pi/A/nka not 2pi/A/(nka-1)
    dkaa = 4 * np.pi / A / len(state)
    return np.sum(state) * dkaa


@numba.njit()
def calculate_BZ_time(max_field, structE, n=501):
    """ Calculate the how long it takes at this magnetic field to
    cross a Fermi sheet. """

    # Compute it at 1 T
    vbb = compute_velocities_bb(n, structE)
    dkadt = E * vbb * 1 / HBAR

    # Effectively use trapezoidal rule to average dka/dt
    timer = 2 * np.pi / A * np.sum(1 / dkadt) / n
    return timer / abs(max_field)


@numba.njit()
def calculate_D(tau):
    """ Given an aimed for lifetime, get the diffusion constant.

    Backed up by old analytical results from impeded orbital motion
    and empirical tests in file 01_show_diffusion, which is
    incorporated in the unittests.
    """
    if tau == 0:
        return 0
    return 1 / tau / A**2


@numba.njit()
def calculate_tau(D):
    """ Given a diffusion constant, get the effective lifetime.

    Backed up by old analytical results from impeded orbital motion
    and empirical tests in file 01_show_diffusion, which is
    incorporated in the unittests.
    """
    return 1 / D / A**2


@numba.njit()
def calculate_bs(meanL, structE):
    """ Get the average lifetime.
    Really <1/tau> or the average scattering rate.
    """

    vbb = compute_velocities_bb(50, structE)
    rates = meanL / vbb
    return np.mean(rates)

#################
# Cyclotron motion
# Matrix M
#################
# This is where the code diverges from the relaxation time version


def _make_off_diag_matrix(dim, offset, wrap):
    """ Create a diagonal 'offset' above the standard.
    0 gives the identity matrix.
    1 is just above the diagonal.

    If wrap is set, then the diagonal continues as it goes through the edge.
    So for offset 1, it also sets the bottomleft corner to 1.
    If wrap it set, there are always 'dim' elements set to 1.

    The effect of off=1 is to map j+1 rhs onto j lhs.
    """

    if offset == 0:
        return np.identity(dim)
    matrix = np.zeros((dim, dim))

    part1 = np.identity(dim - abs(offset))
    matrix[:-abs(offset), abs(offset):] = part1
    if wrap:
        part2 = np.identity(abs(offset))
        matrix[-abs(offset):, :abs(offset)] = part2
    if offset < 0:
        matrix = matrix.T
    return matrix


def _make_patterned_off_diag_matrix(elements, offset):
    """ Make a wrapping off-diag matrix, but each element is given
    by the sequence rather than set to 1."""

    matrix = np.zeros((len(elements), len(elements)))
    for i, el in enumerate(elements):
        matrix[i, (i + offset) % len(elements)] = el
    return matrix


def _make_second_derivative(elements):
    """ Encode a second derivative following the fully implicit method. """

    n = len(elements)
    base = np.zeros((n, n))
    for i, el in enumerate(elements):
        base[i, i] = -2 * el
        base[i, (i + 1) % n] = el
        base[i, (i - 1) % n] = el
    return base


def _make_diff_interaction(elements):
    """ Make a diffusion-type matrix which encodes the second derivative
    in the fully implicit method. The strength of the derivative
    varies according to elements. The result is 2*elements long,
    for both Fermi sheets the same. No inter-sheet interactions.
    Each sheet is cyclic.
    """

    n = len(elements)
    base = _make_second_derivative(elements)

    full = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            full[i, j] = base[i, j]
            full[i + n, j + n] = base[i, j]
    assert(abs(np.sum(full)) <= np.max(np.abs(elements)) / 1e10)
    return full


def _make_first_derivative(elements, is_forward):
    """ Encode a first derivative following the fully implicit method.
    You want forward when the derivative du/dk is positive.

    You can do either (j) - (j-1) or (j+1) - (j) or 0.5(j+1 - j-1).
    Numerical Recipes ch19.1: important for discontinuities.
    Here: start diffusion on a delta function... Important.

    Option 1) forward [flipped from intuition because implicit]
    Option 2) backward
    Option 3) always bad

    When the option is not right, the result is negative weight
    and >1 weight on other places.
    """

    n = len(elements)
    base = np.zeros((n, n))
    sgn = 1 if is_forward else -1
    for i, el in enumerate(elements):
        base[i, i] = sgn * el
        base[i, (i - sgn) % n] = -sgn * el

    assert(np.abs(np.sum(base)) <= np.abs(np.min(elements) / 1e5))
    return base


def _make_cyclotron_interaction(elements):
    """ Return a 2nr_ka size matrix with off-diagonal cyclotron-type
    interactions.

    Automatically makes this upwind
    Requires ALL elements positive or ALL negative.
    I.e. there is no point where cyclotron motion reverts direction
    (other than between disconnected sheets)
    """

    assert(max(elements) * min(elements) >= 0)
    # Forward or not does not matter when 0, the result will be 0.
    forward = max(elements) >= 0
    nr_ka = len(elements)
    pos = _make_first_derivative(elements, forward)
    # I need to add the sign AND the not-forward.
    # The not-forward makes the derivative consider (j-1) and j
    # the sign makes the cyclotron motion truly backwards.
    neg = -_make_first_derivative(elements, not forward)

    full = np.zeros((2 * nr_ka, 2 * nr_ka))
    for i in range(nr_ka):
        for j in range(nr_ka):
            full[i, j] = pos[i, j]
            full[i + nr_ka, j + nr_ka] = neg[i, j]
    assert(abs(np.sum(full)) <= abs(np.max(np.abs(elements))) / 1e10)
    return full


def construct_diag(elements):
    """ Make a diagonal matrix with these values. """

    matrix = np.zeros((len(elements), len(elements)))
    for i, el in enumerate(elements):
        matrix[i, i] = el
    return matrix


def _make_backscattering_interaction(mean_free_path, vbb, bs_type):
    """ Make the backscattering term. This is dominated by a
    mean free path, hence the rate is L/v_b.

    mean free path 0 is interpreted as infinity

    bs_type: 1 -> flip kb, isotropic L [default, impurity picture]
             2 -> flip kb, isotropic tau [no MR]
             3 -> flip ka and kb, isotropic L
             4 -> go everywhere, isotropic Umklapp
             5 -> amplitude non-preserving decay ~RTA imposter
    """

    nr_ka = len(vbb)
    if mean_free_path > 0:
        # Factor 2 because there are two elements in the
        # matrix that take away from a particular state
        # and this way the total rate is vbb/L per
        # unit time.

        # This is isotropic L
        if bs_type == 1 or bs_type == 3:
            rates = -np.abs(vbb) / mean_free_path / 2

        # This is isotropic tau:
        elif bs_type == 2:
            rates = -np.ones(len(vbb)) * 0.5 * np.mean(vbb) / mean_free_path

        # This is isotropic L, but scattering *everywhere*
        # Factor 2 for both above and below diagonal,
        # another factor 2 because the matrix is filled all-to-all (incl backwards)
        elif bs_type == 4:
            rates = -np.abs(vbb) / mean_free_path / 4 / (nr_ka - 1)

        # Isotropic tau RTA-like exponential decay
        elif bs_type == 5:
            rates = -np.ones(len(vbb)) * np.mean(vbb) / mean_free_path
        else:
            raise ValueError('Unknown backscattering type')
    else:
        rates = 0 * vbb

    full = np.zeros((2 * nr_ka, 2 * nr_ka))

    # Flip kb only
    if bs_type in [1, 2]:
        for i in range(nr_ka):
            full[i, i] = -rates[i]
            full[i + nr_ka, i] = rates[i]
            full[i, i + nr_ka] = rates[i]
            full[i + nr_ka, i + nr_ka] = -rates[i]
    elif bs_type in [3]:
        for i in range(nr_ka):
            full[i, i] = -rates[i]
            full[2 * nr_ka - i - 1, i] = rates[i]
            full[i, 2 * nr_ka - i - 1] = rates[i]
            full[2 * nr_ka - i - 1, 2 * nr_ka - i - 1] = -rates[i]
    elif bs_type in [4]:
        for i in range(2 * nr_ka):
            for j in range(2 * nr_ka):
                full[i, i] -= rates[i % nr_ka]
                full[i, (i + j) % (2 * nr_ka)] += rates[i % nr_ka]
                full[(i + j) % (2 * nr_ka), i] += rates[i % nr_ka]
                full[(i + j) % (2 * nr_ka), (i + j) % (2 * nr_ka)] -= rates[i % nr_ka]
    elif bs_type == 5:
        for i in range(nr_ka):
            full[i, i] -= rates[i]
            full[i + nr_ka, i + nr_ka] -= rates[i]

    else:
        raise ValueError('Unknown backscattering type')

    if bs_type != 5:
        # Basically 5 is known to violate charge conservation
        # The rest should not.
        assert(abs(np.sum(full)) <= np.mean(np.abs(rates)) / 1e6)
    return full


def make_invM(D, field, backL, dt, nr_ka, structE, bs_type=1):
    """ Create the time evolution matrix (inverse of M) following the
    implicit method for the differential equation. See word. """

    # See word document `03 diffusion' for derviation.
    kaa = get_ka_values(nr_ka)
    dka = kaa[1] - kaa[0]
    vva_pos = compute_velocities_aa(nr_ka, structE)
    vvb_pos = compute_velocities_bb(nr_ka, structE)

    # The base is that the state stays the same.
    # This identity forms the whole of np.sum(M),
    # which is unmodified by the remaining terms such that
    # charge is conserved.
    matrix = np.zeros((2 * nr_ka, 2 * nr_ka))
    matrix += np.identity(2 * nr_ka)

    diff = -D / (1 + vva_pos**2 / vvb_pos**2) / dka**2 * dt
    matrix += _make_diff_interaction(diff)

    cycl = E / HBAR * vvb_pos * field / dka * dt
    matrix += _make_cyclotron_interaction(cycl)
    matrix += _make_backscattering_interaction(backL, vvb_pos, bs_type) * dt

    return np.linalg.inv(matrix)


#################
# Matrix M
# RK4 matrix
#################

def _make_col_second_derivative(elements):
    """ Encode a second derivative following the fully implicit method. """

    n = len(elements)
    base = np.zeros((n, n))
    for i, el in enumerate(elements):
        base[i, i] -= 2 * el
        base[(i + 1) % n, i] += el
        base[(i - 1) % n, i] += el
    return base

def _make_RK4_diff_interaction(elements):
    """ Given a set of interaction strengths, put them in
    the right matrix shape to get diffusion. """

    n = len(elements)
    base = _make_col_second_derivative(elements)
    assert(abs(np.sum(base)) <= np.max(np.abs(elements)) / 1e10)

    full = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            full[i, j] = base[i, j]
            full[i + n, j + n] = base[i, j]

    assert(abs(np.sum(full)) <= np.max(np.abs(elements)) / 1e10)
    return full


def _make_col_first_derivative(elements, is_forward):
    """ See the other one. But now columns sum to zero.
    """

    n = len(elements)
    base = np.zeros((n, n))
    sgn = 1 if is_forward else -1
    for i, el in enumerate(elements):
        base[i, i] = sgn * el
        base[(i + sgn) % n, i] = -sgn * el

    assert(np.abs(np.sum(base)) <= np.abs(np.min(elements) / 1e5))
    return base


def _make_RK4_cyclotron_interaction(elements):
    """ Return a 2nr_ka size matrix with off-diagonal cyclotron-type
    interactions.

    Automatically makes this upwind
    Requires ALL elements positive or ALL negative.
    I.e. there is no point where cyclotron motion reverts direction
    (other than between disconnected sheets)
    """

    assert(max(elements) * min(elements) >= 0)
    # Forward or not does not matter when 0, the result will be 0.
    forward = max(elements) >= 0
    nr_ka = len(elements)
    pos = _make_col_first_derivative(elements, forward)
    # I need to add the sign AND the not-forward.
    # The not-forward makes the derivative consider (j-1) and j
    # the sign makes the cyclotron motion truly backwards.
    neg = -_make_col_first_derivative(elements, not forward)

    full = np.zeros((2 * nr_ka, 2 * nr_ka))
    for i in range(nr_ka):
        for j in range(nr_ka):
            full[i, j] = pos[i, j]
            full[i + nr_ka, j + nr_ka] = neg[i, j]

    assert(abs(np.sum(full)) <= abs(np.max(np.abs(elements))) / 1e10)
    return full

def _make_fast_RK4_cycl(n, field, dka, structE):
    """ Return a 2nr_ka size matrix with off-diagonal cyclotron-type
    interactions.

    Automatically makes this upwind

    Requires ALL positive or ALL negative velocities on the positive-kb sheet
    I.e. there is no point where cyclotron motion reverts direction
    (other than between disconnected sheets)

    Unlike the other version, this one uses RK4.
    That is to say, instead of the derivative at a particular k point
    to model how much moves to the next one, use RK4 to get a better
    estimate of the derivative over a finite jump in k averaging vb
    in a special fashion.
    """

    full = np.zeros((2 * n, 2 * n))

    # RK4, following the wikipedia convention,
    # here has t -> k, y -> t. The differential is t-independent,
    # so I know ahead of time I need vb exactly at halfway points
    vvb = compute_velocities_bb(n * 2, structE)


    # Forward gives me positive on the diagonal
    # and negative 1 below the diagonal.
    #
    # Backward gives me negative on the diagonal
    # and positive 1 above the diagonal.
    forward = max(vvb) >= 0

    # One sheet is forward (k and k+dk), the other backward (k-dk and k).
    # So you always need both types.
    off = 0 if forward else n
    for i in range(n):
        c1 = E / HBAR * vvb[2 * i] * field / dka
        c2 = E / HBAR * vvb[2 * i + 1] * field / dka
        c3 = E / HBAR * vvb[2 * i + 1] * field / dka
        c4 = E / HBAR * vvb[(2 * i + 2) % (2 * n)] * field / dka
        cycl = 1 / 6 * (c1 + 2 * c2 + 2 * c3 + c4)  # RK4
        # cycl = 1/2 * (c1 + c4)  # Trapezoid

        full[i + off, i + off] = cycl
        full[(i + 1) % n + off, i + off] = -cycl

    off = n if forward else 0
    for i in range(n):
        c1 = E / HBAR * vvb[(2 * i - 2) % (2 * n)] * field / dka
        c2 = E / HBAR * vvb[(2 * i - 1) % (2 * n)] * field / dka
        c3 = E / HBAR * vvb[(2 * i - 1) % (2 * n)] * field / dka
        c4 = E / HBAR * vvb[2 * i] * field / dka
        cycl = 1 / 6 * (c1 + 2 * c2 + 2 * c3 + c4)  # RK4
        # cycl = 1/2 * (c1 + c4)  # Trapezoid

        full[i + off, i + off] = cycl
        full[(i - 1) % n + off, i + off] = -cycl

    return full


def _turn_RK4(M, dt):
    """ Given the matrix u'=Mu, return the finite-time RK4
    form of this matrix. """

    iden = np.identity(len(M))

    k1 = M
    k2 = np.matmul(M, iden + 0.5 * dt * k1)
    k3 = np.matmul(M, iden + 0.5 * dt * k2)
    k4 = np.matmul(M, iden + dt * k3)
    return iden + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def deriv_RK4(D, field, backL, nr_ka, structE, *, bs_type=1):
    """ Create the derivative operator u'=Mu

    See _make_backscattering_interaction for the meaning of bs_type
    """

    # See word document `04 diffusion faster' for derviation.
    kaa = get_ka_values(nr_ka)
    dka = kaa[1] - kaa[0]
    vva_pos = compute_velocities_aa(nr_ka, structE)
    vvb_pos = compute_velocities_bb(nr_ka, structE)

    # The base is that the state stays the same.
    # This identity forms the whole of np.sum(M),
    # which is unmodified by the remaining terms such that
    # charge is conserved.
    matrix = np.zeros((2 * nr_ka, 2 * nr_ka))

    diff = D / (1 + vva_pos**2 / vvb_pos**2) / dka**2
    matrix += _make_RK4_diff_interaction(diff)
    matrix -= _make_fast_RK4_cycl(nr_ka, field, dka, structE)
    matrix -= _make_backscattering_interaction(backL, vvb_pos, bs_type)

    # Print the matrix - do this for low nr_ka debugging
    # print()
    # print()
    # for i in range(n):
    #     string = ''
    #     for j in range(n):
    #         string += f' {op[i][j]:.5e}'
    #     string += '  '
    #     for j in range(n):
    #         string += f' {op[i][j + n]:.5e}'
    #     print(string)
    # print()
    # for i in range(n):
    #     string = ''
    #     for j in range(n):
    #         string += f' {op[i + n][j]:.5e}'
    #     string += '  '
    #     for j in range(n):
    #         string += f' {op[i + n][j + n]:.5e}'
    #     print(string)
    # print()
    # print()


    if bs_type != 5:
        assert(abs(np.sum(matrix)) < np.max(matrix) / 1e8)
    return matrix


def make_RK4(deriv, dt):
    """ Create the time evolution operator from the derivative. """

    RK4 = _turn_RK4(deriv, dt)
    if np.max(RK4) > 1:
        # Strict larger
        raise ValueError('dt too long, explicit method divergent.')

    return RK4

#################
# RK4 matrix
# Conductivity
#################


@numba.njit()
def time_evolve(op, state):
    """ Do 1 step, advancing time by dt, ensuring positive state """
    state = op @ state
    assert(np.min(state) >= 0)
    return state


@numba.njit()
def compute_vcorr(op, state, dt, steps, structE):
    """ Calculate the integrated va and vb over time.

    integral dt_0^infty sum_sheets integral dka'_-pi/A^pi/A  v_a(ka',t) A(ka',t)
    infty -> dt * steps
    Perform this for va and vb

    Assumes state is at time 0.
    Simpson for time, trapezoidal for ka.
    """

    assert(len(state) % 2 == 0)
    assert(len(state) == len(op))
    assert(steps > 2)
    assert(dt > 0)

    # There were issues here because of floats being entered
    # ending up with even values of steps that invalidate the algo.
    steps = int(steps)
    steps = int(steps + steps % 2 + 1)
    assert(steps % 2)

    n = len(state) // 2
    vaa = compute_velocities_aa(n, structE, True)
    vbb = compute_velocities_bb(n, structE, True)

    total_va = 0
    total_vb = 0
    # The loop is the t integral
    for i in range(steps - 1):
        # 1 4 2 4 2 ... 4
        w = 1 if i == 0 else 2 * (i % 2) + 2

        # Compute the ka' integral and sheet sum here
        va = compute_norm(vaa * state)
        vb = compute_norm(vbb * state)

        total_va += w * va
        total_vb += w * vb

        state = time_evolve(op, state)

    assert(w == 4)

    # and the final weight 1
    va = compute_norm(vaa * state)
    vb = compute_norm(vbb * state)
    total_va += va
    total_vb += vb
    total_va *= dt / 3
    total_vb *= dt / 3
    return total_va, total_vb, state


# @numba.njit()
def sigma(deriv, dt, steps, structE):
    """ Compute the conductivity given the state evolution
    operator invM (incl field, scattering) and the Fermi surface structE.

    Allows for multiple dt and steps (array).
    Use the first dt for the first steps,
    etc until both run out.
    """

    op = make_RK4(deriv, dt)
    n = len(op) // 2

    total_aa = 0
    total_ab = 0
    total_bb = 0

    kaa = get_ka_values(n)
    kbb = np.array([get_kb(ka, structE) for ka in kaa])
    positive = True
    # st = time.time()
    for i, (ka, kb) in enumerate(zip(kaa, kbb)):
        kb = kb if positive else -kb
        va0 = compute_velocity_a(kb, ka, structE)
        vb0 = compute_velocity_b(kb, ka, structE)

        state = make_state(n, ka, positive)
        corrA, corrB, state = compute_vcorr(op, state, dt, steps, structE)

        # Trapezoid
        # First and last point are merged.
        # Factor 2 for sheets
        DoS_b = 1 / HBAR / abs(vb0)
        total_aa += va0 * corrA * DoS_b * 2
        total_bb += vb0 * corrB * DoS_b * 2
        total_ab += vb0 * corrA * DoS_b * 2

    const = E**2 / (2 * np.pi**2 * C)
    total_aa *= (kaa[1] - kaa[0]) * const
    total_ab *= (kaa[1] - kaa[0]) * const
    total_bb *= (kaa[1] - kaa[0]) * const

    return total_bb, total_aa, total_ab


def esigma(D, field, backL, nr_ka, dt1, steps1, structE, dt2, steps2):
    """ Get sbb, saa, sab, dsbb, dsaa, dsab

    This costs about 2.5x what sigma does.
    Evaluate sigma
    Then evaluate half steps, half nr_ka, double dt
    """

    op = make_invM(D, field, backL, dt1, nr_ka, structE)
    sbb, saa, sab = sigma(op, dt1, steps1, structE, dt2, steps2)

    reason = []
    bb_err = []
    aa_err = []
    ab_err = []

    op = make_invM(D, field, backL, dt1, nr_ka // 2, structE)
    sbb2, saa2, sab2 = sigma(op, dt1, steps1, structE, dt2, steps2)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('number of ka')

    op = make_invM(D, field, backL, dt1, nr_ka, structE)
    sbb2, saa2, sab2 = sigma(op, dt1, steps1 // 2, structE, dt2, steps2)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('number of steps1')

    sbb2, saa2, sab2 = sigma(op, dt1, steps1, structE, dt2, steps2 // 2)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('number of steps2')

    sbb2, saa2, sab2 = sigma(op, dt1, steps1, structE, dt2 * 2, steps2 // 2)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / sab)
    reason.append('dt2')

    op = make_invM(D, field, backL, dt1 * 2, nr_ka, structE)
    sbb2, saa2, sab2 = sigma(op, dt1 * 2, steps1 // 2, structE, dt2, steps2)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('dt1')

    leading = np.argmax(bb_err)
    print(f'Leading sbb error is {reason[leading]}: {bb_err[leading]:.1e}')
    for reas, err in zip(reason, bb_err):
        print(f'  > {err:.1e} {reas}')
    dsbb = bb_err[leading]

    leading = np.argmax(aa_err)
    print(f'Leading saa error is {reason[leading]}: {aa_err[leading]:.1e}')
    for reas, err in zip(reason, aa_err):
        print(f'  > {err:.1e} {reas}')
    dsaa = aa_err[leading]

    leading = np.argmax(ab_err)
    print(f'Leading sab error is {reason[leading]}: {ab_err[leading]:.1e}')
    for reas, err in zip(reason, ab_err):
        print(f'  > {err:.1e} {reas}')
    dsab = ab_err[leading]

    return sbb, saa, sab, dsbb, dsaa, dsab

#################
# Conductivity
# Adaptive conductivity
#################


def adaptive_vcorr(deriv, state, t_end, err_aim, dt0fact, structE):
    """ Calculate the integrated va and vb over time.

    integral dt_0^infty sum_sheets integral dka'_-pi/A^pi/A  v_a(ka',t) A(ka',t)
    infty -> dt * steps
    Perform this for va and vb

    Assumes state is at time 0.
    Adaptive Simpson for time, trapezoidal for ka, explicit RK4 time evolution

    deriv: see deriv_RK4()
    state: see make_state()
    dt: timestep at lowest times. Will increase manyfold with time.
        Choose 'sufficiently small'
    t_end: time to stop integral.
        Typically ~20 tau for the slower of diffusion / backscattering
    err_aim: float, relative error to aim for
    dt0frac: go for 5
        determines ratio between maximum dt for RK and actual.
        Around 2 nothing is detected, this is O(dt^4) so take 5.
        A parameter so you can quantify the error of it.
    structE: energy parametrization.
    """

    assert(len(state) % 2 == 0)
    assert(len(state) == len(deriv))

    # Enable to see how the integral converges
    # and to see the final distribution across the FS.
    TEST = False

    n = len(state) // 2
    vaa = compute_velocities_aa(n, structE, True)
    vbb = compute_velocities_bb(n, structE, True)

    total_va = 0
    total_vb = 0
    t = 0
    dt = 1 / abs(deriv[0][0]) / dt0fact

    v0a = compute_norm(vaa * state)
    v0b = compute_norm(vbb * state)
    op = make_RK4(deriv, dt)
    iters = 0

    if TEST:
        caa = []
        cbb = []
        tt = []

    # The loop is the t integral
    while t < t_end:
        # Loop unroll the next 4 states
        state1 = time_evolve(op, state)
        v1a = compute_norm(vaa * state1)
        v1b = compute_norm(vbb * state1)

        state2 = time_evolve(op, state1)
        v2a = compute_norm(vaa * state2)
        v2b = compute_norm(vbb * state2)

        state3 = time_evolve(op, state2)
        v3a = compute_norm(vaa * state3)
        v3b = compute_norm(vbb * state3)

        state4 = time_evolve(op, state3)
        v4a = compute_norm(vaa * state4)
        v4b = compute_norm(vbb * state4)

        # Compute with given and double timestep
        # using Simpson's method
        contribution_a = v0a + 4 * v1a + 2 * v2a + 4 * v3a + v4a
        contribution_a *= dt / 3
        total_va += contribution_a
        contribution_b = v0b + 4 * v1b + 2 * v2b + 4 * v3b + v4b
        contribution_b *= dt / 3
        total_vb += contribution_b

        # alternative_a = v0a + 4 * v2a + v4a
        # alternative_a *= 2 * dt / 3
        # err_a = abs((alternative_a - contribution_a) * iters / total_va)
        alternative_b = v0b + 4 * v2b + v4b
        alternative_b *= 2 * dt / 3
        err_b = abs((alternative_b - contribution_b) * iters / total_vb)

        state = state4
        v0a = v4a
        v0b = v4b
        t += 4 * dt

        # Initially used erra check as well ('and')
        # but with field=0 there are points where va=0
        # and remains 0 with diffusion. This leads to enormous
        # execution time (stick to dt0 forever, easily >100x)
        if err_b < err_aim and dt / t < 1 / 30 and iters > 5 and t < t_end / 10:
            dt *= 2
            op = np.matmul(op, op)

            # This is a numerical aberation.
            # Perhaps left best unperturbed, but it leads to charge loss
            # and I much prefer that to be stable.
            # No material effect on the value or error.
            #
            # The real issue is that the matrix after squaring for a few times
            # gets slight numerical errors, which lead to a stable position
            # at t->infty where the net <v> is non-zero.
            # That in turn leads to a dependence of the integral on t_end
            # and this error is actually leading.
            #
            # The PC available has no float80 or float128
            # and leaves no real room for improving this.
            # op *= 2 * n / np.sum(op)

        iters += 1

        if TEST:
            tt.append(t)
            caa.append(contribution_a / dt)
            cbb.append(contribution_b / dt)



    if TEST:
        plt.figure()
        plt.semilogy(tt, np.abs(caa))
        plt.semilogy(tt, np.abs(cbb))

        plt.figure()
        plt.plot(get_ka_values(n), state[:n])
        plt.scatter(get_ka_values(n), state[:n])
        # Put here a - for the x values to see it matches perfectly.
        plt.plot(get_ka_values(n), state[n:])
        plt.scatter(get_ka_values(n), state[n:])

        print(compute_norm(vaa * state))

        vff = np.sqrt(vaa**2 + vbb**2)
        plt.plot(get_ka_values(n), vbb[0]/vbb[:n] * state[0], color='black')
        plt.plot(get_ka_values(n), vff[0]/vff[:n] * state[0], color='black')

        plt.show()


    return total_va, total_vb, state, iters


def check_interval(nr_ka, fs_points):
    """ Get once how many state-ka values to take for ka-integral.

    There are nr_ka points per sheet for the state
    Aim for k_fs points per sheet for ka integral
    But the latter is restricted to points on the former
        and must be Simpson compatible [here: even]. """

    # Odd for Simpson, but first and last joined
    if fs_points % 2 == 1:
        raise ValueError('fs_points must be even')
    if nr_ka < fs_points:
        raise ValueError('nr_ka must be at least fs_points')
    if nr_ka // fs_points != nr_ka / fs_points:
        raise ValueError('nr_ka must be a multiple of fs_points')


def adaptive(deriv, t_end, err_frac, dt0fact, fs_points, structE):
    """ Now with adaptive simpsons over time
    and explict RK4 for time evolution.  """

    n = len(deriv) // 2
    check_interval(n, fs_points)

    total_aa = 0
    total_ab = 0
    total_bb = 0
    total_tevol = 0

    kaa = get_ka_values(fs_points)
    kbb = np.array([get_kb(ka, structE) for ka in kaa])
    for i, (ka, kb) in enumerate(zip(kaa, kbb)):
        # First point is 1, but merged with last point
        # Hence pattern is 2 4 2 4 ... 4
        w = 4 if i % 2 else 2

        va0 = compute_velocity_a(kb, ka, structE)
        vb0 = compute_velocity_b(kb, ka, structE)

        state = make_state(n, ka, True)
        corrA, corrB, state, iters = adaptive_vcorr(deriv, state, t_end,
                                             err_frac, dt0fact, structE)

        # Simpson
        # First and last point are merged.
        # Factor 2 for sheets
        DoS_b = 1 / HBAR / abs(vb0)
        total_aa += va0 * corrA * DoS_b * w * 2
        total_bb += vb0 * corrB * DoS_b * w * 2
        total_ab += va0 * corrB * DoS_b * w * 2
        total_tevol += iters

    assert(w == 4)

    const = E**2 / (2 * np.pi**2 * C)
    total_aa *= (kaa[1] - kaa[0]) * const / 3
    total_ab *= (kaa[1] - kaa[0]) * const / 3
    total_bb *= (kaa[1] - kaa[0]) * const / 3

    return total_bb, total_aa, total_ab, total_tevol


def eadaptive(D, field, backL, nr_ka, t_end, err_frac, dt0fact, fs_points, structE,*, bs_type=1):
    """ Get sbb, saa, sab, dsbb, dsaa, dsab

    This costs about 2.5x what sigma does.
    Evaluate sigma
    Then evaluate half steps, half nr_ka, double dt
    """

    check_interval(nr_ka, fs_points)

    deriv = deriv_RK4(D, field, backL, nr_ka, structE, bs_type=bs_type)
    sbb, saa, sab, iters = adaptive(deriv, t_end, err_frac, dt0fact, fs_points, structE)

    reason = []
    bb_err = []
    aa_err = []
    ab_err = []

    deriv = deriv_RK4(D, field, backL, nr_ka // 2, structE, bs_type=bs_type)
    try:
        sbb2, saa2, sab2, _ = adaptive(deriv, t_end, err_frac, dt0fact, fs_points, structE)
    except ValueError:
        print('>> NO nr_ka error analysis. Choos nr_ka/fs_points even.')
    else:
        bb_err.append(abs(sbb2 - sbb) / sbb)
        aa_err.append(abs(saa2 - saa) / saa)
        ab_err.append(abs(sab2 - sab) / abs(sab))
        reason.append('number of ka')

    deriv = deriv_RK4(D, field, backL, nr_ka, structE, bs_type=bs_type)
    try:
        sbb2, saa2, sab2, _ = adaptive(deriv, t_end, err_frac, dt0fact, fs_points // 2, structE)
    except ValueError:
        print('>> NO fs points error analysis. Choos fs_points divisible by 4.')
    else:
        bb_err.append(abs(sbb2 - sbb) / sbb)
        aa_err.append(abs(saa2 - saa) / saa)
        ab_err.append(abs(sab2 - sab) / abs(sab))
        reason.append('fs points')

    deriv = deriv_RK4(D, field, backL, nr_ka, structE, bs_type=bs_type)
    sbb2, saa2, sab2, _ = adaptive(deriv, t_end / 2, err_frac, dt0fact, fs_points, structE)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('t_end')

    deriv = deriv_RK4(D, field, backL, nr_ka, structE, bs_type=bs_type)
    sbb2, saa2, sab2, _ = adaptive(deriv, t_end, err_frac * 2, dt0fact, fs_points, structE)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('err_frac')

    deriv = deriv_RK4(D, field, backL, nr_ka, structE, bs_type=bs_type)
    sbb2, saa2, sab2, _ = adaptive(deriv, t_end, err_frac, dt0fact * 2, fs_points, structE)
    bb_err.append(abs(sbb2 - sbb) / sbb)
    aa_err.append(abs(saa2 - saa) / saa)
    ab_err.append(abs(sab2 - sab) / abs(sab))
    reason.append('dt0fact')


    leading = np.argmax(bb_err)
    print(f'  Leading REL sbb error is {reason[leading]}: {bb_err[leading]:.1e}')
    for reas, err in zip(reason, bb_err):
        print(f'   > {err:.1e} {reas}')
    dsbb = bb_err[leading]

    leading = np.argmax(aa_err)
    print(f'  Leading REL saa error is {reason[leading]}: {aa_err[leading]:.1e}')
    for reas, err in zip(reason, aa_err):
        print(f'   > {err:.1e} {reas}')
    dsaa = aa_err[leading]

    leading = np.argmax(ab_err)
    print(f'  Leading REL sab error is {reason[leading]}: {ab_err[leading]:.1e} [{sab:.1e}]')
    for reas, err in zip(reason, ab_err):
        print(f'   > {err:.1e} {reas}')
    dsab = ab_err[leading]
    print()

    return sbb, saa, sab, dsbb, dsaa, dsab, iters
