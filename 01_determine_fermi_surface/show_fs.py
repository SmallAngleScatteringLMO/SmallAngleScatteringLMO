# This code is used to estimate tight binding
# coefficients that reproduce the Fermi surface
# of Merino2012 and Nuss2014

import numpy as np
import matplotlib.pyplot as plt
import numba
import os

plt.rc('font', size=25)

# Lattice parameters, see Popovic2006
A = 9.499e-10
B = 5.523e-10
C = 12.762e-10
E = 1.602176634e-19
KB = 1.380649e-23
HBAR = 1.054571817e-34


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

#############################
# Dispersion
# Plotting
#############################


def show_energy(structE, *, axis=None):
    """ Show the energy dispersion. """

    if axis is None:
        _, axis = plt.subplots()

    xx = np.linspace(-np.pi / B, np.pi / B, 100)
    yy = np.linspace(-np.pi / A, np.pi / A, 100)
    X, Y = np.meshgrid(xx, yy)
    EE = compute_energy(X, Y, structE)

    p = axis.pcolor(X, Y, EE)
    axis.set_xlabel('$k_b$ (1/\u212B)')
    axis.set_ylabel('$k_a$ (1/\u212B)')
    plt.colorbar(p, label='$\epsilon$ (eV)')

    return axis


def show_FS(structE, conj, *, axis=None):
    """ Plot the Fermi surfaces. """

    if axis is None:
        _, axis = plt.subplots()

    kcc = np.linspace(-np.pi / A, np.pi / A, 1000)
    kbb = np.array([get_kb(kc, structE) for kc in kcc])
    axis.plot(kbb, kcc, color='red')
    axis.plot(-kbb, kcc, color='red')

    kbb = np.array([get_kb(kc, conj) for kc in kcc])
    axis.plot(kbb, kcc, color='black')
    axis.plot(-kbb, kcc, color='black')


def download(path):
    """ Get the data in this file. Column-major. """

    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
        for line in f:
            line = line.replace(',', '')
            try:
                float(line.split()[0])
                break
            except Exception:
                pass

        data = [[float(i) for i in line.split()]]
        for line in f:
            line = line.replace(',', '')
            data.append([float(i) for i in line.split()])

    return np.array(data).T


def show_data_nuss(axis):
    """ Digitized data. Artificially double a axis. """

    file = os.path.abspath('10_data_nuss.dat')
    # This data is in 1/Angstrom
    x, y = download(file)
    axis.set_ylim(*axis.get_ylim())
    axis.scatter(x * 1e10, y * 2e10, color='tab:cyan')
    axis.scatter(x * 1e10, y * -2e10, color='tab:cyan')
    axis.scatter(x * -1e10, y * -2e10, color='tab:cyan')
    axis.scatter(x * -1e10, y * 2e10, color='tab:cyan')


def show_data_merino(axis):
    """ Digitized data. """

    file = os.path.abspath('01_merino2012_splitting.txt')
    y, x = download(file)
    y = 1 - y

    # Use that the plot limits delimit the BZ
    ylim = axis.get_ylim()
    xlim = axis.get_xlim()
    axis.set_ylim(*ylim)
    x *= xlim[1]
    y *= ylim[1]

    axis.scatter(x, y, color='tab:cyan')
    axis.scatter(x, -y, color='tab:cyan')
    axis.scatter(-x, y, color='tab:cyan')
    axis.scatter(-x, -y, color='tab:cyan')


#############################
# Plotting
# User Input
#############################


print('Welcome to the RTA calculation')
choice = '0'
while choice not in ['1', '2', '3']:
    choice = input('Enter 1 for Merino, 2 for Nuss, 3 for anisotropic-fs: ')

#############################
# User Input
# Fermi surface
#############################

# Merino based fs
if choice == '1':
    structE = tuple([0.5 / 8, 0.036 / 8, -0.02])
    conj = tuple([structE[0], -structE[1], structE[2] + 0.02])
    print('Showing the Merino-based Fermi surface')

# Nuss based (Referee)
elif choice == '2':
    structE = tuple([0.0625, 0.003, -0.0075])
    conj = tuple([0.0625, -0.00038, 0.00088])
    print('Showing the Nuss-based Fermi surface')

# Artificial (large MR, excessive anisotropy - RTA vs SAS check)
elif choice == '3':
    structE = tuple([0.5, 0.286, -0.26])
    structE = tuple([s / 8 for s in structE])
    conj = tuple([structE[0], -structE[1], structE[2] + 0.085])
    print('Showing the special high-anisotropy Fermi surface')

#############################
# Fermi surface
# Compute properties
#############################

print('Compare these splittings to published figures:')
kb1 = get_kb(np.pi / A, structE)
kb2 = get_kb(np.pi / A, conj)
print(f'Splitting at BZ edge: {abs(kb1 - kb2) / np.pi * B}')

kb1 = get_kb(0, structE)
kb2 = get_kb(0, conj)
print(f'Splitting at Gamma: {abs(kb1 - kb2) / np.pi * B}')


print()
kb1 = get_kb(0, structE)
vG1 = compute_velocity_b(kb1, 0, structE)
kb2 = get_kb(np.pi / A, structE)
vG2 = compute_velocity_b(kb2, np.pi / A, structE)
print(f'sheet1: vb varies between {vG1:.0f} and {vG2:.0f} m/s')
kb1 = get_kb(0, conj)
vG1 = compute_velocity_b(kb1, 0, conj)
kb2 = get_kb(np.pi / A, conj)
vG2 = compute_velocity_b(kb2, np.pi / A, conj)
print(f'sheet2: vb varies between {vG1:.0f} and {vG2:.0f} m/s')

n1 = compute_density(structE)
n2 = compute_density(conj)
print(f'Densities are {2 * n1:.2f} + {2 * n2:.2f} = '
      f'{2 * n1 + 2 * n2:.2f} filling (max = 2, spin)')

#############################
# Compute properties
# Show
#############################

axis = show_energy(structE)
show_FS(structE, conj, axis=axis)

if choice == '1':
    show_data_merino(axis)
elif choice == '2':
    show_data_nuss(axis)
plt.show()
