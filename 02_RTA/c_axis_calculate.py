# RTA extension
# Compute the influence of c-axis warping on MRa and MRb keeping H//c
# and *not* computing rho_cc.

import time
import os
import numpy as np
import core



print('Choose which Fermi surface to use')
choice = '0'
choice = input('Enter "1" for Merino, "2" for Nuss: ')
while choice not in ['1', '2']:
    print('Illegal entry.')
    choice = input('Enter "1" for Merino, "2" for Nuss: ')
merino = choice == '1'


###########################
# User Input
# Definitions
###########################



# Merino based
if merino:
    structE = tuple([0.5 / 8, 0.036 / 8, -0.02])
    conj = tuple([structE[0], -structE[1], structE[2] + 0.02])

# Nuss based
else:
    structE = tuple([0.0625, 0.003, -0.0075])
    conj = tuple([0.0625, -0.00038, 0.00088])

tau = 1e-14
print(f'Using tau={tau} [edit in the code]')
dtfrac = 50
print(f'Using dt=tau/{dtfrac} [edit in the code]')
ntau = 25
print(f'Cutting off the time integral at {ntau} * tau [edit in the code]')
n_ka = 37
print(f'Using {n_ka} points along ka [edit in the code]')

# nr_kc needs to be 4*n + 1 such that I can do
# Simpson using an odd number of kcc//2 + 1 points as well
# for error analysis.
# Else Simpson is ill defined and a large error ~1e-2
# is reported that is not representative.
n_kc = 37
print(f'Using {n_kc} points along kc [edit in the code]')

fields = list(np.linspace(0, 10, 40)) + list(np.linspace(11, 100, 25))
fields = np.array(fields)
print(f'Using {len(fields)} B-values between {min(fields)}-{max(fields)} [edit in the code]')

###########################
# Definitions
# Compute
###########################

# A shorthand way to compute this is to say that tc does not
# impact any of te a/b velocities or energies.
# This is because E = tb_a + tb_b + tb_c is additive
# and v = dE/dk also has no cross terms.
#
# Hence when we keep H//c and look at rho_aa, rho_bb
# there is a formal (i.e. NOT an approximation)
# equivalence between introducing c-axis warping
# and shifting the chemical potential as a function of k_z.
# Again, this step introduces *no* approximations.
#
# The reason we shall see this has essentially no effect is:
#   1) C-axis warping is small
#   2) There are no cross terms above
#   3) The dispersion is approximately k+k^3 at half filling
#       so v~1+k^2 has no linear term and chemical potential
#       shifts influence v only in second order.
#
# In the end, simply compute the conductivity for various chemical
# potential shifts along kz given by the tc warping.
# This simple method does exclude rho_cc from the results,
# but it suffices to quantify the influence of neglecting
# tc on rho_aa and rho_bb - and the answer this code gives is that this
# is totally negligible.
#
# tc = ta / 3.3 to get the anisotropy ratio for Rcc/Raa right.
Sbb = np.zeros(len(fields))
Saa = np.zeros(len(fields))
Sab = np.zeros(len(fields))
Sbb_check = np.zeros(len(fields))
Saa_check = np.zeros(len(fields))
Sab_check = np.zeros(len(fields))
kcc = np.linspace(-np.pi / core.C, np.pi / core.C, n_kc)
st = time.time()
print('(Error analysis for each k_c slice follows:)')
print('(Note that single ">" expectation times are per kc slice)')
print('\n' * 3)
assert((len(kcc) // 2) % 2 == 0)

for i, kc in enumerate(kcc):
    # 1 4 2 4 ... 4 1
    # Do Simpson integration over kc
    # This could be outsourced to scipy.quad().
    weight = 1 if i == 0 or i == len(kcc) - 1 else 2 + (i % 2) * 2
    dmu = -2 * structE[1] / 3.3 * np.cos(kc * core.C)
    structE_here = tuple([structE[0], structE[1], structE[2] + dmu])
    conj_here = tuple([conj[0], conj[1], conj[2] + dmu])
    sbb, saa, sab, err1 = core.compute_Bsweep(fields, structE_here, tau)
    sbb2, saa2, sab2, err2 = core.compute_Bsweep(fields, conj_here, tau)
    sbb += sbb2
    saa += saa2
    sab += sab2
    # Assumes the contribution of both sheets is similar
    # Also, I assume here that it does not change with field, component or kc
    # You can verify this by checking the numbers printed on the screen
    # during the calculation.
    err = np.sqrt(err1**2 + err2**2)

    # dkc * weight/3 is Simpson.
    # C/2pi is to remove the factor 2pi/C that is in the sigma calculation
    # that was already added in core.
    Sbb += (kcc[1] - kcc[0]) * weight / 3 * sbb * core.C / np.pi / 2
    Saa += (kcc[1] - kcc[0]) * weight / 3 * saa * core.C / np.pi / 2
    Sab += (kcc[1] - kcc[0]) * weight / 3 * sab * core.C / np.pi / 2

    # Do a second integration with double the kc step size.
    # This functions to estimate the error made.
    if i % 2 == 0:
        weight = 1 if i == 0 or i == len(kcc) - 1 else 2 + (i % 4)
        Sbb_check += (kcc[2] - kcc[0]) * weight / 3 * sbb * core.C / np.pi / 2
        Saa_check += (kcc[2] - kcc[0]) * weight / 3 * saa * core.C / np.pi / 2
        Sab_check += (kcc[2] - kcc[0]) * weight / 3 * sab * core.C / np.pi / 2
    expect = (len(kcc) - i - 1) / (i + 1) * (time.time() - st)
    print(f'>> Expect another {expect:.0f} s for full c-axis warping')
print(f'Total exe time: {time.time() - st:.0f} s')
print('\n' * 3)

# This is the kc-integral induced error,
# which does do
err_a = np.mean(np.abs(Saa - Saa_check) / Saa)
print(f'Error fraction is {err_a:.1e} for Saa')
err_b = np.mean(np.abs(Sbb - Sbb_check) / Sbb)
print(f'Error fraction is {err_b:.1e} for Sbb')


###########################
# Compute
# Store
###########################

name = 'merino' if merino else 'nuss'
path = os.path.abspath(os.path.join('output c-axis', f'c-axis_{name}.dat'))
with open(path, 'w') as f:
    f.write('# c-axis included RTA results\n')
    f.write(f'# merino: {merino}\n')
    f.write(f'# tau: {tau}\n')
    f.write(f'# dt = tau / {dtfrac}\n')
    f.write(f'# n_ka = {n_ka}\n')
    f.write(f'# n_kc = {n_kc}\n')
    f.write(f'# in-sheet error fraction = {err}\n')
    f.write(f'# kc error fraction saa = {err_a}\n')
    f.write(f'# kc error fraction sbb = {err_b}\n')
    f.write(f'# Exe time {time.time() - st:.0f} s\n')
    f.write('# Units are Tesla and 1/muOhmcm x3\n\n')

    f.write('BB saa sbb sab\n')
    for field, a, b, ab in zip(fields, Saa, Sbb, Sab):
        f.write(f'{field:18.10f} {a:18.10e} {b:18.10e} {ab:18.10e}\n')
