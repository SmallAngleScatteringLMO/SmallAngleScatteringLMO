# Field along C
# Corrugation along AB only
#
# This RTA code is used for multiple purposes:
#   1) Check the MR estimate from Hussey2002 [check]
#   2) RTA calculation of MR size Nuss & Merino
#   3) MR saturation MR Nuss & Merino
#   4) Provide a thorough check for the beyond-RTA model
#
# This file contains the script that guides the calculation
# The core.py file contains the actual calculation and is tested.
#
# This is the next level up from straight Drude theory,
# the simplest that can estimate any MR.
#
# -----------------------
# Part of the SmallAngleScatteringLMO repository
# Subject to the LICENSE terms and conditions
# Written by Roemer Hinlopen, 2022


import numpy as np
import os
import matplotlib.pyplot as plt
import core

#############################
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
    nuss = False
    tau = 1e-14
    fname = 'merino.dat'
    print('Starting calcuation for Merino-based Fermi surface')

# Nuss based (Referee)
elif choice == '2':
    structE = tuple([0.0625, 0.003, -0.0075])
    conj = tuple([0.0625, -0.00038, 0.00088])
    nuss = True
    tau = 1e-14
    fname = 'nuss.dat'
    print('Starting calcuation for Nuss-based Fermi surface')

# Artificial (large MR, excessive anisotropy)
elif choice == '3':
    structE = tuple([0.5, 0.286, -0.26])
    structE = tuple([s / 8 for s in structE])
    conj = tuple([structE[0], -structE[1], structE[2] + 0.085])
    nuss = 2
    tau = 1.15e-14
    fname = 'special_fs.dat'
    print('Starting calcuation for special high-anisotropy Fermi surface')

dtfrac = 101
print(f'dt=tau/{dtfrac} [change in the code]')
ntau = 30
print(f'Time integral cut off at {ntau}*tau [change in the code]')
n_ka = 31
print(f'Using {n_ka} ka points [change in the code]')

#############################
# Fermi surface
# Calculate
#############################

fieldscale = core.compute_fieldscale(structE, tau)
gamma = core.HBAR / core.KB / tau
print(f'Lifetime {tau:.1e} is the equivalent of Gamma={gamma:.1f} K')
print(f'Field scale for saturation is {fieldscale:.1f} T')

n1 = core.compute_density(structE)
n2 = core.compute_density(conj)
print(f'Densities are {2 * n1:.2f} + {2 * n2:.2f} = '
      f'{2 * n1 + 2 * n2:.2f} filling (max = 2, spin)')

# Note: The error is observed to be field-independent
#   and fractional. So do not bother with dsaa, dsbb etcetera like in SAS model,
#   the error propagation is simple.
# Note2: Errors are on absolute rho sizes. Typically, the majority of the error
#   is fractional and cancels out when plotting rho(H)/rho(0).
#   Nevertheless, make the errors much smaller than the MR.
fields = list(np.linspace(0, 10, 40)) + list(np.linspace(11, 100, 25))
fields += list(np.linspace(150, 1500, 11)) + list(np.linspace(2000, 10000, 9))
fields = np.array(fields)
sbb, saa, sab, err = core.compute_Bsweep(fields, structE, tau, dtfrac=dtfrac, ntau=ntau, n_ka=n_ka)
sbb2, saa2, sab2, err2 = core.compute_Bsweep(fields, conj, tau, dtfrac=dtfrac, ntau=ntau, n_ka=n_ka)
sbb += sbb2
saa += saa2
sab += sab2
err = np.sqrt(err**2 + err2**2)

#############################
# Calculate
# Export
#############################

direc = os.path.abspath('output')
if not os.path.isdir(direc):
    os.mkdir(direc)
path = os.path.join(direc, fname)

with open(path, 'w') as f:
    f.write('# Relaxation time approximation results\n')
    f.write(f'# tau = {tau}\n')
    f.write(f'# structE: tb={structE[0]} ta={structE[1]} mu={structE[2]}\n')
    f.write(f'# conj:    tb={conj[0]} ta={conj[1]} mu={conj[2]}\n')
    f.write(f'# relative err:  {err}\n')
    f.write(f'# is_nuss: {nuss}\n')
    f.write(f'# units T, 1/Ohmm x3\n')
    f.write(f'\nB saa sbb sab\n')

    for field, sa, sb, soff in zip(fields, saa, sbb, sab):
        print(field, sa, sb, soff)
        f.write(f'{field:18.10f} {sa:18.10e} {sb:18.10e} {soff:18.10e}\n')


#############################
# Export
# Plot
#############################

rbb = saa / (sbb * saa + sab**2)
raa = sbb / (saa * sbb + sab**2)
rab = sab / (saa * sbb + sab**2)
print(f'\u03C1_a/\u03C1_b={raa[0] / rbb[0]:.3f}')

for ratio in [2, 5, 10, 50]:
    print(ratio)
    index = np.argmin(np.abs(fields - fieldscale / ratio))
    a_result = (raa[index] - raa[0]) / raa[0] / fields[index]**2
    print(f'\u0394\u03C1_a/\u03C1_a = {a_result:.2e}  (1 T)')
    b_result = (rbb[index] - rbb[0]) / rbb[0] / fields[index]**2
    print(f'\u0394\u03C1_b/\u03C1_b = {b_result:.2e}  (1 T)')
    print(f' > ratio is {a_result / b_result:.0f}')
    print(f' > predict {(raa[0] * core.A / rbb[0] / core.B)**2:.1e}')
    print()

plt.rc('font', size=35)
lw = 5
plt.figure('rbb')
plt.plot(fields, rbb / rbb[0] - 1, lw=lw, label='\u03C1')
plt.plot(fields, sbb[0] / sbb - 1, lw=lw, label='$1/\sigma$', zorder=-10)
plt.ylabel('$\u0394\u03C1_{bb}/\u03C1_{bb,0}$')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel('$B$ (T)')
plt.title('$\u03C1_{bb,0}$ = ' + f'{rbb[0]:.1f} \u03BC\u03A9cm')
plt.legend()

plt.figure('raa')
plt.plot(fields, raa / raa[0] - 1, lw=lw, label='\u03C1', dashes=[10, 1])
plt.plot(fields, saa[0] / saa - 1, lw=lw, label='$1/\sigma$', zorder=-10)
plt.xlabel('$B$ (T)')
plt.ylabel('$\u0394\u03C1_{aa}/\u03C1_{aa,0}$')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.title('$\u03C1_{aa,0}$ = 'f'{raa[0]:.0f} \u03BC\u03A9cm')
plt.legend()

plt.figure('Rab')
plt.plot(fields, rab, lw=lw)
plt.xlabel('$B$ (T)')
plt.ylabel('$\u03C1_{ab}$')
plt.xlim(left=0)
plt.show()
