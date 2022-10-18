# Take the last results generated and show them

import os
import numpy as np
import matplotlib.pyplot as plt


print('Choose which Fermi surface to use')
choice = '0'
choice = input('Enter "1" for Merino, "2" for Nuss: ')
while choice not in ['1', '2']:
    print('Illegal entry.')
    choice = input('Enter "1" for Merino, "2" for Nuss: ')
merino = choice == '1'

print()
print('Choose whether to look for low-field or all-field')
choice = '0'
choice = input('Enter "1" for realistic field, "2" for all-field: ')
while choice not in ['1', '2']:
    print('Illegal entry.')
    choice = input('Enter "1" for Merino, "2" for Nuss: ')
maxB = 1e99 if choice == '2' else 200

###########################
# User Input
# Definitions
###########################

# Lattice parameters, see e.g. Popovic2006
A = 9.499e-10
B = 5.523e-10
C = 12.762e-10
E = 1.602176634e-19
KB = 1.380649e-23
HBAR = 1.054571817e-34

###########################
# Definitions
# Import
###########################

name = 'merino' if merino else 'nuss'
path = os.path.abspath(os.path.join('output c-axis', f'c-axis_{name}.dat'))
saa, sbb, sab = [], [], []
with open(path) as f:
    for line in f:
        if line.startswith('BB'):
            break

    BB = []
    Saa = []
    Sbb = []
    Sab = []
    for line in f:
        if line:
            nums = [float(i) for i in line.split()]
            BB.append(nums[0])
            Saa.append(nums[1])
            Sbb.append(nums[2])
            Sab.append(nums[3])
BB = np.array(BB)
Saa = np.array(Saa)
Sbb = np.array(Sbb)
Sab = np.array(Sab)


###########################
# Import
# Report
###########################


rbb = Saa / (Sbb * Saa + Sab**2)
raa = Sbb / (Saa * Sbb + Sab**2)
rab = Sab / (Saa * Sbb + Sab**2)
print(f'\u03C1_b = {rbb[0]:.3f}')
print(f'\u03C1_a/\u03C1_b={raa[0] / rbb[0]:.3f}')
index = np.argmin(np.abs(BB - 100))
a_result = (raa[index] - raa[0]) / raa[0] / BB[index]**2
print(f'\u0394\u03C1_a/\u03C1_a = {a_result:.2e}  (1 T)')
b_result = (rbb[index] - rbb[0]) / rbb[0] / BB[index]**2
print(f'\u0394\u03C1_b/\u03C1_b = {b_result:.2e}  (1 T)')
print(f' > ratio is {a_result / b_result:.0f}')
print(f' > Hussey2002 predicts ratio {(raa[0] * A / rbb[0] / B)**2:.0f}')

###########################
# Report
# Import no-warping case
###########################

if merino:
    path = os.path.abspath(os.path.join('output', 'merino.dat'))
else:
    path = os.path.abspath(os.path.join('output', 'nuss.dat'))

with open(path) as f:
    for line in f:
        if line.startswith('B '):
            break

    BBr = []
    saar = []
    sbbr = []
    sabr = []
    for line in f:
        nums = [float(x) for x in line.split()]
        BBr.append(nums[0])
        saar.append(nums[1])
        sbbr.append(nums[2])
        sabr.append(nums[3])
BBr = np.array(BBr)
wh = BBr < maxB
BBr = BBr[wh]
saar = np.array(saar)[wh]
sbbr = np.array(sbbr)[wh]
sabr = np.array(sabr)[wh]
raar = sbbr / (saar * sbbr + sabr**2)
rbbr = saar / (saar * sbbr + sabr**2)
rabr = sabr / (saar * sbbr + sabr**2)

i0 = np.argmin(BB)
if abs(BBr[i0]) > 1e-10:
    raise ValueError('No 0 T data in the RTA results')
i1 = np.argmin(np.abs(BB - 100))
if abs(BBr[i0]) > 1e-10:
    raise ValueError('No 100 T data in the RTA results')

print()
print('Compare with RTA no c-axis warping:')
print(f'\u03C1_b({BBr[i0]:.0e}) = {rbbr[i0]:.3f}')
print(f'\u03C1_a\u03C1_b = {raar[i0] / rbbr[i0]:.3f}')
print(f'\u0394\u03C1_a/\u03C1_a = {(raar[i1] - raar[i0]) / raar[i0]:.2e}  (100 T)')
print(f'\u0394\u03C1_b/\u03C1_b = {(rbbr[i1] - rbbr[i0]) / rbbr[i0]:.2e}  (100 T)')
ani_rta = (raar[i1] - raar[i0]) * rbbr[i0] / (rbbr[i1] - rbbr[i0]) / raar[i0]
print(f' > MR ratio is {ani_rta:.0f}')


###########################
# Report
# Plots
###########################

plt.rc('font', size=25)
lw = 5
plt.figure('Rbb')
plt.plot(BBr, rbbr / rbbr[0] - 1, lw=lw, label='RTA no c-axis')
plt.plot(BB, rbb / rbb[0] - 1, lw=lw, label='RTA c-axis')
plt.legend()
plt.xlabel('$B$ (T)')
plt.ylabel('$\u0394\u03C1_{bb}/\u03C1_0$')
plt.title(f'$\u03C1_b$ = {rbb[0]:.1f} \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.figure('Raa')
plt.plot(BBr, raar / raar[0] - 1, lw=lw, label='RTA no c-axis')
plt.plot(BB, raa / raa[0] - 1, lw=lw, label='RTA c-axis')
plt.legend()
plt.xlabel('$B$ (T)')
plt.ylabel('$\u0394\u03C1_{aa}/\u03C1_0$')
plt.title(f'$\u03C1_a$ = {raa[0]:.0f} \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.figure('RH')
plt.plot(BBr, np.abs(rabr), lw=lw, label='RTA no c-axis')
plt.plot(BB, np.abs(rab), lw=lw, label='RTA c-axis')
plt.legend()
plt.xlabel('$B$ (T)')
plt.ylabel('$|\u03C1_{ab}|$ \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
