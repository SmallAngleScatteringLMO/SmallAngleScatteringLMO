# Take the last results generated and show them

import os
import numpy as np
import matplotlib.pyplot as plt


merino = False

# Lattice parameters, see e.g. Popovic2006
A = 9.499e-10
B = 5.523e-10
C = 12.762e-10
E = 1.602176634e-19
KB = 1.380649e-23
HBAR = 1.054571817e-34

###########################
# Import
###########################

name = 'merino' if merino else 'nuss'
path = os.path.abspath(os.path.join('output', f'c-axis_{name}.dat'))
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
        if ' 0.000' in line[:20]:
            line_0T = line
        if '100.000' in line[:20]:
            line_100T = line

_, saa, sbb, sab = [float(num) for num in line_0T.split()]
rbb0 = saa / (saa * sbb + sab**2)
raa0 = sbb / (saa * sbb + sab**2)
_, saa, sbb, sab = [float(num) for num in line_100T.split()]
rbb100 = saa / (saa * sbb + sab**2)
raa100 = sbb / (saa * sbb + sab**2)

print()
print('Compare with RTA no c-axis warping:')
print(f'\u03C1_b = {rbb0:.3f}')
print(f'\u03C1_a\u03C1_b = {raa0 / rbb0:.3f}')
print(f'\u0394\u03C1_a/\u03C1_a = {(raa100 - raa0) / raa0:.2e}  (1 T)')
print(f'\u0394\u03C1_b/\u03C1_b = {(rbb100 - rbb0) / rbb0:.2e}  (1 T)')
print(f' > ratio is {(raa100 - raa0) * rbb0 / (rbb100 - rbb0) / raa0:.0f}')


###########################
# Report
# Plots
###########################

plt.rc('font', size=25)
lw = 5
plt.figure('Rbb')
plt.plot(BB, rbb / rbb[0] - 1, lw=lw)
plt.xlabel('$B$ (T)')
plt.ylabel('$\u0394\u03C1_{bb}/\u03C1_0$')
plt.title(f'$\u03C1_b$ = {rbb[0]:.1f} \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.figure('Raa')
plt.plot(BB, raa / raa[0] - 1, lw=lw)
plt.xlabel('$B$ (T)')
plt.ylabel('$\u0394\u03C1_{aa}/\u03C1_0$')
plt.title(f'$\u03C1_a$ = {raa[0]:.0f} \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.figure('RH')
plt.plot(BB, np.abs(Sab) / (Sab**2 + Saa * Sbb), lw=lw)
plt.xlabel('$B$ (T)')
plt.ylabel('$|\u03C1_{ab}|$ \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()
