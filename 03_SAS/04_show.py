import os
import numpy as np
import matplotlib.pyplot as plt
import mine


# 065: special fs: RTA-like isotropic tau matches RTA (bs_type = 5)
# 066: special fs: isotropic L (bs_type=1) zero MR - like 62/63
#       --> Also analytical proof in overview
# 072: Nuss isotropic L (bs_type = 1)  [63 repeat - zero MRb]
# 073: Merino isotropic tau (bs_type = 2)
# 074: Nuss isotropic tau (bs_type = 2)
# 075: Merino isotropic L (bs_type = 1)  [62 repeat - zero MRb]


file = os.path.abspath('output_03/03_output_074.dat')
info, _, data = mine.labelled(file)

BB, sbb, dsbb, saa, dsaa, sab, dsab = np.array(data) / 1e8
BB *= 1e8

# Resistivities
denom = saa * sbb + sab**2
Rbb = saa / denom
err_aa = (1 / denom - saa * sbb / denom**2)**2 * dsaa**2
err_bb = (-saa**2 / denom**2)**2 * dsbb**2
err_ab = (-2 * saa * sab / denom**2)**2 * dsab**2
print(f'Fractional error on Rbb: aa-> {np.mean(np.sqrt(err_aa)/Rbb):.1e} '
      f'bb-> {np.mean(np.sqrt(err_bb)/Rbb):.1e}, '
      f'ab-> {np.mean(np.sqrt(err_ab)/Rbb):.1e}')
dRbb = np.sqrt(err_aa + err_bb + err_ab)

Raa = sbb / denom
err_aa = (-sbb**2 / denom**2)**2 * dsaa**2
err_bb = (1 / denom - saa * sbb / denom**2)**2 * dsbb**2
err_ab = (-2 * sbb * sab / denom**2)**2 * dsab**2

print(f'Fractional error on Raa: aa-> {np.mean(np.sqrt(err_aa)/Raa):.1e} '
      f'bb-> {np.mean(np.sqrt(err_bb)/Raa):.1e}, '
      f'ab-> {np.mean(np.sqrt(err_ab)/Raa):.1e}')
dRaa = np.sqrt(err_aa + err_bb + err_ab)

Rab = sab / denom
err_aa = (-sab * sbb / denom**2)**2 * dsaa**2
err_bb = (-sab * saa / denom**2)**2 * dsbb**2
err_ab = (1 / denom - 2 * sbb * sab / denom**2)**2 * dsab**2
print(f'Fractional error on Rab: aa-> {np.mean(np.sqrt(err_aa[2:])/Rab[2:]):.1e} '
      f'bb-> {np.mean(np.sqrt(err_bb[2:])/Rab[2:]):.1e}, '
      f'ab-> {np.mean(np.sqrt(err_ab[2:])/Rab[2:]):.1e}')
dRab = np.sqrt(err_aa + err_bb + err_ab)


# Lifetimes

for line in info.split('\n'):
    if 't_diff' in line:
        tau_a = float(line.split()[-2])
    if 'back_L' in line:
        tau_b = float(line.split()[-2]) / 1e5

print()
# xx = np.linspace(0, max(BB), 1000)
# yy_b = xx**2 * 4.58e-11 * (tau_b / 1e-14)**2
# yy_a = xx**2 * 2.29e-6 * (tau_a / 1e-14)**2

# Plotting

print()
print(f'Anisotropy rho_aa / rho_bb: {Raa[0] / Rbb[0]:.3f}')
print(
    f'MR anisotropy: {(Raa[-1] - Raa[0]) * Rbb[0] / Raa[0] / (Rbb[-1] - Rbb[0]):.0f}')

plt.rc('font', size=35)
plt.errorbar(BB, (Rbb - Rbb[0]) / Rbb[0], yerr=dRbb /
             Rbb[0], label='Rbb', ms=10, marker='o', lw=5)
plt.ylim(*plt.ylim())
# plt.plot(xx, yy_b, color='black', lw=5)
plt.title('$\u03C1_{bb,0}$ = ' + f'{Rbb[0]:.1f} \u03BC\u03A9cm')
plt.xlim(left=0)
plt.legend()
plt.xlabel('$\mu_0H$ (T)')
plt.ylabel('$\u0394\u03C1_{bb}/\u03C1_{bb,0}$')

plt.figure()
plt.errorbar(BB, (Raa - Raa[0]) / Raa[0], yerr=dRaa /
             Raa[0], label='Raa', ms=10, marker='o', lw=5)
plt.title('$\u03C1_{aa,0}$ = 'f'{Raa[0]:.0f} \u03BC\u03A9cm')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.xlabel('$\mu_0H$ (T)')
plt.ylabel('$\u0394\u03C1_{aa}/\u03C1_{aa,0}$')
xx = np.linspace(0, max(BB), 1000)
# plt.plot(xx, yy_a, color='black', lw=5)

plt.figure()
plt.errorbar(BB, np.abs(Rab), yerr=dRab, label='Rab', ms=10, marker='o', lw=5)
plt.xlim(left=0)
plt.legend()
plt.xlabel('$\mu_0H$ (T)')
plt.ylabel('$|\u03C1_{ab}|$ (\u03BC\u03A9cm)')



plt.figure()
plt.plot(BB, saa)
plt.plot(BB, sbb)
plt.plot(BB, sab)
plt.ylabel('$\sigma_{ij}$ (1/\u03A9m)')
plt.xlabel('$B$ (T)')


plt.show()
