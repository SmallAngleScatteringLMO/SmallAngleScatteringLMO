# A dedicated functionality to combine the results of the two types of calculation

import os
import numpy as np
import matplotlib.pyplot as plt
import mine


def get_bRTA_data():
    """ Get the resistivities with errors from file. """

    file = os.path.abspath('output/03_output_065.dat')
    info, _, bRTA = mine.labelled(file)

    BB, sbb, dsbb, saa, dsaa, sab, dsab = np.array(bRTA) / 1e8
    # Cap AB errors to the absolute value, because that
    # is ultimately the maximum influence it should be given.
    dsab = np.array([min(abs(v), dv) for v, dv in zip(sab, dsab)])
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

    return BB, Rbb, dRbb, Raa, dRaa, Rab, dRab


def get_RTA_data():
    """ From file. Resistivities, no error bars. """

    file = os.path.abspath('15_RTA_special_fs.dat')
    _, _, RTA = mine.labelled(file)

    BB, saa, sbb, sab = RTA
    Raa = sbb / (saa * sbb + sab**2)
    Rbb = saa / (saa * sbb + sab**2)
    Rab = sab / (saa * sbb + sab**2)

    return BB, Rbb, Raa, Rab


BB, Rbb, dRbb, Raa, dRaa, Rab, dRab = get_bRTA_data()
bb, rbb, raa, rab = get_RTA_data()


#################
# Import
# Stats
#################

print()
print(f'Rbb(0) is RTA={rbb[0]:.1f} vs bRTA={Rbb[0]:.1f} muOhmcm')
print(
    f'Resistivity anisotropy is RTA={raa[0]/rbb[0]:.3f} vs bRTA={Raa[0]/Rbb[0]:.3f}')

i50 = np.argmin(np.abs(BB - 50))
MRb = (Rbb[i50] - Rbb[0]) / Rbb[0]
MRa = (Raa[i50] - Raa[0]) / Raa[0]
i50 = np.argmin(np.abs(bb - 50))
mrb = (rbb[i50] - rbb[0]) / rbb[0]
mra = (raa[i50] - raa[0]) / raa[0]
print()
print('MR stats')
print('--------')
print(f'MRa is RTA={mra:.2e} vs bRTA={MRa:.2e}')
print(f'MRb is RTA={mrb:.2e} vs bRTA={MRb:.2e}')
print(f'MR anisotropy is RTA={mra/mrb:.2f} vs bRTA={MRa / MRb:.2f}')
print()
print('Saturation stats')
print('----------------')
saturation = (rbb[-1] - rbb[0]) / rbb[0]
Saturation = (Rbb[-1] - Rbb[0]) / Rbb[0]
print(f'MRb saturation at RTA={saturation:.3f} vs bRTA={Saturation:.3f}')


#################
# Stats
# Plotting
#################


plt.rc('font', size=25)

f, axes = plt.subplots(ncols=3, nrows=2, figsize=(21, 12))
f.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.1,
                  hspace=0.2, wspace=0.35)

# MR b
axes[0][0].errorbar(BB, (Rbb - Rbb[0]) / Rbb[0], yerr=dRbb / Rbb[0], lw=3,
             elinewidth=5, color='tab:blue', label='RTA-like')
axes[0][0].plot(bb, (rbb - rbb[0]) / rbb[0], lw=3, color='tab:orange', label='RTA')
axes[0][0].legend()
axes[0][0].set_xlabel('$\mu_0H$ (T)')
axes[0][0].set_ylabel('$\u0394\u03C1_{bb}/\u03C1_{bb,0}$')
axes[0][0].set_xlim(left=0)
axes[0][0].set_ylim(bottom=0)

# MR b low field
axes[1][0].errorbar(BB, (Rbb - Rbb[0]) / Rbb[0], yerr=dRbb / Rbb[0], lw=3,
            elinewidth=5, color='tab:blue', label='beyond RTA')
axes[1][0].plot(bb, (rbb - rbb[0]) / rbb[0], lw=3, color='tab:orange', label='RTA')
axes[1][0].set_xlabel('$\mu_0H$ (T)')
axes[1][0].set_ylabel('$\u0394\u03C1_{bb}/\u03C1_{bb,0}$')
axes[1][0].set_xlim(0, 100)
axes[1][0].set_ylim(0, 0.0025)

# MR a
axes[0][1].errorbar(BB, (Raa - Raa[0]) / Raa[0], yerr=dRaa / Raa[0], lw=3,
             elinewidth=5, color='tab:blue', label='beyond RTA')
axes[0][1].plot(bb, (raa - raa[0]) / raa[0], lw=3, color='tab:orange', label='RTA')
axes[0][1].set_xlabel('$\mu_0H$ (T)')
axes[0][1].set_ylabel('$\u0394\u03C1_{aa}/\u03C1_{aa,0}$')
axes[0][1].set_xlim(left=0)
axes[0][1].set_ylim(bottom=0)

# MR a low field
axes[1][1].errorbar(BB, (Raa - Raa[0]) / Raa[0], yerr=dRaa / Raa[0], lw=3,
             elinewidth=5, color='tab:blue', label='beyond RTA')
axes[1][1].plot(bb, (raa - raa[0]) / raa[0], lw=3, color='tab:orange', label='RTA')
axes[1][1].set_xlabel('$\mu_0H$ (T)')
axes[1][1].set_ylabel('$\u0394\u03C1_{aa}/\u03C1_{aa,0}$')
axes[1][1].set_xlim(0, 100)
axes[1][1].set_ylim(0, 0.025)

# Hall
axes[0][2].errorbar(BB, Rab, yerr=dRab, lw=3, elinewidth=5,
             color='tab:blue', label='beyond RTA')
axes[0][2].plot(bb, rab, lw=3, color='tab:orange', label='RTA')
axes[0][2].set_xlabel('$\mu_0H$ (T)')
axes[0][2].set_ylabel('$\u03C1_{ab}$ (\u03BC\u03A9cm)')
axes[0][2].set_xlim(left=0)
axes[0][2].set_ylim(bottom=0)

# Hall low field
axes[1][2].errorbar(BB, Rab, yerr=dRab, lw=3, elinewidth=5,
             color='tab:blue', label='beyond RTA')
axes[1][2].plot(bb, rab, lw=3, color='tab:orange', label='RTA')
axes[1][2].set_xlabel('$\mu_0H$ (T)')
axes[1][2].set_ylabel('$\u03C1_{ab}$ (\u03BC\u03A9cm)')
axes[1][2].set_xlim(0, 100)
axes[1][2].set_ylim(0, 15)


plt.savefig('16_RTA_comparison.png', dpi=300)

plt.show()
