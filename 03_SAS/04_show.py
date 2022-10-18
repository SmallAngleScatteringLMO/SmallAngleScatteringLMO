import os
import numpy as np
import matplotlib.pyplot as plt
import mine


# 001 -> n=30
# 002 -> n=100
# 003 -> n=200
# 004 -> n=100, err_frac=1e-10 [from 1e-9]
# 005 -> change to -sigma_ba from sigma_ab
# 006 -> dt0fact=8 [from 5]
# 007 -> n=200 -> find a bug in error computation
#
# 008 -> bug fixed, t_diff=1e-15 [before 1e-13]
#   ==> negative MR outside errorbar
#       seems to be that sbb is increasing with B ?
# 009 -> t_diff=1e-14 [same as backscattering, relaxation result!]
#   ==> negative MR *violates* relaxation time results. -4e-5 @3000T
# 010 -> more field values
# 011 -> t_diff=1e-12 [very long, try for sbb=const]
#   ==> Increase of sbb with field *exactly* the same, slightly faster with B.
# 012 -> Change backscattering to isotropic tau (not isotropic L like before)
#       rates = -np.ones(len(vbb)) * 0.5 * np.mean(vbb) / mean_free_path
#   ==> sbb constant
#   ==> still 5e-6 negative MR, must be from sab^2/saa term in denom
# 013 -> back to t_diff=1e-14, n=500 to check not n aberation.
#       -7e-5 saturation MR still.
#   ==> most accurate yet
#
# 018 -> n=10 x fs=16 [investigate sab merino_big]
# 019 -> n=10 x fs=24
# 020 -> n=10 x fs=20
#
# 021 -> keep 10x20, now small_merino only
#       ==> good looking RH
#       ==> -2e-4 saturation MR from sbb increasing 100%
# 022 -> big_merino [0 RH :(]
#       ==> -5e-6 saturation MR from sbb increasing 100%
# 023 -> 11x20 big merino [0 RH :(]
# 024 -> 10x18 big merino [0 RH :(]
# 025 -> bug
# 026 -> small_merino 10x20, now integrate both sheets [unstable?]
# 027 -> small_merino 10x20, now integrate neg sheets [unstable?]
# 028 -> pos sheet [stable]
# 029 -> neg sheet
#   ==> abs value in time doubling added. Fixed.
# 030 -> checked all good
#   ==> Add err_a condition on doubling, remove again since field=0 err_a stuck.
#   ==> change max t_end/4 to t_end/n*5 as max dt.
#           For matrix multiplication rises faster than vector multiplication
#   ==> change back to sigma_ba from sigma_ab because nr_fs << nr_ka
#
# 032 very accurate
# 033 even more accurate [Ra/Rb=121, Merino]
# 034 Nuss fs very accurate [Ra/Rb=575]
# 035 Nuss fs rho_aa/rho_bb = 78
# 036 now with proper error (n_ka, much bigger)
# 037 now with more fs points (leading s_ab error on rho_bb reduced)
# 038 Nuss fs very accurate [Ra/Rb=80]
# 039 Merino very accurate [Ra/Rb=80]
# ----> Here: fix bug sab 2x too small. Still confused why. Swapped sab to sba
# 040 Merino [Ra/Rb=80], now w/o the bug. Lower settings.
# 041 Merino [Ra/Rb=80], fs=26 x 20
# !042 Merino [Ra/Rb=80], fs=28 x 32
# !043 Nuss [Ra/Rb=80], fs=28 x 32 ===> initial figure for the paper
#
# 044 Nuss, but now bs_type=4, isotropic not back scattering to try positive MRb.
#   Settings are not high enough (28 x 10)
# 045 Go for higher settings (48 x 10) and *disable* small angle alltogether.
#   Error is now 1/n_ka
# 046 Try really high n_ka (28 x 100), go to Merino for higher MR.
# 047 change structE to a very anisotropic version which almost
#   touches the BZ edge and thus has enormous v-anisotorpy.
#   But the ab error is huge, clearly sbb increases with field still.
# 048 bs_type=5, actual exponential RTA-like decay, very low settings 16 x 6
# 049 Swap sigma_ab for sigma_ba. Nothing. Nice.
# 050 Higher settings 24 x 10
#   !! Positive MR saturation 0.006
#   --> RTA agrees rho_aa(0), rho_bb(0), MRa
#   --> RTA has 10x larger MRb.
#      --> RTA 10% sbb decline with field, here 1%.
#      --> Also, RTA has 5x larger sigma_ab and the error here is 2000% (!)
#         --> The leading s_ab error is fs_points. Files 40-42 show scaling as 1/(n_ka/fs_points)
#               of the error, indicating these are improperly aligned
#         --> Must have to do with corrvB AND field, where one alone has no error.
# 051 48 x 10 changes:
#   rho_ab 30 -> 5 and MRb 0.006 -> 0.0008 and MRa 122 -> 130
# 052 48x20 unchanged
#   --> Clearly it is the outer integral
# 053 480x1
#   rho_ab 0 (flakey) and MRb 3e-7 instant 0 field, and MRa 130
#
###########################################
# --> cyclotron motion fixed
#
# 062: Merino isotropic L (bs_type = 1)
# 063: Nuss isotropic L   (bs_type = 1)  --> low settings
#   --> Both ZERO mr.
# 065: special fs: RTA-like isotropic tau matches RTA (bs_type = 5)
# 066: special fs: isotropic L (bs_type=1) zero MR - like 62/63
#       --> Also analytical proof in document 06
#
# --> Also switch to full RK4 not 0.5(c1+c4) [see code] trapezoid cyclotron.
#   Do testing on Merino FS and find no change, only a perhaps ~30% lower error with nr_ka
#
# !072: Nuss isotropic L (bs_type = 1)  [63 repeat - zero MRb]
# !073: Merino isotropic tau (bs_type = 2)
# !074: Nuss isotropic tau (bs_type = 2)
# !075: Merino isotropic L (bs_type = 1)  [62 repeat - zero MRb]



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
