# As a function of kc,
# Shift the energy appropriately and show
# the influence on va and vb.
#
# This is what explains why the impact of tc
# on rho_aa and rho_bb is as tiny as it is.

import core
import numpy as np
import matplotlib.pyplot as plt


structE = tuple([0.5 / 8, 0.036 / 8, -0.02])
conj = tuple([structE[0], -structE[1], structE[2] + 0.02])
tc = structE[1] / 3.3


kaa = np.linspace(-np.pi / core.A, np.pi / core.A, 51)
kcc = np.linspace(-np.pi / core.C, np.pi / core.C, 51)
Ka, Kc = np.meshgrid(kaa, kcc)
Kb1 = np.zeros(np.shape(Ka))
Kb2 = np.zeros(np.shape(Ka))
Vb1 = np.zeros(np.shape(Ka))
Vb2 = np.zeros(np.shape(Ka))
Va1 = np.zeros(np.shape(Ka))
Va2 = np.zeros(np.shape(Ka))

for i in range(len(Ka)):
    for j in range(len(Ka[0])):
        dmu = -2 * tc * np.cos(Kc[i, j] * core.C)
        structE_here = tuple([structE[0], structE[1], structE[2] + dmu])
        Kb1[i, j] = core.get_kb(Ka[i, j], structE_here)
        Va1[i, j] = core.compute_velocity_a(Kb1[i, j], Ka[i, j], structE_here)
        Vb1[i, j] = core.compute_velocity_b(Kb1[i, j], Ka[i, j], structE_here)

        conj_here = tuple([conj[0], conj[1], conj[2] + dmu])
        Kb2[i, j] = core.get_kb(Ka[i, j], conj_here)
        Va2[i, j] = core.compute_velocity_a(Kb2[i, j], Ka[i, j], structE_here)
        Vb2[i, j] = core.compute_velocity_b(Kb2[i, j], Ka[i, j], structE_here)

plt.rc('font', size=25)
fig = plt.figure('Fermi surface', figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(Ka, Kc, Kb1, color='tab:red')
ax.plot_surface(Ka, Kc, Kb2, color='tab:grey')
# ax.plot_surface(Ka, Kc, -Kb1, color='tab:red')
# ax.plot_surface(Ka, Kc, -Kb2, color='tab:grey')
ax.set_xlabel('$k_a$ (1/\u212B)')
ax.set_ylabel('$k_c$ (1/\u212B)')
ax.set_zlabel('$k_b$ (1/\u212B)')


colors = plt.get_cmap('viridis', len(Ka)).colors
width1 = []
width2 = []
for i, c in enumerate(colors):
    kaa = Ka[i, :]
    kbb1 = Kb1[i, :]
    kbb2 = Kb2[i, :]
    vvb1 = Vb1[i, :]
    vvb2 = Vb2[i, :]
    vva1 = Va1[i, :]
    vva2 = Va2[i, :]

    plt.figure('slices')
    plt.plot(kaa, kbb1, color=c)
    plt.plot(kaa, kbb2, color=c)
    plt.xlabel('$k_a$ (1/m)')
    plt.ylabel('$k_b$ (1/s)')

    plt.figure('velocities a')
    plt.plot(kaa, vva1, color=c)
    plt.plot(kaa, vva2, color=c)
    plt.xlabel('$k_a$ (1/m)')
    plt.ylabel('$v_a$ (m/s)')

    plt.figure('velocities b')
    plt.plot(kaa, vvb1, color=c)
    plt.plot(kaa, vvb2, color=c)
    plt.xlabel('$k_a$ (1/m)')
    plt.ylabel('$v_b$ (m/s)')

    width1.append(max(kbb1) - min(kbb1))
    width2.append(max(kbb2) - min(kbb2))

plt.figure('Width variation')
plt.plot(Kc[:, 0], width1, lw=5, color='tab:red')
plt.plot(Kc[:, 0], width2, lw=5, color='tab:grey')
plt.xlabel('$k_c$ (1/\u212B)')
plt.ylabel('$\Delta k_b$ (1/\u212B)')
plt.xlim(min(Kc[:, 0]), max(Kc[:, 0]))

plt.show()
