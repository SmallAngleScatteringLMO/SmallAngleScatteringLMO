# First time using this code for results
from code import *
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

fs_points = 28
nr_ka = fs_points * 36
back_L = 1e-9

# Merino
# structE = np.array([0.5 / 8, 0.036 / 8, -0.02])
# conj = tuple([structE[0], -structE[1], structE[2] + 0.02])
# is_nuss = False
# t_diff = 1.5e-14
# bs_type = 2

# Nuss
structE = tuple([0.0625, 0.003, -0.0075])
conj = tuple([0.0625, -0.00038, 0.00088])
is_nuss = True
t_diff = 6.8e-14
bs_type = 2

# RTA
# structE = tuple([0.5, 0.286, -0.26])
# structE = tuple([s / 8 for s in structE])
# conj = tuple([structE[0], -structE[1], structE[2] + 0.085])
# is_nuss = 2
# t_diff = 1e-14
# bs_type = 2


t_end = 2e-12
fields = list(range(0, 110, 10))
fields += [5, 15, 20, 25, 35]
fields += [300, 500, 1000, 2000, 4000, 6000, 8000, 10000]
fields = np.sort(fields)

err_frac = 1e-11
dt0fact = 10

def main(field):
    """ Calculate the conductivity tensor with errors at this B-field. """

    D = calculate_D(t_diff)

    Saa = []
    Sbb = []
    Sab = []
    dSaa = []
    dSbb = []
    dSab = []

    st = time.time()

    r1 = eadaptive(D, field, back_L, nr_ka, t_end, err_frac, dt0fact,
        fs_points, structE, bs_type=bs_type)
    r2 = eadaptive(D, field, back_L, nr_ka, t_end, err_frac, dt0fact,
        fs_points, conj, bs_type=bs_type)
    sbb1, saa1, sab1, dsbb1, dsaa1, dsab1, iters1 = r1
    sbb2, saa2, sab2, dsbb2, dsaa2, dsab2, iters2 = r2

    Saa = saa1 + saa2
    Sbb = sbb1 + sbb2
    Sab = sab1 + sab2
    dSaa = np.sqrt((dsaa1 * saa1)**2 + (dsaa2 * saa2)**2)
    dSbb = np.sqrt((dsbb1 * sbb1)**2 + (dsbb2 * sbb2)**2)
    dSab = np.sqrt((dsab1 * sab1)**2 + (dsab2 * sab2)**2)

    print(f'\n{field} T:'
          f'\nComputation took {time.time() - st:.2f} s [{iters1 + iters2} T evolutions]'
          f'\nConductivities are saa={Saa:.5f}, sbb={Sbb:.8f}, sab={Sab:.5f} (SI)'
          f'\nAbsolute errors are {dSaa:.1e}, {dSbb:.1e}, {dSab:.1e}', flush=True)
    return Sbb, dSbb, Saa, dSaa, Sab, dSab


def launch_all():


    st = time.time()
    if nr_ka < 500:
        with multiprocessing.Pool(6) as p:
            result = p.map(main, fields)
    else:
        result = []
        for i, field in enumerate(fields):
            result.append(main(field))
            expect = (len(fields) - i - 1) / (i + 1) * (time.time() - st)
            print(f'Expect another {expect:.0f} s')
    result = np.array(result).T


    # Put all the results in the output file
    # Ready to plot any which way you like.
    Sbb, dSbb, Saa, dSaa, Sab, dSab = result
    output = 'output_03/03_output_001.dat'
    index = 1
    while os.path.isfile(output):
        index += 1
        output = os.path.abspath(f'output_03/03_output_{index:03d}.dat')
    txt = f'# nr_ka = {nr_ka}\n'
    txt += f'# fs_points = {fs_points}\n'
    txt += f'# t_diff = {t_diff} s\n'
    txt += f'# back_L = {back_L} m\n'
    txt += f'# t_end = {t_end} s\n'
    txt += f'# dt0fact = {dt0fact}\n'
    txt += f'# err_frac = {err_frac}\n'
    txt += f'# nuss FS = {is_nuss}\n'
    txt += f'# bs_type = {bs_type} (default 1)\n'
    txt += f'# Time taken {time.time() - st:.0f} s\n'
    txt += '# Errors are absolute\n'
    txt += '# Field in Tesla, rest in SI for conductivity so 1/Ohm/m\n\n'
    txt += 'B Sbb dSbb Saa dSaa Sab dSab\n'
    for B, a, b, c, d, e, f in zip(fields, Sbb, dSbb, Saa, dSaa, Sab, dSab):
        print(B, a, b, c, d, e, f)
        txt += f'{B}, {a}, {b}, {c}, {d}, {e}, {f}\n'

    with open(output, 'w') as f:
        f.write(txt)
    print(f'> Written to file {output}')


if __name__ == '__main__':
    st = time.time()

    base = 2 * nr_ka / 250 if nr_ka < 500 else (nr_ka / 512)**3 * 10
    timer = base * fs_points * len(fields) / 6 * 5
    print(f'Rough expected exe time is {timer / 60:.0f} minutes')

    launch_all()
    plt.show()
