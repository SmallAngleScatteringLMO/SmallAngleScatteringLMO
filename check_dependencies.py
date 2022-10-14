import sys

print('The following code crashes if you lack a package you need')
print('It will also report the version numbers.')
print('Usually an equal major number will be required, a higher minor is usually fine.')
print()
print(f'You are using Python {sys.version}, the code was made with 3.8.2')

try:
    import numpy as np
except Exception:
    raise ImportError('Please install numpy')
print(f'You use Numpy {np.__version__}, the code was made with 1.23.3')

try:
    import numba
except Exception:
    raise ImportError('Please install numpy')
print(f'You use Numba {numba.__version__}, the code was made with 0.56.2')

try:
    import matplotlib as mpl
except Exception:
    raise ImportError('Please install numpy')
print(f'You use Matplotlib {mpl.__version__}, the code was made with 3.5.3')

try:
    import scipy
except Exception:
    raise ImportError('Please install numpy')
print(f'You use Scipy {scipy.__version__}, the code was made with 1.9.1')
