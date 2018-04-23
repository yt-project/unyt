from collections import OrderedDict
import subprocess
import os

setup = OrderedDict([
    ('astropy', 'import astropy.units as u'),
    ('pint', 'from pint import UnitRegistry; u = UnitRegistry()'),
    ('unyt', 'import unyt as u'),
    ('yt', 'import yt.units as u'),
    ('numpy', 'import numpy as np'),
])

base_args = ['python3', '-m', 'perf', 'timeit', '--rigorous']

numpy_setup = 'import numpy as np'


base_setups = OrderedDict([
    ('small_list', 'data = [1, 2, 3]'),
    ('small_tuple', 'data = (1, 2, 3)'),
    ('small_array', 'data = np.array([1, 2, 3])'),
    ('big_list', 'data = list(range(int(1e6)))'),
    ('big_array', 'data = np.arange(1e6)'),
])

for bs in base_setups:
    for package in sorted(setup):
        print(package)
        setup_s = (
            numpy_setup + '; ' + setup[package] + '; ' + base_setups[bs] + ' ')
        args = base_args + ['-s', setup_s]
        if package == 'numpy':
            args.append('np.asarray(data)')
        else:
            args.append('data*u.g')
        json_name = '{}_{}.json'.format(package, bs)
        if os.path.exists(json_name):
            os.remove(json_name)
        args.append('-o {}'.format(json_name))
        print(' '.join(args))
        p = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        print(out.decode())
    
