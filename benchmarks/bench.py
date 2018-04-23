from collections import OrderedDict
import subprocess
import os

def run_perf(args, json_name):
    if os.path.exists(json_name):
        os.remove(json_name)
    args = args + ['-o', json_name]
    print(args)
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(out.decode())
    print(err.decode())


setup = OrderedDict([
    ('astropy', 'import astropy.units as u'),
    ('pint', 'from pint import UnitRegistry; u = UnitRegistry()'),
    ('unyt', 'import unyt as u'),
    ('yt', 'import yt.units as u'),
    ('numpy', 'import numpy as np'),
])

base_args = ['python3', '-m', 'perf', 'timeit', '--rigorous']

shared_setup = 'import numpy as np; import operator'

base_setups = OrderedDict([
    ('small_list', 'data = [1., 2., 3.]'),
    ('small_tuple', 'data = (1., 2., 3.)'),
    ('small_array', 'data = np.array([1., 2., 3.])'),
    ('big_list', 'data = np.arange(1e6).tolist()'),
    ('big_array', 'data = np.arange(1e6)'),
])

op_ufuncs = OrderedDict([
    ('operator.add', 'np.add'),
    ('operator.sub', 'np.subtract'),
    ('operator.mul', 'np.multiply'),
    ('operator.truediv', 'np.true_divide'),
])

for bs in base_setups:
    for package in sorted(setup):
        print(package)
        setup_s = '; '.join([shared_setup, setup[package], base_setups[bs]])
        args = base_args + ['-s', setup_s + ' ']
        if package == 'numpy':
            args.append('np.asarray(data)')
        else:
            args.append('data*u.g')
        json_name = '{}_{}_create'.format(package, bs)
        run_perf(args, json_name)

for bs in base_setups:
    for op, ufunc in op_ufuncs.items():
        for bench, bench_name in [
                (op + r'(data1, data2)', op + '12.json'),
                (op + r'(data2, data1)', op + '21.json'),
                (ufunc + r'(data1, data2)', ufunc + '12.json'),
                (ufunc + r'(data2, data1)', ufunc + '21.json'),
                (ufunc + r'(data1, data2, out=out)', ufunc + '12out.json'),
                (ufunc + r'(data2, data1, out=out)', ufunc + '21out.json'),
        ]:
            for package in sorted(setup):
                print(package)
                setup_s = '; '.join(
                    [shared_setup, setup[package], base_setups[bs]]) + '; '
                if package == 'numpy':
                    setup_s += '; '.join(
                        [r'data1 = np.array(data)', r'data2 = np.array(data)'])
                    _bench = bench.replace('data1', '.001*data1')
                else:
                    setup_s += '; '.join(
                        ['data1 = data*u.g', 'data2 = data*u.kg'])
                    _bench = bench
                args = base_args + ['-s', setup_s + ' ']
                json_name = '{}_{}'.format(package, bs)
                run_perf(args + [_bench], json_name + '_' + bench_name)
    
