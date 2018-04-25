import matplotlib
matplotlib.use('agg')
from collections import OrderedDict
from matplotlib import pyplot as plt
import json
import os
import perf
import subprocess

def run_perf(args, json_name):
    if os.path.exists(json_name):
        return
    args = args + ['-o', json_name]
    print(args)
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print(out.decode())
    print(err.decode())


def make_plot(extension):
    means = OrderedDict()
    stddevs = OrderedDict()
    for package in setup:
        benchmark = perf.Benchmark.load(
            open('{}_{}'.format(package, extension), 'r'))
        means[package] = benchmark.mean()
        stddevs[package] = benchmark.stdev()
    fig, ax = plt.subplots()
    packages = means.keys()
    means = means.values()
    stddevs = stddevs.values()
    ax.bar(packages, means, yerr=stddevs)
    fig.suptitle(extension.replace('.json', '').replace('_', ' ').title())
    ax.set_ylabel('time (s); lower is better')
    plt.savefig(extension.replace('.json', '.png'))


setup = OrderedDict([
    ('numpy', 'import numpy as np'),
    ('pint', 'from pint import UnitRegistry; u = UnitRegistry()'),
    ('astropy', 'import astropy.units as u'),
    ('unyt', 'import unyt as u'),
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
        json_name = '{}_{}_create.json'.format(package, bs)
        run_perf(args, json_name)
    make_plot("{}_create.json".format(bs))

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
            for unit_choice in [('g', 'g'), ('kg', 'g')]:
                for package in sorted(setup):
                    print(package)
                    setup_s = '; '.join(
                        [shared_setup, setup[package], base_setups[bs]]) + '; '
                    if 'out' in bench:
                        if package not in ('pint', 'numpy'):
                            setup_s += 'out=data*u.{}; '.format(unit_choice[0])
                        else:
                            setup_s += 'out=np.array(data); '
                    if package == 'numpy':
                        setup_s += '; '.join([r'data1 = np.array(data)',
                                              r'data2 = np.array(data)'])
                        if unit_choice[0] != unit_choice[1]:
                            _bench = bench.replace('data1', '.001*data1')
                    else:
                        setup_s += '; '.join(
                            ['data1 = data*u.{}'.format(unit_choice[0]),
                             'data2 = data*u.{}'.format(unit_choice[1])])
                        _bench = bench
                    args = base_args + ['-s', setup_s + ' ']
                    json_name = '{}_{}_{}{}'.format(
                        package, bs, unit_choice[0], unit_choice[1])
                    run_perf(args + [_bench], json_name + '_' + bench_name)
                make_plot("{}_{}{}_{}".format(
                    bs, unit_choice[0], unit_choice[1], bench_name))
    
