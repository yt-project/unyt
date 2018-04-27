import matplotlib
matplotlib.use('agg')
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
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
    ratios = OrderedDict()
    stddevs = OrderedDict()
    np_bench = perf.Benchmark.load(
        open('{}_{}'.format('numpy', extension), 'r'))
    np_mean = np_bench.mean()
    np_stddev = np_bench.stdev()
    for package in setup:
        if package == 'numpy':
            continue
        benchmark = perf.Benchmark.load(
            open('{}_{}'.format(package, extension), 'r'))
        mean = benchmark.mean()
        stddev = benchmark.stdev()
        ratios[package] = mean/np_mean
        stddevs[package] = ratios[package]*np.sqrt(
            (np_stddev/np_mean)**2 + (stddev/mean)**2)
    fig, ax = plt.subplots()
    packages = ratios.keys()
    ratios = ratios.values()
    stddevs = stddevs.values()
    ax.bar(packages, ratios, yerr=stddevs)
    fig.suptitle(extension.replace('.json', '').replace('_', ' ').title())
    ax.set_ylabel('numpy overhead (x time for numpy); lower is better')
    plt.savefig(extension.replace('.json', '.png'))
    plt.close(fig)


setup = OrderedDict([
    ('numpy', 'import numpy as np'),
    ('pint', 'from pint import UnitRegistry; u = UnitRegistry()'),
    ('astropy', 'import astropy.units as u'),
    ('unyt', 'import unyt as u'),
])

base_args = ['python3.6', '-m', 'perf', 'timeit']

shared_setup = 'import numpy as np; import operator'

base_setups = OrderedDict([
    ('small_list', 'data = [1., 2., 3.]'),
    ('small_tuple', 'data = (1., 2., 3.)'),
    ('small_array', 'data = np.array([1., 2., 3.])'),
    ('big_list', 'data = (np.arange(1e6)+1).tolist()'),
    ('big_array', 'data = (np.arange(1e6)+1)'),
])

op_ufuncs = OrderedDict([
    ('operator.add', 'np.add'),
    ('operator.sub', 'np.subtract'),
    ('operator.mul', 'np.multiply'),
    ('operator.truediv', 'np.true_divide'),
    ('operator.eq', 'np.equal'),
])

for bs in base_setups:
    for package in sorted(setup):
        print(package)
        setup_s = '; '.join([shared_setup, setup[package], base_setups[bs]])
        args = base_args + ['-s', setup_s + ' ']
        if package == 'numpy':
            args.append('np.array(data)')
        else:
            args.append('data*u.g')
        json_name = '{}_{}_create.json'.format(package, bs)
        run_perf(args, json_name)

        if 'list' in bs or 'tuple' in bs:
            continue

        args = base_args + ['-s', setup_s +
                            '; data=np.asarray(data); out=data.copy()']
        if package == 'numpy':
            args[-1] += '; '
        else:
            if package != 'pint':
                args[-1] += '*u.g'
            args[-1] += '; data = data*u.g '

        args.append('data**2')
        json_name = '{}_{}_square.json'.format(package, bs)
        run_perf(args, json_name)

        args[-1] = 'np.power(data, 2)'
        json_name = '{}_{}_npsquare.json'.format(package, bs)
        run_perf(args, json_name)

        args[-1] = 'np.power(data, 2, out=out)'
        json_name = '{}_{}_npsquareout.json'.format(package, bs)
        run_perf(args, json_name)

        args[-1] = 'data**0.5'
        json_name = '{}_{}_sqrt.json'.format(package, bs)
        run_perf(args, json_name)

        args[-1] = 'np.sqrt(data)'
        json_name = '{}_{}_npsqrt.json'.format(package, bs)
        run_perf(args, json_name)

        args[-1] = 'np.sqrt(data, out=out)'
        json_name = '{}_{}_npsqrtout.json'.format(package, bs)
        run_perf(args, json_name)



    make_plot("{}_create.json".format(bs))
    if 'list' not in bs and 'tuple' not in bs:
        make_plot("{}_square.json".format(bs))
        make_plot("{}_npsquare.json".format(bs))
        make_plot("{}_npsquareout.json".format(bs))
        make_plot("{}_sqrt.json".format(bs))
        make_plot("{}_npsqrt.json".format(bs))
        make_plot("{}_npsqrtout.json".format(bs))


for bs in base_setups:
    if 'list' in bs or 'tuple' in bs:
        continue
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
                        if (package not in ('pint', 'numpy') and
                            'equal' not in bench):
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
