import numpy as np
import perf
from collections import OrderedDict
from matplotlib import pyplot as plt


def make_plot(benchmark_name, benchmarks, fig_filename):
    plt.style.use('tableau-colorblind10')
    plt.rc('font', family='stixgeneral')
    plt.rc('mathtext', fontset='cm')
    fig, ax = plt.subplots()

    ratios = OrderedDict()
    stddevs = OrderedDict()

    width = 0.2
    packages = ['pint', 'astropy', 'unyt']
    offsets = [-width, 0, width]

    for ind, benchmark in enumerate(benchmarks):
        with open('../benchmarks/numpy_{}.json'.format(benchmark), 'r') as f:
            np_bench = perf.Benchmark.load(f)
        np_mean = np_bench.mean()
        np_stddev = np_bench.stdev()
        for package, offset in zip(packages, offsets):
            fname = '../benchmarks/{}_{}.json'.format(package, benchmark)
            with open(fname, 'r') as f:
                pbench = perf.Benchmark.load(f)
            mean = pbench.mean()
            stddev = pbench.stdev()
            ratios[package] = mean/np_mean
            stddevs[package] = ratios[package]*np.sqrt(
                (np_stddev/np_mean)**2 + (stddev/mean)**2)
            color_name = 'C{}'.format(packages.index(package))
            ax.barh(ind + 1 + offset, ratios[package], width,
                    xerr=stddevs[package], color=color_name, label=package)

    ax.legend(packages)
    ax.set_xlabel(r'$T_{\rm package} / T_{\rm numpy}$')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(0, 100)
    ax.set_yticks(np.arange(len(benchmarks))+1)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   2, 3, 4, 5, 6, 7, 8, 9,
                   20, 30, 40, 50, 60, 70, 80, 90],
                  minor=True)
    ax.set_xticks([0, 1, 10, 100])
    ax.set_xticklabels(['0', '1', '10', '100'])
    ax.set_yticklabels(benchmarks.values())
    ax.plot([1, 1], [0, len(benchmarks)+1], '--', color='k', lw=0.75,
            alpha=0.5)
    ax.set_ylim(0.5, len(benchmarks)+0.5)
    fig.suptitle(benchmark_name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    plt.close(fig)


make_plot(
    benchmark_name='Apply units to data',
    benchmarks={
        'small_list_create': '3-element list',
        'medium_list_create': '1,000-element list',
        'big_list_create': '1,000,000-element list',
        'small_array_create': '3-element ndarray',
        'medium_array_create': '1000-element ndarray',
        'big_array_create': '1,000,000-element ndarray',
    },
    fig_filename='apply.png',
)

make_plot(
    benchmark_name='Unary operations',
    benchmarks={
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
    },
    fig_filename='unary.png'
)
