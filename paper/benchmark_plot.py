import numpy as np
import perf
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

HATCH_MAP = {'small': '\\\\\\', 'medium': '...', 'big': 'xxx'}
COLOR_MAP = {'unyt': 'C0', 'astropy': 'C1', 'pint': 'C2'}
SIZE_LABELS = {'small': '3 elements', 'medium': '1,000 elements',
               'big': '1,000,000 elements'}


def make_plot(benchmark_name, benchmarks, fig_filename):
    plt.style.use('tableau-colorblind10')
    plt.rc('font', family='stixgeneral')
    plt.rc('mathtext', fontset='cm')
    fig, ax = plt.subplots()

    ratios = OrderedDict()
    stddevs = OrderedDict()

    width = 0.1
    packages = ['pint', 'astropy', 'unyt']
    package_offsets = [-width, 0, width]
    size_offsets = [-width*3, 0, width*3]
    static_offset = 0
    sizes = ['big', 'medium', 'small']
    yticks = []
    all_yticks = []

    for ind, benchmark in enumerate(benchmarks):
        if 'create' not in benchmark:
            benchmark = 'array_' + benchmark
        for size, size_offset in zip(sizes, size_offsets):
            np_fname = '../benchmarks/numpy_{}_{}.json'.format(size, benchmark)
            with open(np_fname, 'r') as f:
                np_bench = perf.Benchmark.load(f)
            np_mean = np_bench.mean()
            np_stddev = np_bench.stdev()
            for package, package_offset in zip(packages, package_offsets):
                fname = '../benchmarks/{}_{}_{}.json'.format(
                    package, size, benchmark)
                with open(fname, 'r') as f:
                    pbench = perf.Benchmark.load(f)
                mean = pbench.mean()
                stddev = pbench.stdev()
                ratios[package] = mean/np_mean
                stddevs[package] = ratios[package]*np.sqrt(
                    (np_stddev/np_mean)**2 + (stddev/mean)**2)
                ytick = ind + 1 + size_offset + package_offset + static_offset
                ax.barh(ytick, ratios[package], width, xerr=stddevs[package],
                        color=COLOR_MAP[package], label=package,
                        hatch=HATCH_MAP[size])
                all_yticks.append(ytick)
                if package == 'astropy' and size == 'medium':
                    yticks.append(ytick)
            static_offset += .01

    size_patches = [Patch(hatch=h, fc='w', ec='k') for h in HATCH_MAP.values()]
    size_labels = [SIZE_LABELS[h] for h in HATCH_MAP.keys()]
    package_patches = [Patch(color=c) for c in COLOR_MAP.values()]
    leg = ax.legend(package_patches, COLOR_MAP.keys(), loc=1)
    ax.legend(size_patches, size_labels, loc=4)
    ax.add_artist(leg)
    ax.set_xlabel(r'$T_{\rm package} / T_{\rm numpy}$')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(0, 450)
    ax.set_yticks(yticks)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   2, 3, 4, 5, 6, 7, 8, 9,
                   20, 30, 40, 50, 60, 70, 80, 90,
                   200, 300, 400],
                  minor=True)
    ax.set_xticks([0, 1, 10, 100])
    ax.set_xticklabels(['0', '1', '10', '100'])
    ax.set_yticklabels(benchmarks.values())
    ax.plot([1, 1], [0, len(benchmarks)+1], '--', color='k', lw=0.75,
            alpha=0.5)
    spacing = (len(benchmarks)//2 + 1)
    ax.set_ylim(all_yticks[0] - 0.05*spacing, all_yticks[-1] + 0.05*spacing)
    fig.suptitle(benchmark_name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    plt.close(fig)


make_plot(
    benchmark_name='Apply units to data',
    benchmarks={
        'list_create': "List",
        'array_create': "Array"
    },
    fig_filename='apply.png',
)

make_plot(
    benchmark_name='Unary operations',
    benchmarks={
        'npsqrt': 'np.sqrt(data)',
        'sqrt': '(data)**(0.5)',
        'npsquare': 'np.power(data, 2)',
        'square': 'data**2',
    },
    fig_filename='unary.png'
)
