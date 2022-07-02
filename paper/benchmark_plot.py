from collections import OrderedDict

import numpy as np
import perf
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

ALPHA_MAP = {"small": 0.3333, "medium": 0.666, "big": 1.0}
COLOR_MAP = {"unyt": "C0", "astropy": "C1", "pint": "C2"}
SIZE_LABELS = {"small": "3", "medium": "$10^3$", "big": "$10^6$"}


def make_plot(benchmark_name, benchmarks, fig_filename):
    plt.style.use("tableau-colorblind10")
    plt.rc("font", family="stixgeneral")
    plt.rc("mathtext", fontset="cm")
    fig, ax = plt.subplots()

    ratios = OrderedDict()
    stddevs = OrderedDict()

    width = 0.1
    packages = ["pint", "astropy", "unyt"]
    package_offsets = [-width, 0, width]
    size_offsets = [-width * 3, 0, width * 3]
    static_offset = 0
    sizes = ["big", "medium", "small"]
    yticks = []
    all_yticks = []

    for ind, benchmark in enumerate(benchmarks):
        if "create" not in benchmark:
            benchmark = "array_" + benchmark
        for size, size_offset in zip(sizes, size_offsets):
            np_fname = "../benchmarks/numpy_{}_{}.json".format(size, benchmark)
            with open(np_fname, "r") as f:
                np_bench = perf.Benchmark.load(f)
            np_mean = np_bench.mean()
            np_stddev = np_bench.stdev()
            for package, package_offset in zip(packages, package_offsets):
                fname = "../benchmarks/{}_{}_{}.json".format(package, size, benchmark)
                with open(fname, "r") as f:
                    pbench = perf.Benchmark.load(f)
                mean = pbench.mean()
                stddev = pbench.stdev()
                ratios[package] = mean / np_mean
                stddevs[package] = ratios[package] * np.sqrt(
                    (np_stddev / np_mean) ** 2 + (stddev / mean) ** 2
                )
                ytick = ind + 1 + size_offset + package_offset + static_offset
                ax.barh(
                    ytick,
                    ratios[package],
                    width,
                    xerr=stddevs[package],
                    color=COLOR_MAP[package],
                    label=" ".join([package, SIZE_LABELS[size]]),
                    alpha=ALPHA_MAP[size],
                )
                all_yticks.append(ytick)
                if package == "astropy" and size == "medium":
                    yticks.append(ytick)
            static_offset += 0.01

    legend_patches = [Patch(color=c) for c in COLOR_MAP.values()]
    leg = ax.legend(legend_patches, COLOR_MAP.keys(), loc=1)
    size_patches = [Patch(color="k", alpha=a) for a in ALPHA_MAP.values()]
    size_labels = [SIZE_LABELS[l] for l in ALPHA_MAP.keys()]
    ax.legend(size_patches, size_labels, loc=4, title="Number of elements")
    ax.add_artist(leg)
    ax.set_xlabel(r"$T_{\rm package} / T_{\rm numpy}$")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 1000)
    ax.set_yticks(yticks)
    ax.set_xticks(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
        ],
        minor=True,
    )
    ax.set_xticks([0, 1, 10, 100, 1000])
    ax.set_xticklabels(["0", "1", "10", "100", "1000"])
    ax.set_yticklabels(benchmarks.values())
    ax.plot([1, 1], [0, len(benchmarks) + 1], "--", color="k", lw=0.75, alpha=0.5)
    spacing = len(benchmarks) // 2 + 1
    ax.set_ylim(all_yticks[0] - 0.05 * spacing, all_yticks[-1] + 0.05 * spacing)
    fig.suptitle(benchmark_name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    plt.close(fig)


make_plot(
    benchmark_name="Apply units to data",
    benchmarks={"list_create": "List", "array_create": "Array"},
    fig_filename="apply.png",
)

make_plot(
    benchmark_name="Unary operations",
    benchmarks={"sqrt": r"$\mathtt{data**0.5}$", "square": r"$\mathtt{data**2}$"},
    fig_filename="unary.png",
)

make_plot(
    benchmark_name="Binary operations, different units",
    benchmarks={
        "kgg_operator.add12": r"$\mathtt{a + b}$",
        "kgg_operator.sub12": r"$\mathtt{a - b}$",
        "kgg_operator.mul12": r"$\mathtt{a * b}$",
        "kgg_operator.truediv12": r"$\mathtt{a / b}$",
        "kgg_operator.eq12": r"$\mathtt{a == b}$",
    },
    fig_filename="binary_different_units.png",
)

make_plot(
    benchmark_name="Binary operations, same units",
    benchmarks={
        "gg_operator.add12": r"$\mathtt{a + b}$",
        "gg_operator.sub12": r"$\mathtt{a - b}$",
        "gg_operator.mul12": r"$\mathtt{a * b}$",
        "gg_operator.truediv12": r"$\mathtt{a / b}$",
        "gg_operator.eq12": r"$\mathtt{a == b}$",
    },
    fig_filename="binary_same_units.png",
)

make_plot(
    benchmark_name="NumPy ufunc",
    benchmarks={
        "kgg_np.add12": r"$\mathtt{np.add(a, b)}$",
        "kgg_np.subtract12": r"$\mathtt{np.subtract(a, b)}$",
        "kgg_np.multiply12": r"$\mathtt{np.multiply(a, b)}$",
        "kgg_np.true_divide12": r"$\mathtt{np.divide(a, b)}$",
        "kgg_np.equal12": r"$\mathtt{np.equal(a, b)}$",
        "npsqrt": r"$\mathtt{np.sqrt(data)}$",
        "npsquare": r"$\mathtt{np.power(data, 2)}$",
    },
    fig_filename="ufunc.png",
)

make_plot(
    benchmark_name="In-place ufunc",
    benchmarks={
        "kgg_np.add12out": r"$\mathtt{np.add(a, b, out=out)}$",
        "kgg_np.subtract12out": r"$\mathtt{np.subtract(a, b, out=out)}$",
        "kgg_np.multiply12out": r"$\mathtt{np.multiply(a, b, out=out)}$",
        "kgg_np.true_divide12out": r"$\mathtt{np.divide(a, b, out=out)}$",
        "kgg_np.equal12out": r"$\mathtt{np.equal(a, b, out=out)}$",
        "npsqrtout": r"$\mathtt{np.sqrt(data, out=out)}$",
        "npsquareout": r"$\mathtt{np.power(data, 2, out=out)}$",
    },
    fig_filename="ufuncout.png",
)
