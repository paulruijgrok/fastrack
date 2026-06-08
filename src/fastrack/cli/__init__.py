"""Command-line entry points for FAST, LIMA and stack2tifs.

These mirror the argparse interfaces of the original ``bin/fast``, ``bin/lima``
and ``bin/stack2tifs`` scripts, but dispatch to the importable functions in this
package instead of relying on ``ppss`` for parallelism.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# fast
# --------------------------------------------------------------------------- #
def fast_main(argv=None):
    usage = "\n".join([
        "%(prog)s -d [DIRECTORY]",
        "-" * 72,
        "FAST v2.0: Fast Actin filament Spud Trekker (Python 3 port)",
        "Original: Tural Aksel.  Modernized port.",
        "",
        "FAST provides fast and accurate analysis of actin gliding assay movies.",
        "-" * 72,
    ])
    parser = argparse.ArgumentParser(description="", usage=usage)
    parser.add_argument("-d", help="top directory of the movies to be analyzed")
    parser.add_argument("-f", action="store_true", default=False, help="force analyze all the movies")
    parser.add_argument("-r", action="store_true", default=False, help="recalculate instantaneous velocities from saved filament files")
    parser.add_argument("-m", action="store_true", default=False, help="make filament tracking movie")
    parser.add_argument("-p", default=5, type=int, help="minimum length for the paths to be analyzed (Default:5)")
    parser.add_argument("-n", default=5, type=int, help="number of consecutive frames for averaging (Default:5)")
    parser.add_argument("-pt", default=500, type=int, help="percent tolerance (Default:none)")
    parser.add_argument("-px", default=80.65, type=float, help="pixel size in nm (Default:80.65 nm)")
    parser.add_argument("-ymax", default=1500, type=int, help="maximum velocity for the plot in nm/s (Default:1500)")
    parser.add_argument("-xmax", default=10000, type=int, help="maximum length for the plot in nm (Default:10000)")
    parser.add_argument("-cl", default="b", type=str, help="color for maximum velocity points (Default:blue)")
    parser.add_argument("-fx", default="none", choices=["none", "exp", "uyeda"], type=str, help="function to be fitted to maximum velocity data")
    parser.add_argument("-maxd", default=2016.25, type=float, help="maximum allowed distance in nm between adjacent frames for a filament (Default:2016.25 nm)")
    parser.add_argument("-minv", default=80, type=float, help="minimum average path velocity for a filament to be classified as stuck (Default:80 nm/s)")
    parser.add_argument("-oscore", default=0.4, type=float, help="overlap score cutoff (advanced, Default:0.4)")
    parser.add_argument("-lascore", default=1.0, type=float, help="log-area score cutoff (advanced, Default:1.0)")
    parser.add_argument("-dlascore", default=0.5, type=float, help="difference-log-area score cutoff (advanced, Default:0.5)")
    parser.add_argument("-j", default=None, type=int, help="number of parallel worker processes (Default: cpu_count-1)")
    parser.add_argument("--legacy-linking", action="store_true", default=False,
                        help="reproduce the original Python 2 frame-linking behaviour (leftover-loop-variable "
                             "partner selection) for bit-for-bit reproduction of published results")
    parser.add_argument("--exact-rank", "--no-fast-rank", dest="fast_rank",
                        action="store_false", default=True,
                        help="run the full-frame percentile filters on the native 16-bit data "
                             "instead of the default 8-bit rescaling. Slower but exact; use this "
                             "for the reference/validation path. (Default: fast 8-bit rank filters.)")
    parser.add_argument("--morph-contrast", action="store_true", default=False,
                        help="compute the local-contrast map with a one-pass morphological gradient "
                             "(local max-min) instead of two percentile passes. Faster but more "
                             "noise-sensitive; off by default, A/B with compare_fast_rank.py")
    parser.add_argument("-v", action="store_true", default=False, help="verbose output for debugging")
    args = parser.parse_args(argv)

    from ..pipelines import gliding
    gliding.run(
        main_dir=args.d,
        force_analysis=args.f,
        recalculate=args.r,
        make_movie=args.m,
        min_path_length=args.p,
        num_frames_ave=args.n,
        percent_tolerance=args.pt,
        pixel_size=args.px,
        plot_ymax=args.ymax,
        plot_xmax=args.xmax,
        maxvel_color=args.cl,
        fit_function=args.fx,
        max_velocity=args.maxd,
        min_velocity=args.minv,
        overlap_score_cutoff=args.oscore,
        log_area_score_cutoff=args.lascore,
        diff_log_area_score_cutoff=args.dlascore,
        legacy_linking=args.legacy_linking,
        fast_rank=args.fast_rank,
        morph_contrast=args.morph_contrast,
        nprocs=args.j,
        verbose=args.v,
    )


# --------------------------------------------------------------------------- #
# lima
# --------------------------------------------------------------------------- #
def lima_main(argv=None):
    usage = "\n".join([
        "%(prog)s -d [DIRECTORY]",
        "-" * 72,
        "LIMA v2.0: Loaded In vitro Motility Assay (Python 3 port)",
        "-" * 72,
    ])
    parser = argparse.ArgumentParser(description="", usage=usage)
    parser.add_argument("-d", default=None, help="top directory of the output files to be analyzed")
    parser.add_argument("-amin", default=0.0, type=float, help="minimum load concentration for analysis")
    parser.add_argument("-amax", default=0.0, type=float, help="maximum load concentration for analysis")
    parser.add_argument("-pmin", default=0.0, type=float, help="minimum load concentration for plotting")
    parser.add_argument("-pmax", default=0.0, type=float, help="maximum load concentration for plotting")
    parser.add_argument("-p", nargs="*", default=None, type=str, help="protein names to be analyzed")
    parser.add_argument("-r", default=True, action="store_false", help="print fit parameters")
    parser.add_argument("-g", default=None, nargs="*", type=float, help="initial guess for the parameters to be fitted")
    parser.add_argument("-cl", nargs="*", default=None, type=str, help="plotting colors")
    args = parser.parse_args(argv)

    from ..pipelines import loaded
    loaded.run(
        main_dir=args.d,
        min_load_analysis=args.amin,
        max_load_analysis=args.amax,
        min_load_plot=args.pmin,
        max_load_plot=args.pmax,
        protein_names=args.p,
        print_params=args.r,
        init_params=args.g,
        colors=args.cl,
    )


# --------------------------------------------------------------------------- #
# stack2tifs
# --------------------------------------------------------------------------- #
def stack2tifs_main(argv=None):
    usage = "\n".join([
        "%(prog)s -d [DIRECTORY]",
        "-" * 72,
        "stack2tifs: convert TIFF stacks to micro-manager frame files (Python 3 port)",
        "-" * 72,
    ])
    parser = argparse.ArgumentParser(description="", usage=usage)
    parser.add_argument("-d", help="top directory of the stack files to be converted")
    parser.add_argument("-s", default=6, type=float, help="lower bound for tif file size in MB (Default:6)")
    parser.add_argument("-f", default=1, type=float, help="frame rate for the movies (Default:1)")
    args = parser.parse_args(argv)

    from ..io import convert
    convert.run(main_dir=args.d, min_size=args.s, frame_rate=args.f)
