"""Command-line entry points for FAST, LIMA and stack2tifs.

These mirror the argparse interfaces of the original ``bin/fast``, ``bin/lima``
and ``bin/stack2tifs`` scripts, but dispatch to the importable functions in this
package instead of relying on ``ppss`` for parallelism.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")


#: Maps ``fast`` argparse dests to flat Settings field names (unique across
#: sections).  Only flags the user explicitly passes appear in the namespace
#: (every flag defaults to argparse.SUPPRESS), so this is the override layer on
#: top of any --config file.
_CLI_TO_FIELD = {
    "f": "force_analysis", "r": "recalculate", "m": "make_movie",
    "overlay_movie": "overlay_movie",
    "p": "min_path_length", "n": "num_frames_ave", "pt": "percent_tolerance",
    "px": "pixel_size_nm", "ymax": "ymax", "xmax": "xmax", "cl": "maxvel_color",
    "fx": "fit_function", "maxd": "max_inter_frame_distance_nm",
    "minv": "stuck_velocity_nm_s", "oscore": "overlap_score_cutoff",
    "lascore": "log_area_score_cutoff", "dlascore": "diff_log_area_score_cutoff",
    "legacy_linking": "legacy_linking", "fast_rank": "fast_rank",
    "morph_contrast": "morph_contrast", "detector": "detection_algorithm",
    "cache_layout": "cache_layout",
    "export_trajectories": "export_trajectories", "export_contours": "export_contours",
    "j": "nprocs", "v": "verbose",
    "ridge_line_widths": "line_widths", "ridge_low_contrast": "low_contrast",
    "ridge_high_contrast": "high_contrast", "ridge_min_len": "min_len",
    "ridge_max_len": "max_len", "ridge_dark_line": "dark_line",
    "ridge_estimate_width": "estimate_width",
    "overlay_fps": "fps", "overlay_frame_label": "frame_label",
    "overlay_time_label": "time_label", "overlay_frame_interval": "frame_interval_s",
    "overlay_font_scale": "font_scale",
}


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
    # Every analysis/overlay flag defaults to SUPPRESS so that only flags the
    # user *explicitly* passes appear in the namespace.  That lets a --config
    # TOML provide the base values and CLI flags override just what they set.
    S = argparse.SUPPRESS
    parser = argparse.ArgumentParser(description="", usage=usage)
    parser.add_argument("-d", help="top directory of the movies to be analyzed")
    parser.add_argument("--config", nargs="*", default=[], metavar="FILE.toml",
                        help="TOML config file(s) providing defaults; later files and explicit "
                             "CLI flags override earlier ones (requires Python 3.11+)")
    parser.add_argument("-f", action="store_true", default=S, help="force analyze all the movies")
    parser.add_argument("-r", action="store_true", default=S, help="recalculate instantaneous velocities from saved filament files")
    parser.add_argument("-m", action="store_true", default=S, help="make filament tracking movie")
    parser.add_argument("--overlay-movie", action="store_true", default=S,
                        help="make an overlay movie: original frames with tracked filaments drawn "
                             "on top, colored green (moving) / red (stuck). Produces overlay_tracks.mp4")
    parser.add_argument("-p", default=S, type=int, help="minimum length for the paths to be analyzed (Default:5)")
    parser.add_argument("-n", default=S, type=int, help="number of consecutive frames for averaging (Default:5)")
    parser.add_argument("-pt", default=S, type=int, help="percent tolerance (Default:none)")
    parser.add_argument("-px", default=S, type=float, help="pixel size in nm (Default:80.65 nm)")
    parser.add_argument("-ymax", default=S, type=int, help="maximum velocity for the plot in nm/s (Default:1500)")
    parser.add_argument("-xmax", default=S, type=int, help="maximum length for the plot in nm (Default:10000)")
    parser.add_argument("-cl", default=S, type=str, help="color for maximum velocity points (Default:blue)")
    parser.add_argument("-fx", default=S, choices=["none", "exp", "uyeda"], type=str, help="function to be fitted to maximum velocity data")
    parser.add_argument("-maxd", default=S, type=float, help="maximum allowed distance in nm between adjacent frames for a filament (Default:2016.25 nm)")
    parser.add_argument("-minv", default=S, type=float, help="minimum average path velocity for a filament to be classified as stuck (Default:80 nm/s)")
    parser.add_argument("-oscore", default=S, type=float, help="overlap score cutoff (advanced, Default:0.4)")
    parser.add_argument("-lascore", default=S, type=float, help="log-area score cutoff (advanced, Default:1.0)")
    parser.add_argument("-dlascore", default=S, type=float, help="difference-log-area score cutoff (advanced, Default:0.5)")
    parser.add_argument("-j", default=S, type=int, help="number of parallel worker processes (Default: all cores)")
    parser.add_argument("--legacy-linking", action="store_true", default=S,
                        help="reproduce the original Python 2 frame-linking behaviour")
    parser.add_argument("--exact-rank", "--no-fast-rank", dest="fast_rank",
                        action="store_false", default=S,
                        help="use the exact 16-bit percentile filters instead of the default 8-bit path")
    parser.add_argument("--morph-contrast", action="store_true", default=S,
                        help="one-pass morphological-gradient contrast instead of two percentile passes")
    parser.add_argument("--cache-layout", default=S, choices=["per-frame", "per-movie"],
                        dest="cache_layout",
                        help="intermediate filXYs cache: 'per-frame' (one .npy per frame, default) "
                             "or 'per-movie' (a single .npz per movie)")
    parser.add_argument("--export-trajectories", action="store_true", default=S,
                        dest="export_trajectories",
                        help="write a tidy per-movie trajectory CSV (one row per filament per "
                             "frame; physical units; group by path_id) for downstream analysis")
    parser.add_argument("--export-contours", action="store_true", default=S,
                        dest="export_contours",
                        help="with --export-trajectories, also write the skeleton geometry as a "
                             "long-format contour CSV (one row per contour point)")
    parser.add_argument("--detector", default=S, choices=["entropy", "ridge", "ridge-fast"],
                        dest="detector",
                        help="filament detection algorithm (Default: entropy). 'ridge' needs "
                             "pip install 'fastrack[ridge]'; 'ridge-fast' is the ~4x faster, "
                             "numerically-identical drop-in: pip install 'fastrack[ridge-fast]'")
    # Ridge-detector parameters (used only with --detector ridge).
    parser.add_argument("--ridge-line-widths", nargs="*", type=int, default=S, dest="ridge_line_widths")
    parser.add_argument("--ridge-low-contrast", type=float, default=S, dest="ridge_low_contrast")
    parser.add_argument("--ridge-high-contrast", type=float, default=S, dest="ridge_high_contrast")
    parser.add_argument("--ridge-min-len", type=float, default=S, dest="ridge_min_len")
    parser.add_argument("--ridge-max-len", type=float, default=S, dest="ridge_max_len")
    parser.add_argument("--ridge-dark-line", action="store_true", default=S, dest="ridge_dark_line")
    parser.add_argument("--ridge-no-width", dest="ridge_estimate_width", action="store_false", default=S)
    # Overlay-movie styling.
    parser.add_argument("--overlay-fps", type=float, default=S, dest="overlay_fps",
                        help="overlay movie playback frame rate (Default:10)")
    parser.add_argument("--frame-label", action=argparse.BooleanOptionalAction, default=S,
                        dest="overlay_frame_label", help="show frame number on overlay (bottom-left)")
    parser.add_argument("--time-label", action=argparse.BooleanOptionalAction, default=S,
                        dest="overlay_time_label", help="show mm:ss time on overlay (bottom-right)")
    parser.add_argument("--frame-interval", type=float, default=S, dest="overlay_frame_interval",
                        help="seconds per frame for the time label when no metadata (Default:1.0)")
    parser.add_argument("--overlay-font-scale", type=float, default=S, dest="overlay_font_scale",
                        help="overlay label font scale (Default:0.6)")
    parser.add_argument("-v", action="store_true", default=S, help="verbose output for debugging")
    args = parser.parse_args(argv)

    from ..config import Settings

    # Base settings from config file(s), then override with explicitly-passed flags.
    settings = Settings.from_toml(*args.config) if args.config else Settings()

    overrides = {_CLI_TO_FIELD[dest]: val for dest, val in vars(args).items()
                 if dest in _CLI_TO_FIELD}
    settings = settings.with_overrides(**overrides)

    from ..pipelines import gliding
    gliding.run(main_dir=args.d, **settings.to_run_kwargs())


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


# --------------------------------------------------------------------------- #
# fast-batch (unattended multi-dataset runner)
# --------------------------------------------------------------------------- #
def fast_batch_main(argv=None):
    usage = "\n".join([
        "%(prog)s MANIFEST [options]",
        "-" * 72,
        "fast-batch: run FAST over many datasets unattended (overnight-safe).",
        "",
        "MANIFEST is a .csv/.tsv/.xlsx table with a base-directory column and an",
        "optional 'config' (TOML) and 'name' column, one row per dataset. Each",
        "dataset runs through the analysis; failures are logged and the run moves",
        "on. A state file records successes so re-runs skip finished datasets.",
        "-" * 72,
    ])
    parser = argparse.ArgumentParser(description="", usage=usage)
    parser.add_argument("manifest", help="dataset list (.csv/.tsv/.xlsx)")
    parser.add_argument("--state", default=None,
                        help="resume state file (default: <logdir>/batch_state.json)")
    parser.add_argument("--logdir", default="fastrack_batch_logs",
                        help="directory for run + per-dataset logs (Default: fastrack_batch_logs)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="only run the pre-flight checks; do not process anything")
    parser.add_argument("--smoke", action="store_true",
                        help="pre-flight also detects frame 0 of each dataset (slower, catches "
                             "detector/dependency errors early)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="re-run every dataset, ignoring saved state")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-run datasets previously marked failed (unchanged ones are "
                             "otherwise skipped)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="abort the whole run on the first failure (default: keep going)")
    parser.add_argument("-j", default=None, type=int, dest="nprocs",
                        help="worker processes per dataset (Default: all cores)")
    parser.add_argument("--num-shards", default=1, type=int,
                        help="split the manifest into this many slices (for HPC job arrays)")
    parser.add_argument("--shard-index", default=0, type=int,
                        help="0-based slice this invocation should process (with --num-shards)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose console output")
    args = parser.parse_args(argv)

    from ..pipelines import batch
    batch.run_batch(
        manifest=args.manifest,
        state=args.state,
        logdir=args.logdir,
        force=args.force,
        retry_failed=args.retry_failed,
        preflight_only=args.preflight_only,
        smoke=args.smoke,
        nprocs=args.nprocs,
        stop_on_error=args.stop_on_error,
        verbose=args.verbose,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
