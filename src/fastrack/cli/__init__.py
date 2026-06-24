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
    "cache_layout": "cache_layout", "frame_rate": "frame_rate_hz",
    "input_format": "input_format",
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


#: fastplus-only argparse dests -> flat DirectionalSettings field names.
_PLUS_CLI_TO_FIELD = {
    "mode": "mode", "head_channel": "head_channel",
    "filament_channel": "filament_channel",
    "head_sigma": "head_sigma", "head_radius": "head_radius",
    "head_quality": "head_quality", "head_subpixel": "head_subpixel",
    "head_tracker": "head_tracking_algorithm",
    "initial_search": "initial_search_radius",
    "kalman_search": "kalman_search_radius", "max_gap": "max_frame_gap",
    "end_fraction": "end_fraction", "max_end_distance": "max_end_distance_nm",
    "register": "register_channels", "channel_map": "channel_map",
    "perturb": "perturbation_times_s", "kinetic_model": "kinetic_model",
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
    parser.add_argument("--frame-rate", default=S, type=float, dest="frame_rate",
                        help="acquisition frame rate in Hz; forces uniform timing (overrides "
                             "metadata.txt). Required for TIFF-stack input, which carries no clock")
    parser.add_argument("--input-format", default=S, choices=["auto", "stack", "frames"],
                        dest="input_format",
                        help="movie input: auto-detect (default), or force 'stack' (each .tif is "
                             "a multi-page movie) / 'frames' (micro-manager frame folders)")
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


# --------------------------------------------------------------------------- #
# fastplus (directional / polarity-aware two-channel analysis)
# --------------------------------------------------------------------------- #
def fastplus_main(argv=None):
    usage = "\n".join([
        "%(prog)s -d [DIRECTORY] --mode {head-centric|filament-centric}",
        "-" * 72,
        "FASTplus: directional (polarity-aware) gliding-assay analysis.",
        "Two-channel polarity-labelled movies; signed (plus/minus-end) velocity.",
        "Two-channel registration needs:  pip install 'fastrack[plus]'",
        "-" * 72,
    ])
    S = argparse.SUPPRESS
    p = argparse.ArgumentParser(description="", usage=usage)
    p.add_argument("-d", help="top directory of the *RGB.tif movies to analyze")
    p.add_argument("--config", nargs="*", default=[], metavar="FILE.toml",
                   help="TOML config file(s) providing defaults; CLI flags override")

    # mode + channels
    p.add_argument("--mode", default=S, choices=["head-centric", "filament-centric"],
                   help="head-centric (track the labels, Default) or filament-centric")
    p.add_argument("--head-channel", default=S, choices=["red", "green", "blue"],
                   dest="head_channel", help="channel carrying the heads (Default: red)")
    p.add_argument("--filament-channel", default=S, choices=["red", "green", "blue"],
                   dest="filament_channel", help="channel carrying the filaments (Default: green)")
    p.add_argument("--channel-map", default=S, dest="channel_map",
                   help="e.g. 'red=heads,green=filaments' (overrides --head/--filament-channel)")

    # head detection (≈ TrackMate LoG)
    p.add_argument("--head-sigma", default=S, type=float, dest="head_sigma",
                   help="Gaussian pre-blur sigma (Default: 1.5)")
    p.add_argument("--head-radius", default=S, type=float, dest="head_radius",
                   help="estimated head radius in px (Default: 5.0)")
    p.add_argument("--head-quality", default=S, type=float, dest="head_quality",
                   help="LoG quality threshold (Default: 5.0)")
    p.add_argument("--no-head-subpixel", dest="head_subpixel", action="store_false",
                   default=S, help="disable subpixel localization")

    # head tracking (≈ TrackMate LinearMotionLAP)
    p.add_argument("--head-tracker", default=S, choices=["kalman-lap"],
                   dest="head_tracker", help="head linker (Default: kalman-lap)")
    p.add_argument("--initial-search", default=S, type=float, dest="initial_search",
                   help="initial linking search radius px (Default: 20)")
    p.add_argument("--kalman-search", default=S, type=float, dest="kalman_search",
                   help="Kalman search radius px once velocity known (Default: 15)")
    p.add_argument("--max-gap", default=S, type=int, dest="max_gap",
                   help="max frame gap for gap closing (Default: 4)")

    # association + disambiguation
    p.add_argument("--end-fraction", default=S, type=float, dest="end_fraction",
                   help="fraction of filament length counted as an 'end' (Default: 0.15)")
    p.add_argument("--max-end-distance", default=S, type=float, dest="max_end_distance",
                   help="max head-to-endpoint distance nm for association (Default: 500)")

    # registration
    p.add_argument("--register", action=argparse.BooleanOptionalAction, default=S,
                   dest="register", help="register the two channels via optomerge (Default: on)")

    # per-frame averaging + kinetics
    p.add_argument("--perturb", nargs="*", type=float, default=S, dest="perturb",
                   help="perturbation onset times in seconds (one per event)")
    p.add_argument("--kinetic-model", default=S, dest="kinetic_model",
                   choices=["none", "exp_rise", "exp_decay", "exp_rise_decay"],
                   help="kinetic model to fit to the per-frame mean signed velocity")

    # timing (stacks carry no clock; set one of these to get nm/s, else nm/frame)
    p.add_argument("--spf", default=S, type=float, dest="spf",
                   help="seconds per frame (e.g. 1.356 for '1356mspf' data)")
    p.add_argument("--frame-rate", default=S, type=float, dest="frame_rate_hz",
                   help="acquisition frame rate in Hz (alternative to --spf)")

    # subsetting (for quick tests on large datasets)
    p.add_argument("--limit", default=S, type=int, dest="limit",
                   help="process at most N movies")
    p.add_argument("--max-frames", default=S, type=int, dest="max_frames",
                   help="use only the first N frames of each movie")
    p.add_argument("--frame-step", default=S, type=int, dest="frame_step",
                   help="temporal subsampling: analyse every Kth frame")
    p.add_argument("--output", default=S, dest="output_dir",
                   help="output directory (Default: <main_dir>/fastplus_out)")

    # QC overlay (heads coloured by polarity classification)
    p.add_argument("--overlay", action="store_true", default=S, dest="overlay",
                   help="write qc_overlay.png/.mp4 per movie: heads coloured by "
                        "classification (green=plus_end, red=both_ends, "
                        "orange=middle, grey=none)")
    p.add_argument("--overlay-fps", default=S, type=int, dest="overlay_fps",
                   help="frame rate for qc_overlay.mp4 (Default: 10)")
    p.add_argument("--montage-frames", default=S, type=int, dest="montage_frames",
                   help="number of frames in qc_overlay.png (Default: 12)")

    # shared hardware / runtime
    p.add_argument("-px", default=S, type=float, dest="px", help="pixel size in nm (Default: 80.65)")
    p.add_argument("-minv", default=S, type=float, dest="minv",
                   help="stuck-velocity threshold nm/s (Default: 80)")
    p.add_argument("--detector", default=S, choices=["entropy", "ridge", "ridge-fast"],
                   dest="detector", help="filament detector (Default: entropy)")
    p.add_argument("-j", default=S, type=int, dest="j", help="worker processes (Default: all)")
    p.add_argument("-v", action="store_true", default=S, dest="v", help="verbose output")
    args = p.parse_args(argv)

    from ..config import Settings
    settings = Settings.from_toml(*args.config) if args.config else Settings()

    mapping = {**_CLI_TO_FIELD, **_PLUS_CLI_TO_FIELD}
    overrides = {mapping[d]: v for d, v in vars(args).items() if d in mapping}
    # perturbation times arrive as a list; DirectionalSettings stores a tuple.
    if overrides.get("perturbation_times_s") is not None:
        overrides["perturbation_times_s"] = tuple(overrides["perturbation_times_s"])
    # --spf is sugar for the hardware frame rate (Hz); explicit --frame-rate wins.
    if getattr(args, "spf", None) and not getattr(args, "frame_rate_hz", None):
        overrides["frame_rate_hz"] = 1.0 / args.spf
    elif getattr(args, "frame_rate_hz", None):
        overrides["frame_rate_hz"] = args.frame_rate_hz
    settings = settings.with_overrides(**overrides)

    # run-only knobs (not Settings fields) passed straight through
    run_only = {k: getattr(args, k) for k in
                ("limit", "max_frames", "frame_step", "output_dir",
                 "overlay", "overlay_fps", "montage_frames")
                if getattr(args, k, S) is not S}

    from ..pipelines import directional
    directional.run(main_dir=args.d, **settings.to_directional_kwargs(), **run_only)
