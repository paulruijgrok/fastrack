"""FAST analysis driver (modernized Python 3 port of ``bin/fast``).

The original driver parallelised per-frame filament extraction by writing a
``.py`` worker script and a ``.in`` frame-list per folder, then shelling out to
the ``ppss`` bash tool to run those scripts across cores.  This module replaces
that machinery with :mod:`multiprocessing`: a top-level worker function extracts
one frame and is mapped over the frame list with a process pool.  Everything
else (link building, path processing, plotting, combined statistics) mirrors the
original logic.
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np

from . import motility

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Per-frame worker (runs in a child process)
# --------------------------------------------------------------------------- #
def _extract_frame(task):
    """Extract filaments from a single frame and persist them.

    ``task`` is a tuple ``(directory, header, tail, frame_no, force)``.  The
    worker changes into ``directory`` is *not* required: the Motility object is
    given the directory directly, matching how the original per-folder script
    set ``new_Motility.directory``.
    """
    directory, header, tail, frame_no, force, fast_rank, morph_contrast = task
    m = motility.Motility()
    m.directory = directory
    m.header = header
    m.tail = tail
    m.fast_rank = fast_rank
    m.morph_contrast = morph_contrast
    try:
        m.read_frame(float(frame_no), force)
        m.save_frame()
        return (frame_no, None)
    except Exception as exc:  # pragma: no cover - surfaced to the parent
        return (frame_no, str(exc))


def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    main_dir,
    force_analysis=False,
    recalculate=False,
    make_movie=False,
    min_path_length=5,
    num_frames_ave=5,
    percent_tolerance=500,
    pixel_size=80.65,
    plot_ymax=1500,
    plot_xmax=10000,
    maxvel_color="b",
    fit_function="none",
    max_velocity=2016.25,
    min_velocity=80,
    overlap_score_cutoff=0.4,
    log_area_score_cutoff=1.0,
    diff_log_area_score_cutoff=0.5,
    legacy_linking=False,
    fast_rank=True,
    morph_contrast=False,
    nprocs=None,
    verbose=False,
):
    """Run the full FAST analysis over a directory tree of movies."""
    if nprocs is None:
        # Frame extraction is the dominant, embarrassingly-parallel stage, so
        # default to all logical cores (override with -j).
        nprocs = max(1, cpu_count())

    if main_dir is not None and len(main_dir) > 0:
        # Strip a trailing path separator (either kind, for cross-platform use).
        main_dir = main_dir.rstrip("/\\")

    if main_dir is None or not os.path.isdir(main_dir):
        sys.exit("Directory doesn't exist. Program is exiting.")

    # Resolve the dataset path up front so the analysis works regardless of
    # whether the caller passed a relative or an absolute ``-d``.  Output names
    # are derived from the dataset *basename* (not the full path), so the
    # results always land under ``<cwd>/outputs/<dataset>__pt_.../`` and never
    # collapse to an absolute path via ``os.path.join``.
    main_dir_abs = os.path.abspath(main_dir)
    anchor = os.path.dirname(main_dir_abs)
    dataset_name = os.path.basename(main_dir_abs)

    def flat(p):
        """Dataset-relative path flattened into a single filename component."""
        rel = os.path.relpath(os.path.abspath(p), anchor)
        return rel.replace(os.sep, "_")

    tolerance_prop = "none" if percent_tolerance == 500 else str(percent_tolerance)

    main_out_dir = (
        dataset_name
        + "__pt_" + str(tolerance_prop)
        + "__n_" + str(num_frames_ave)
        + "__ymax_" + str(int(plot_ymax))
        + "__p_" + str(min_path_length)
        + "__fx_" + fit_function
    )
    cwd = os.getcwd()

    # ----- normalise folder names + discover tif naming tail --------------- #
    tail_tif = ""
    for root, subFolders, files in os.walk(main_dir):
        if len(subFolders) != 0:
            continue
        tif_files = [x for x in files if x[-4:] == ".tif"]
        if len(tif_files) == 0:
            continue
        first_tif = tif_files[0]
        tail_tif = first_tif.split("_")[2]

        # Remove spaces in directory names (they break downstream paths).
        head, tail_dir = os.path.split(root)
        clean_tail = "_".join(tail_dir.split())
        if clean_tail != tail_dir:
            new_root = os.path.join(head, clean_tail)
            os.rename(root, new_root)

    # ----- output directory scaffolding ----------------------------------- #
    out_base = os.path.join("outputs", main_out_dir)
    os.makedirs(os.path.join(out_base, "combined"), exist_ok=True)

    out_MEAN_fname = os.path.abspath(os.path.join(out_base, "combined", "MEAN_values.txt"))
    out_SEM_fname = os.path.abspath(os.path.join(out_base, "combined", "SEM_values.txt"))

    data_header = ("%6s\t%4s\t%80s" + "\t%20s" * 13 + "\n") % (
        "slide", "exp", "filename", "protein", "points-filtered", "conc(mg/ml)",
        "utrophin(nM)", "top-vel-5", "p-stuck", "MVEL", "MVEL-filtered", "plateau",
        "MVIS", "mean-length-all", "mean-length-filtered", "mean-length-mobile",
    )

    if not os.path.isfile(out_MEAN_fname) or force_analysis or recalculate:
        m_stats = open(out_MEAN_fname, "w")
        m_stats.write(data_header)
        s_stats = open(out_SEM_fname, "w")
        s_stats.write(data_header)
    else:
        m_stats = open(out_MEAN_fname, "a")
        s_stats = open(out_SEM_fname, "a")

    # ----- discover folders to process ------------------------------------ #
    process_folders = {}
    for root, subFolders, files in os.walk(main_dir):
        if len(subFolders) == 0 and (
            len([x for x in files if x[-4:] == ".tif"]) > 0
            or len([x for x in files if x[:6] == "filXYs"]) > 0
        ):
            entries = root.split(os.sep)
            top_folder = os.sep.join(entries[:-1])
            exp_num = entries[-1]
            process_folders.setdefault(top_folder, []).append(exp_num)

    sorted_top_roots = sorted(process_folders.keys())
    for top_root in sorted_top_roots:
        process_folders[top_root] = sorted(process_folders[top_root])

    combined_data_counter = 0
    number_of_frames = 0

    for top_root in sorted_top_roots:
        combined_stats = []
        combined_full_len_vel = []
        combined_max_len_vel = []

        data_info = top_root.split(os.sep)
        top_folder = data_info[-1]
        root_header = flat(top_root)

        slide_num = -1
        if len(data_info) > 1:
            slide_entries = data_info[-2].split("_")
            if len(slide_entries) == 2 and slide_entries[0] == "slide":
                slide_num = int(slide_entries[1])

        fname_entries = top_folder.split("_")
        protein_name = fname_entries[0] if len(fname_entries) > 0 else "N/A"

        utrophin_conc = 0.0
        if "utr" in fname_entries:
            pos = fname_entries.index("utr")
            if pos - 1 > -1 and fname_entries[pos - 1][-2:] == "nM" and is_number(
                fname_entries[pos - 1][:-2]
            ):
                utrophin_conc = float(fname_entries[pos - 1][:-2])

        protein_conc = 0.0
        if "ml" in fname_entries:
            pos = fname_entries.index("ml")
            if pos - 1 > -1 and fname_entries[pos - 1][-2:] == "mg" and is_number(
                fname_entries[pos - 1][:-2]
            ):
                protein_conc = float(fname_entries[pos - 1][:-2])

        combined_vl_png_name = os.path.join(
            cwd, "outputs", main_out_dir, "combined", root_header + "_length_velocity.png"
        )
        if not recalculate and not force_analysis and os.path.isfile(combined_vl_png_name):
            continue

        os.chdir(top_root)

        for final_folder in process_folders[top_root]:
            root = os.path.join(top_root, final_folder)

            new_Frame = motility.Frame()
            new_Frame.directory = final_folder
            new_Frame.header = "img_000000"
            new_Frame.tail = tail_tif
            new_Frame.fast_rank = fast_rank
            new_Frame.morph_contrast = morph_contrast
            file_exists = new_Frame.read_frame(0)
            frame_width = new_Frame.width
            frame_height = new_Frame.height

            picture_quality = "good" if not file_exists else new_Frame.check_picture_quality()
            if picture_quality == "bad":
                print("Bad picture quality in %s" % (root))
                continue

            root_flat = flat(root)
            out_vl_png_fname = os.path.join(
                cwd, "outputs", main_out_dir, root_flat + "_length_velocity.png"
            )
            out_vl_txt_fname = os.path.join(
                cwd, "outputs", main_out_dir, root_flat + "_"
            )
            out_path_fname = os.path.join(
                cwd, "outputs", main_out_dir, root_flat + "_paths"
            )

            print("Processing tif files in %s" % (root))
            start_t = time.time()

            # ----- enumerate frame numbers from the tif files ------------- #
            tif_in_folder = [
                x for x in os.listdir(final_folder) if os.path.splitext(x)[1] == ".tif"
            ]
            if len(tif_in_folder) > 0:
                frame_nos = sorted(
                    int(os.path.basename(x).split("_")[1]) for x in tif_in_folder
                )
            else:
                # No tifs: fall back to existing filXYs npy files.
                fil_files = [
                    x for x in os.listdir(final_folder) if x[:6] == "filXYs"
                ]
                frame_nos = sorted(
                    int(os.path.splitext(x)[0][6:]) for x in fil_files
                )

            number_of_frames = len(frame_nos)
            if number_of_frames == 0:
                continue

            # ----- parallel per-frame filament extraction ----------------- #
            tasks = [
                (final_folder, "img_000000", tail_tif, no, force_analysis,
                 fast_rank, morph_contrast)
                for no in frame_nos
            ]
            if nprocs > 1 and len(tasks) > 1:
                with Pool(processes=min(nprocs, len(tasks))) as pool:
                    results = pool.map(_extract_frame, tasks)
            else:
                results = [_extract_frame(t) for t in tasks]

            failures = [(no, err) for no, err in results if err is not None]
            if failures:
                # Always surface failures: a silent extraction failure would
                # otherwise look like "no output" downstream.
                print("  %d/%d frames failed to extract in %s"
                      % (len(failures), len(results), root))
                # Show the first failure (and all of them in verbose mode) so
                # the underlying error is visible.
                shown = failures if verbose else failures[:1]
                for no, err in shown:
                    print("    frame %d: %s" % (no, err))
                if len(failures) == len(results):
                    print("  All frames failed; skipping %s." % root)
                    continue

            # ----- build / load frame links ------------------------------- #
            new_motility = motility.Motility()
            new_motility.dx = 1.0 * pixel_size
            new_motility.max_velocity = 1.0 * max_velocity / pixel_size
            new_motility.num_frames = number_of_frames
            new_motility.directory = final_folder
            new_motility.force_analysis = force_analysis
            new_motility.width = frame_width
            new_motility.height = frame_height
            new_motility.min_velocity = min_velocity
            new_motility.overlap_score_cutoff = overlap_score_cutoff
            new_motility.log_area_score_cutoff = log_area_score_cutoff
            new_motility.diff_log_area_score_cutoff = diff_log_area_score_cutoff
            new_motility.legacy_linking = legacy_linking
            new_motility.fast_rank = fast_rank
            new_motility.morph_contrast = morph_contrast

            if not new_motility.read_frame_links():
                new_motility.load_frame1(0)
                new_motility.read_metadata()
                for no in frame_nos[1:]:
                    if verbose:
                        print("Making the links: Frame: %d" % (no))
                    new_motility.load_frame2(no)
                    new_motility.make_frame_links()
                    new_motility.frame1 = new_motility.frame2
                new_motility.save_links()

            new_motility.process_frame_links(num_frames_ave)
            new_motility.plot_2D_path_data(num_frames_ave, extra_fname=out_path_fname)

            if make_movie:
                new_motility.reconstruct_skeleton_images()
                new_motility.make_movie(extra_fname=out_vl_txt_fname)

            if len(new_motility.full_len_vel) < 10:
                continue

            stats = new_motility.plot_length_velocity(
                extra_fname=out_vl_png_fname,
                max_vel=plot_ymax,
                max_length=plot_xmax,
                min_path_length=min_path_length,
                percent_tolerance=percent_tolerance,
                min_points=10,
                print_plot=True,
                maxvel_color=maxvel_color,
                fit_f=fit_function,
            )
            (top_5_velocity, percent_stuck, MVEL, MVEL_filtered, max_vel_u, MVIS,
             mean_len_stuck, mean_len_filtered, mean_len_mobile, mean_len_all,
             num_points_filtered) = stats

            if top_5_velocity == -1:
                continue

            combined_stats.append([
                num_points_filtered, top_5_velocity, percent_stuck, MVEL, MVEL_filtered,
                max_vel_u, MVIS, mean_len_all, mean_len_filtered, mean_len_mobile,
            ])
            new_motility.write_length_velocity(extra_fname=out_vl_txt_fname)
            combined_full_len_vel.append(new_motility.full_len_vel)
            combined_max_len_vel.append(new_motility.max_len_vel)

            print("Time spent: %.1f" % (time.time() - start_t))

        os.chdir(cwd)

        if len(combined_full_len_vel) == 0:
            continue

        print("Combining data in %s" % (root_header))
        combined_full_len_vel = np.vstack(combined_full_len_vel)
        combined_max_len_vel = np.vstack(combined_max_len_vel)

        combined_motility = motility.Motility()
        combined_motility.directory = os.path.join("outputs", main_out_dir, "combined")
        combined_motility.full_len_vel = combined_full_len_vel
        combined_motility.max_len_vel = combined_max_len_vel
        combined_motility.num_frames = number_of_frames
        combined_motility.dx = 1.0 * pixel_size
        combined_motility.max_velocity = 1.0 * max_velocity / pixel_size
        combined_motility.min_velocity = min_velocity
        combined_motility.overlap_score_cutoff = overlap_score_cutoff
        combined_motility.log_area_score_cutoff = log_area_score_cutoff
        combined_motility.diff_log_area_score_cutoff = diff_log_area_score_cutoff

        if len(combined_full_len_vel[:, 0]) < 10:
            continue

        stats = combined_motility.plot_length_velocity(
            header=root_header + "_",
            max_vel=plot_ymax,
            max_length=plot_xmax,
            min_path_length=min_path_length,
            percent_tolerance=percent_tolerance,
            min_points=10,
            print_plot=True,
            maxvel_color=maxvel_color,
            fit_f=fit_function,
        )
        if stats[0] == -1:
            continue

        combined_motility.write_length_velocity(header=root_header + "_")

        combined_stats = np.array(combined_stats)
        mvals = np.mean(combined_stats, axis=0)
        svals = np.std(combined_stats, axis=0)
        total_filtered_points = np.sum(combined_stats[:, 0])

        fmt_string = "%6d\t%4d\t%80s\t%20s\t%20d" + "\t%20.3f" * 11 + "\n"
        mean_line = fmt_string % (
            slide_num, combined_data_counter, root_header, protein_name,
            total_filtered_points, protein_conc, utrophin_conc, mvals[1], mvals[2],
            mvals[3], mvals[4], mvals[5], mvals[6], mvals[7], mvals[8], mvals[9],
        )
        std_line = fmt_string % (
            slide_num, combined_data_counter, root_header, protein_name,
            total_filtered_points, protein_conc, utrophin_conc, svals[1], svals[2],
            svals[3], svals[4], svals[5], svals[6], svals[7], svals[8], mvals[9],
        )
        m_stats.write(mean_line)
        s_stats.write(std_line)
        combined_data_counter += 1

    m_stats.close()
    s_stats.close()
