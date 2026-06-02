"""LIMA: Loaded In vitro Motility Assay (modernized Python 3 port of ``bin/lima``).

Reads the combined ``MEAN_values.txt`` / ``SEM_values.txt`` produced by the FAST
driver, fits the small-molecule stop model ``Vmax/(1 + Vmax*x/Ks)`` to the
fraction-of-time-mobile vs. load-concentration data, and writes plots plus the
fitted force parameter ``Ks`` for each protein.

The only behavioural change from the original is the Matplotlib backend: this
port imports :mod:`fastrack.plotparams`, which selects the headless ``Agg``
backend so figures render without a display server.
"""
import os
import sys

import numpy as np
from scipy import optimize

from . import plotparams  # noqa: F401  (sets rcParams + Agg backend)
import matplotlib.pyplot as py


def sms(x, params):
    Vmax, Ks = params[0], params[1]
    return Vmax / (1.0 + Vmax * x / Ks)


def sms_utr_half(params):
    Vmax, Ks = params[0], params[1]
    return Ks / Vmax, Vmax


def err_sms(params, x, y):
    return y - sms(x, params)


def read_stats_files(fname):
    """Parse a MEAN/SEM stats file into (valid_indices, rows)."""
    with open(fname, "r") as g:
        lines = g.readlines()

    stats_out = []
    valid_lines = []
    if len(lines) > 1:
        for i in range(1, len(lines)):
            line = lines[i]
            entries = line[1:].strip().split("\t")
            slide_num = int(entries[0])
            exp_num = int(entries[1])
            row_fname = entries[2].strip()
            protein = entries[3].strip()
            data = [slide_num, exp_num, row_fname, protein] + [
                float(x) for x in entries[4:]
            ]
            stats_out.append(data)
            if not line[0] == "#":
                valid_lines.append(i - 1)

    return np.array(valid_lines), stats_out


def run(
    main_dir,
    min_load_analysis=0.0,
    max_load_analysis=0.0,
    min_load_plot=0.0,
    max_load_plot=0.0,
    protein_names=None,
    print_params=True,
    init_params=None,
    colors=None,
    load_molecule="utrophin",
):
    if colors is None:
        colors = ["b", "r", "g", "c", "m", "y"]

    mobile_init_params = init_params if init_params else [100, 0.2]
    fit_function = sms
    fit_err_function = err_sms
    fit_function_name = "_stop_model"

    if main_dir is not None and len(main_dir) > 0 and main_dir[-1] == "/":
        main_dir = main_dir[:-1]
    if main_dir is None or not os.path.isdir(main_dir):
        sys.exit("Directory or file doesn't exist. Program is exiting.")

    all_mean_file = main_dir + "/combined/MEAN_values.txt"
    all_std_file = main_dir + "/combined/SEM_values.txt"
    if not os.path.isfile(all_mean_file):
        sys.exit("Combined analysis file does not exist. Program is exiting.")

    valid, std_data = read_stats_files(all_std_file)
    valid, mean_data = read_stats_files(all_mean_file)

    header = main_dir + "/combined/lima"
    os.makedirs(header, exist_ok=True)

    header_data = [mean_data[i][:9] for i in valid]
    slide_list = np.array([int(x[0]) for x in header_data])
    expnum_list = np.array([int(x[1]) for x in header_data])
    protein_list = np.array([x[3] for x in header_data])
    utr_list = np.array([float(x[6]) for x in header_data])

    mean_data = np.array([mean_data[i][7:] for i in valid])
    std_data = np.array([std_data[i][7:] for i in valid])

    proteins = sorted(set(protein_list))
    utr_set = sorted(set(utr_list))

    if max_load_analysis == 0.0:
        max_load_analysis = max(utr_set)
    if max_load_plot == 0.0:
        max_load_plot = max_load_analysis

    if protein_names is not None:
        proteins = protein_names
    else:
        protein_names = proteins

    cap_size = 25
    tail = "_with_parameters" if print_params else ""

    x_plot, y_plot = plotparams.get_figsize(1200)
    py.figure(0, figsize=(2 * x_plot, 1 * y_plot))

    lines = []
    for i in range(len(proteins)):
        valid_pro = np.nonzero(protein_list == proteins[i])[0]
        if len(valid_pro) == 0:
            continue

        mean_data_filtered = mean_data[valid_pro, :]
        std_data_filtered = std_data[valid_pro, :]
        utr_conc = utr_list[valid_pro]

        maxvelocity = mean_data_filtered[:, 0]
        std_maxvelocity = std_data_filtered[:, 0]
        mean_percent_stuck = mean_data_filtered[:, 1]
        std_percent_stuck = std_data_filtered[:, 1]
        mean_MVEL = mean_data_filtered[:, 2]
        std_MVEL = std_data_filtered[:, 2]
        mean_MVIS = mean_data_filtered[:, 5]
        std_MVIS = std_data_filtered[:, 5]
        slide_nums = slide_list[valid_pro]
        exp_nums = expnum_list[valid_pro]

        # Sort by utrophin concentration.
        order = np.argsort(utr_conc)
        utr = utr_conc[order]
        mean_frac_mobile = 1.0 - mean_percent_stuck[order] / 100.0
        std_frac_mobile = std_percent_stuck[order] / 100.0
        mean_MVEL = mean_MVEL[order]
        std_MVEL = std_MVEL[order]
        mean_MVIS = mean_MVIS[order]
        std_MVIS = std_MVIS[order]
        maxvelocity = maxvelocity[order]
        std_maxvelocity = std_maxvelocity[order]
        slide_nums = slide_nums[order]
        exp_nums = exp_nums[order]

        ref_velocity = maxvelocity[0]
        mobile_correction = mean_MVEL / ref_velocity
        mean_frac_time_mobile = mean_frac_mobile * mobile_correction
        std_frac_time_mobile = std_frac_mobile * mobile_correction

        valid_utr = np.nonzero(
            (utr <= max_load_analysis) * (utr >= min_load_analysis)
        )[0]
        utr = utr[valid_utr]
        maxvelocity = maxvelocity[valid_utr]
        mean_frac_mobile = mean_frac_mobile[valid_utr]
        std_frac_mobile = std_frac_mobile[valid_utr]
        mean_MVIS = mean_MVIS[valid_utr]
        std_MVIS = std_MVIS[valid_utr]
        mean_frac_time_mobile = mean_frac_time_mobile[valid_utr]
        std_frac_time_mobile = std_frac_time_mobile[valid_utr]

        num_points = len(utr)
        if num_points == 0:
            sys.exit("No data points for plotting!.Exiting.")

        py.subplot(121)
        py.errorbar(utr, mean_frac_time_mobile * 100, yerr=std_frac_time_mobile * 100,
                    marker="o", color=colors[i], linestyle="None", capsize=cap_size)
        py.subplot(122)
        py.errorbar(utr, mean_MVIS, yerr=std_MVIS, marker="o", color=colors[i],
                    linestyle="None", capsize=cap_size)

        np.savetxt(header + "/" + proteins[i] + "_percent_mobile.txt",
                   np.column_stack((utr, mean_frac_mobile * 100.0, std_frac_mobile * 100.0)))
        np.savetxt(header + "/" + proteins[i] + "_percent_time_mobile.txt",
                   np.column_stack((utr, mean_frac_time_mobile * 100.0, std_frac_time_mobile * 100.0)))
        np.savetxt(header + "/" + proteins[i] + "_MVIS.txt",
                   np.column_stack((utr, mean_MVIS, std_MVIS)))
        np.savetxt(header + "/" + proteins[i] + "_MVEL.txt",
                   np.column_stack((utr, mean_MVIS, std_MVIS)))

        if num_points < 3:
            continue

        py.subplot(121)
        sim_utr = np.linspace(min_load_analysis, max_load_analysis, 1000)
        best_params, success = optimize.leastsq(
            fit_err_function, mobile_init_params, args=(utr, mean_frac_time_mobile), maxfev=1000
        )
        V0, Ks = best_params[0], best_params[1]
        X_val = fit_function(Ks, best_params)

        line = py.plot(sim_utr, fit_function(sim_utr, best_params) * 100, color=colors[i], linestyle="-")
        if print_params:
            py.text(Ks, 50 - i * 10, r"$%.2f nM^{K_S}$" % (Ks), color=colors[i], fontsize=50)
        lines.append(line[0])

        residuals = fit_err_function(best_params, utr, mean_frac_time_mobile)
        py.plot([Ks, Ks], [X_val * 100, 0.0], color=colors[i], linestyle="--", linewidth=10)

        np.savetxt(header + "/" + proteins[i] + "_percent_time_mobile_fit_residuals.txt",
                   np.column_stack((utr, residuals * 100.0)))
        np.savetxt(header + "/" + proteins[i] + "_percent_time_mobile_simulated.txt",
                   np.column_stack((sim_utr, fit_function(sim_utr, best_params) * 100)))

        py.subplot(122)
        py.plot(sim_utr, fit_function(sim_utr, best_params) * ref_velocity, color=colors[i], linestyle="-")
        py.plot([Ks, Ks], [X_val * ref_velocity, 0.0], color=colors[i], linestyle="--", linewidth=10)
        np.savetxt(header + "/" + proteins[i] + "_MVIS_simulated.txt",
                   np.column_stack((sim_utr, fit_function(sim_utr, best_params) * ref_velocity)))

    py.subplot(121)
    if lines:
        py.legend(lines, proteins, loc=1)
    py.xlim([min_load_plot, max_load_plot])
    py.ylim([0, 100])
    py.ylabel("% Time mobile")
    py.xlabel("Utrophin (nM)")

    py.subplot(122)
    if lines:
        py.legend(lines, proteins, loc=1)
    py.xlim([min_load_plot, max_load_plot])
    py.ylabel("MVIS (nm/s)")
    py.xlabel("Utrophin (nM)")

    py.savefig(header + "/" + "_vs_".join(proteins) + fit_function_name + tail + ".png", dpi=100)
    py.close()

    # ----- filament length vs. load --------------------------------------- #
    py.figure(1, figsize=(x_plot, y_plot))
    lines = []
    for i in range(len(proteins)):
        valid_pro = np.nonzero(protein_list == proteins[i])[0]
        if len(valid_pro) == 0:
            continue
        mean_data_filtered = mean_data[valid_pro, :]
        std_data_filtered = std_data[valid_pro, :]
        utr_conc = utr_list[valid_pro]

        mean_length = mean_data_filtered[:, -3]
        std_length = std_data_filtered[:, -3]

        order = np.argsort(utr_conc)
        utr = utr_conc[order]
        mean_length = mean_length[order]
        std_length = std_length[order]

        valid_utr = np.nonzero(
            (utr <= max_load_analysis) * (utr >= min_load_analysis)
        )[0]
        utr = utr[valid_utr]
        mean_length = mean_length[valid_utr]
        std_length = std_length[valid_utr]

        line = py.errorbar(utr, mean_length, yerr=std_length, marker="o",
                           color=colors[i], linestyle="-", capsize=cap_size)
        lines.append(line[0])
        np.savetxt(header + "/" + proteins[i] + "_length.txt",
                   np.column_stack((utr, mean_length, std_length)))

    py.xlim([min_load_plot, max_load_analysis])
    py.xlabel(load_molecule + "(nM)")
    py.ylabel("Filament length(nm)")
    if lines:
        py.legend(lines, proteins, loc=1)
    py.savefig(header + "/" + "_vs_".join(proteins) + "_length" + tail + ".png", dpi=200)
    py.close()
