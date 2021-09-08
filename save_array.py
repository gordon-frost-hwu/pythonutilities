#! /usr/bin/python
import pdb
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from copy import deepcopy
# Imports that are used for the ordering of the input files according to their index number
from collections import OrderedDict
from itertools import cycle
import re
from matplotlib import rc
import pandas as pd
from random import randint
# from utilities.optimal_control_response import *

rc('text', usetex=False)

line_cycle = cycle(["-",":","--","-.",])
marker_cycle = cycle(["p","v","o","D","s",])
plt.ion()
DEFAULT_PATH = '/tmp/learning_interactions.txt'
INCLUDE_THRESHOLD = False
THRESHOLD = -10000

def consolidateData(paths_to_avg_data_over):
    df = {}

    num_files = len(paths_to_avg_data_over)
    print("DEBUG: length of paths: {0}".format(num_files))
    min_num_samples = 100000
    _idx = 0
    for path in paths_to_avg_data_over:
        d = np.loadtxt(path, comments='#', delimiter=DELIMITER)
        to_insert = d[:, INDEX_TO_PLOT]
        print("shape of column to insert: {0}".format(to_insert.shape))
        df[_idx] = to_insert
        _idx += 1

    max_length = max([c.shape[0] for c in df.values()])
    result = np.zeros([max_length, len(paths_to_avg_data_over)])
    for idx, value in df.items():
        if value.shape[0] < max_length:
            pad = np.ones([max_length, 1]) * value[-1]
            pad[0:value.shape[0], :] = value[:].reshape([value.shape[0], 1])
        else:
            pad = value
        mid_idx = int(round(pad.shape[0] / 2.0))
        if not INCLUDE_THRESHOLD or sum(pad[mid_idx:]) / (pad.shape[0] - mid_idx) > THRESHOLD:
            result[:, idx] = pad.transpose()
    # delete any columns which are zero (and hence did not pass threshold test)
    # result = result.loc[:, (result != 0).any(axis=0)]
    zero_idxs = np.argwhere(np.all(result[..., :] == 0, axis=0))
    result = np.delete(result, zero_idxs, axis=1)
    print(result)
    return result.shape[1], result

# def consolidateData(paths_to_avg_data_over):
#     data = None
#     # TODO: Fix bug occuring when only 1 results file is presented and -m option given!!!!
#     num_files = len(paths_to_avg_data_over)
#     print("DEBUG: length of paths: {0}".format(num_files))
#     min_num_samples = 100000
#     _idx = 0
#     for path in paths_to_avg_data_over:
#         d = np.loadtxt(path, comments='#', delimiter=DELIMITER)
#         if data is None:
#             data = np.zeros(d.shape)
#             data_statistics = np.zeros([d.shape[0], num_files])
#         if d.shape[0] < min_num_samples:
#             min_num_samples = d.shape[0]
#             data = np.delete(data, range(min_num_samples, data.shape[0]), 0)
#             data_statistics = np.delete(data_statistics, range(min_num_samples, data_statistics.shape[0]), 0)
#         data[0:min_num_samples] += d[0:min_num_samples]
#         data_statistics[:, _idx] = d[0:min_num_samples, INDEX_TO_PLOT]
#         _idx += 1
#     data /= len(paths_to_avg_data_over)
#     print("DATA STATS: {0}".format(data_statistics.shape))
#
#     return data_statistics

def sort_input(input_list, num_runs):
    fake_idx = 0
    results_dict = {}
    for filepath in input_list:
        try:
            split_filepath = filepath.split("/")
            res_dir_name, filename = split_filepath[-2], split_filepath[-1]
        except IndexError:
            filename = filepath
        print(filename)
        try:
            result_number = re.findall(r"[-+]?\d*\.*\d+", filename)[-1]
        except:
            print("No index found in filename----trying DIR name")
            try:
                result_number = re.findall(r"[-+]?\d*\.*\d+", res_dir_name)[-1]
            except:
                print("No Index found in DIR name---assigning FAKE one")
                result_number = str(fake_idx)
                fake_idx += 1
        if "north" in filename:
            tag = "north"
        elif "east" in filename:
            tag = "east"
        elif "yaw" in filename:
            tag = "yaw"
        else:
            tag = None
        results_dict[result_number, tag] = filepath
    return results_dict

# Depending on script arguments, either use the supplied path as argument or use default
args = sys.argv
args.pop(0)
print("Args before option parsing: {0}".format(args))

if "--raw" in args:
    # Just plot all argument paths - only the step function will be available
    RAW_PLOT = True
    args.remove("--raw")
else:
    RAW_PLOT = False

if "--log" in args:
    PLOT_LOG = True
    args.remove("--log")
else:
    PLOT_LOG = False

if "-i" in args:
    # Telling which column to plot (from the end one, -1)
    INDEX_TO_PLOT = int(args[args.index("-i") + 1])
    args.remove(args[args.index("-i") + 1])
    args.remove(args[args.index("-i")])
else:
    INDEX_TO_PLOT = -1

if "-v" in args:
    DRAW_VERT_LINE = True
    INDEX_OF_VERT_LINE = int(args[args.index("-v") + 1])
    args.remove(args[args.index("-v") + 1])
    args.remove(args[args.index("-v")])
else:
    DRAW_VERT_LINE = False
    INDEX_OF_VERT_LINE = -1

if "-d" in args:
    DELIMITER = args[args.index("-d") + 1]
    args.remove(args[args.index("-d") + 1])
    args.remove("-d")
else:
    DELIMITER = "\t"

if "--axis" in args:
    AXIS_LABELS = True
    x_label = args[args.index("--axis") + 1]
    y_label = args[args.index("--axis") + 2]
    args.remove("--axis")
    args.remove(x_label)
    args.remove(y_label)
    print("preprocessing axis label strings ...")
    x_label = x_label.replace("_", " ")
    y_label = y_label.replace("_", " ")
else:
    AXIS_LABELS = False
if "--legend" in args:
    LEGEND_GIVEN = True
    print("WARNING!!!!")
    print("WARNING: Legend argument must be last argument given...")
    print("WARNING!!!!")
    opt_indx = args.index("--legend")
    legend_names = args[opt_indx+1:]
    args.remove("--legend")
    for n in legend_names:
        args.remove(n)
else:
    LEGEND_GIVEN = False

if "--ylim" in args:
    Y_LIM = True
    opt_indx = args.index("--ylim")
    y_lim_lower = int(args[opt_indx+1])
    y_lim_upper = int(args[opt_indx+2])
    print(y_lim_lower)
    args.remove("--ylim")
    args.remove(str(y_lim_lower))
    args.remove(str(y_lim_upper))
else:
    Y_LIM = False
if "--nomarker" in args:
    NOMARKER = True
    args.remove("--nomarker")
else:
    NOMARKER = False

if "-o" in args:
    SAVE_TO_FILENAME = args[args.index("-o") + 1]
    args.remove(args[args.index("-o") + 1])
    args.remove("-o")
    if "." in SAVE_TO_FILENAME:
        print("argument must not have extension")
        exit(0)
else:
    SAVE_TO_FILENAME = None

plotted_objects = []

if not RAW_PLOT:
    if "-r" in args:
        num_result_paths = int(args[args.index("-r") + 1])
        args.remove(args[args.index("-r") + 1])
        args.remove("-r")
    else:
        print("MUST supply number of results paths to script using -r flag...")
        exit(0)

    paths_ = args[0:num_result_paths]
    for _p in paths_:
        args.remove(_p)
    print("_DEBUG: paths_: {0}".format(paths_))
    print("_DEBUG: Args before option parsing: {0}".format(args))

    # -------
    # AT THIS STAGE WE HAVE: paths_ & list of operations to do in args
    # -------

    if "--sample" in args:
        SAMPLE_AT_STEP_NUMBER = int(args[args.index("--sample") + 1])
        args.remove(args[args.index("--sample") + 1])
        args.remove(args[args.index("--sample")])
    else:
        SAMPLE_AT_STEP_NUMBER = False

    if "--ordered" in args:
        ARGS_SORTED = True
        args.remove("--ordered")
    else:
        ARGS_SORTED = False

    if not ARGS_SORTED:
        sorted_paths = sort_input(paths_, 0)
    else:
        sorted_paths = {}
        idx = 0
        for path in paths_:
            sorted_paths[(str(idx), None)] = path
            idx += 1
    print("sorted_paths: {0}".format(sorted_paths))
    # Plot all of the desired paths that were given as input to the script using their indices from the dict
    idxs = [int(k[0]) for k in sorted_paths.viewkeys()] # get a list of the integer idx values given to script
    idxs = list(OrderedDict.fromkeys(idxs))
    idxs.sort()
    if len(sorted_paths.values()) < len(paths_):
        print("---------------")
        print("WARNING---integer keys for filenames clash. Plotting randomly")
        print("WARNING---constructing sorted_paths randomly")
        sorted_paths = {}
        p_idx = 0
        idxs = []
        for _path in paths_:
            sorted_paths[(str(p_idx), None)] = _path
            idxs.append(p_idx)
            p_idx += 1
        print("--------------")
    else:
        pass

    print("sorted_paths: {0}".format(sorted_paths))
    print("idxs: {0}".format(idxs))

    print("args: {0}".format(args))
    if len(args) == 0:
        args.append("plot_interactions.py")
    # plt.plot([0 for x in range(5)], "w", alpha=0.0)
    num_result_paths, structured_data = consolidateData([sorted_paths[str(_idx), None] for _idx in idxs])
    print("DEBUG: consolidatedData: {0}".format(structured_data.shape))
    print("First 10 lines of raw consolidated data:")
    print("{0}".format(structured_data[1:10, :]))

    # Calculate the mean and std deviation
    mean_structured_data = structured_data.sum(axis=1) / num_result_paths
    cont_variable = structured_data[:, :]  # should be size: [Num_samples x Num_files]
    # print("Values at episode 300: {0}".format(cont_variable[300, :]))

    cont_variable_mean = np.array([mean_structured_data.tolist(), ] * min(
        structured_data.shape)).transpose()  # should be size: [Num_samples x Num_files]
    # print("cont_variable: {0}".format(cont_variable.shape))
    # print("cont_variable_mean: {0}".format(cont_variable_mean.shape))
    # print("Mean at episode 300: {0}".format(cont_variable_mean[300]))

    variance = cont_variable - cont_variable_mean  # should be size: [Num_steps x Num_files]
    # print("SUBTRACT: {0}".format(variance.shape))
    variance = variance ** 2
    # print("SQUARE: {0}".format(variance[0:4,:]))
    variance = variance.sum(axis=1) / (num_result_paths)  # / n-1 for Bessel's correction
    # print("STD STATS: {0}".format(variance.shape))
    # print("VARIANCE: {0}".format(variance[0:4]))

    try:
        std_deviation = np.sqrt(variance)
        # print("Std Dev at episode 300: {0}".format(std_deviation[300]))
    except:
        print("Runtime error in STD Deviation calculation")
        print(variance)
        print("Negative values used for sqrt(variance)")
        exit(0)

    index = range(0, structured_data.shape[0], 1)
    data_to_write = np.zeros([structured_data.shape[0], structured_data.shape[1] + 1])
    data_to_write[:, 0] = index
    data_to_write[:, 1:] = structured_data


    structured_stats = np.zeros([structured_data.shape[0], 5])
    print(structured_data.shape)
    print(structured_stats.shape)
    structured_stats[:, 0] = index
    structured_stats[:, 1] = mean_structured_data
    structured_stats[:, 2] = std_deviation
    structured_stats[:, 3] = mean_structured_data-std_deviation
    structured_stats[:, 4] = mean_structured_data + std_deviation

    # Save the stuff to file
    if SAVE_TO_FILENAME is not None:
        np.savetxt("{0}Raw.csv".format(SAVE_TO_FILENAME), structured_data, delimiter=",")

        # Save stats with labels for easier processing with pandas
        with open("{0}Stats.csv".format(SAVE_TO_FILENAME), 'wb') as f:
            f.write(b'#Index,Mean,Std_Dev,Lower_Std,Upper_Std\n')
            np.savetxt(f, structured_stats, delimiter=",")
    else:
        print("Must specify output file using -o [filename] option")
