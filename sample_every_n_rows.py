#! /usr/bin/python
import pdb
import numpy as np
import sys
import os


# Depending on script arguments, either use the supplied path as argument or use default
args = sys.argv
args.pop(0)
print("Args before option parsing: {0}".format(args))

paths_ = args
AUTO = False
# row_idxs = [0, 24, 49, 99, 149, 199, 249, 299, 349, 399]
# row_idxs = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
row_idxs = [idx - 1 for idx in [1, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350]]

for path in paths_:
    if AUTO:
        with open(path) as f:
            columns = f.readline()
            columns = columns.strip('#')
            columns = columns.strip('\n')
    original_array = np.loadtxt(path, delimiter=',')
    num_rows = original_array.shape[0]
    if AUTO:
        print(num_rows)
        row_idxs = list(np.linspace(0, 450, 10))
        row_idxs = [int(np.round(i)) for i in row_idxs]
        print(original_array[0,:])
        print(original_array[1,:])
    
    print(row_idxs)
    sampled_array = original_array[row_idxs, :]
    # np.rint(sampled_array)

    filename, extension = os.path.splitext(path)

    result_path = "{0}Sampled{1}".format(filename, extension)
    print("saving to path: {0}".format(result_path))
    if AUTO:
        np.savetxt(result_path, sampled_array, delimiter=",", fmt='%.2f', header=columns)
    else:
        np.savetxt(result_path, sampled_array, delimiter=",", fmt='%.2f')
