from numbers import Complex
import pandas as pd
import numpy as np
import os
import json

def to_shape2D(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a, ((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

def to_shape4D(a, shape):
    z_, y_, x_, c_ = shape
    z, y, x, c = a.shape
    z_pad = (z_-z)
    y_pad = (y_-y)
    x_pad = (x_-x)
    c_pad = (c_-c)
    return np.pad(a, ((z_pad//2, z_pad//2 + z_pad%2), (y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2), (c_pad//2, c_pad//2 + c_pad%2)),
                  mode = 'constant')

def createCurvesFromFile(filename):
    df = pd.read_csv(filename + '/sparameters.csv', sep=',')

    # Figure out which port pairs exist
    f = open(filename + '/description.json')
    desc = json.load(f)
    port_pairs = desc['port pairs']
    f.close()

    # Depending on the number of columns we will have different numbers of ports - trying to compare all pairs of ports with each other
    if(len(port_pairs) < 2):
        # This is a 2 port chip example, can't build a 4x4 matrix
        return None, None
    else:
        # print("FILE NAME: ", filename)

        # Figure out how many unique s4p samples we will be creating
        num_unique_s4p = len(port_pairs) * (len(port_pairs) - 1) / 2

        # Duplicate the frequency values appropriately
        X = [df.iloc[:,0].values]
        X = np.repeat(X, num_unique_s4p, axis=0)
        y = []

        cur_s4p_idx = 0
        for i in range(0, len(port_pairs) - 1):
            for j in range(i + 1, len(port_pairs)):
                # We will have 4 numbers in 2 separate pairs - a, b, c, d
                # Need to look up all combinations of these ports and add them to the y labels
                # Columns will be titled Saa, Sab, ..., Sdc, Sdd
                
                column_selection = []
                column_selection.append('S' + str(port_pairs[i][0] + 1) + str(port_pairs[i][0] + 1))
                column_selection.append('S' + str(port_pairs[i][0] + 1) + str(port_pairs[i][1] + 1))
                column_selection.append('S' + str(port_pairs[i][0] + 1) + str(port_pairs[j][0] + 1))
                column_selection.append('S' + str(port_pairs[i][0] + 1) + str(port_pairs[j][1] + 1))
                column_selection.append('S' + str(port_pairs[i][1] + 1) + str(port_pairs[i][1] + 1))
                column_selection.append('S' + str(port_pairs[i][1] + 1) + str(port_pairs[j][0] + 1))
                column_selection.append('S' + str(port_pairs[i][1] + 1) + str(port_pairs[j][1] + 1))
                column_selection.append('S' + str(port_pairs[j][0] + 1) + str(port_pairs[j][0] + 1))
                column_selection.append('S' + str(port_pairs[j][0] + 1) + str(port_pairs[j][1] + 1))
                column_selection.append('S' + str(port_pairs[j][1] + 1) + str(port_pairs[j][1] + 1))

                # print("COLUMN SELECTION: ", column_selection)
                y.append(np.array(df[column_selection]))

                cur_s4p_idx += 1

        y = np.asarray(y, dtype=complex)

        y = np.stack((y.real, y.imag), axis=3)

        # TODO: messy
        X = to_shape2D(X, (len(X), 9999))
        y = to_shape4D(y, (len(y), 9999, 10, 2))


    return X, y

# ==========================================================================================================================================

def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]

def build_dataset(firstChip):
    raw_data_directory = 'UTAFSDataNew'

    X = np.array([], dtype=np.int64).reshape(0, 9999)
    y = np.array([], dtype=np.int64).reshape(0, 9999, 10, 2)

    for filename in os.listdir(raw_data_directory):
        cur_X, cur_y = createCurvesFromFile(raw_data_directory + '/' + filename)
        if cur_X is not None and cur_y is not None:
            X = np.concatenate((X, cur_X), axis=0)
            y = np.concatenate((y, cur_y), axis=0)
            if firstChip:
                break

    return trim_zeros(X), trim_zeros(y)