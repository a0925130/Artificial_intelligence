import numpy as np
def membership_function(data, ranges):
    zz = None
    for j in range(len(data[0])):
        data_array = []
        for i in data[:, j]:
            if i <= ranges[0]:
                data_array.append([1, 0, 0])
            elif ranges[0] < i < ranges[1]:
                data_array.append(
                    [((0 - 1) / (ranges[1] - ranges[0])) * (i - ranges[0]) + 1, ((1 - 0) / (ranges[1] - ranges[0])) *
                     (i - ranges[0]), 0])
            elif i == ranges[1]:
                data_array.append([0, 1, 0])
            elif ranges[1] < i < ranges[2]:
                data_array.append(
                    [0, ((0 - 1) / (ranges[2] - ranges[1])) * (i - ranges[1]) + 1, ((1 - 0) / (ranges[2] - ranges[1])) *
                     (i - ranges[1])])
            else:
                data_array.append([0, 0, 1])
        if zz is None:
            zz = data_array
        else:
            zz = np.hstack((zz, data_array))
    return zz
