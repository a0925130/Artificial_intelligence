import numpy as np
def membership_function(data, ranges):
    data_array = []
    for i in data:
        if i <= ranges[0]:
            data_array.append([1, 0, 0])
        if ranges[0] < i < ranges[1]:
            data_array.append(
                [((0 - 1) / (ranges[1] - ranges[0])) * (i - ranges[0]) + 1, ((1 - 0) / (ranges[1] - ranges[0])) *
                 (i - ranges[0]), 0])
        if i == ranges[1]:
            data_array.append([0, 1, 0])
        if ranges[1] < i < ranges[2]:
            data_array.append(
                [0, ((0 - 1) / (ranges[2] - ranges[1])) * (i - ranges[1]) + 1, ((1 - 0) / (ranges[2] - ranges[1])) *
                 (i - ranges[1])])
        if i >= ranges[2]:
            data_array.append([0, 0, 1])
    return data_array


z = membership_function([3, 6], [3, 5, 9])
print(np.array(z).shape)
