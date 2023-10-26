import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import glob
import os

def row_wise_intersection_with_indices(arr1, arr2):
    # Remove duplicate rows to avoid duplicated intersection values
    arr1_unique, arr1_indices = np.unique(arr1, axis=0, return_index=True)
    arr2_unique, arr2_indices = np.unique(arr2, axis=0, return_index=True)


    
    # Find the intersection of the unique rows
    intersection_rows, idx_arr1, idx_arr2 = np.intersect1d(
        arr1_unique.view([('', arr1_unique.dtype)] * arr1_unique.shape[1]),
        arr2_unique.view([('', arr2_unique.dtype)] * arr2_unique.shape[1]),
        return_indices=True
    )

    # Get the respective indices in the original arrays
    intersection_indices_arr1 = arr1_indices[idx_arr1]
    intersection_indices_arr2 = arr2_indices[idx_arr2]

    # Convert the intersection back to a 2D array
    # intersection = intersection_rows.view(arr1_unique.dtype).reshape(-1, arr1_unique.shape[1])

    return None, intersection_indices_arr1, intersection_indices_arr2


def row_wise_intersection_with_indices_perf(arr1, arr2):
    # Remove duplicate rows to avoid duplicated intersection values
    # arr1_unique, arr1_indices = np.unique(arr1, axis=0, return_index=True)
    # arr2_unique, arr2_indices = np.unique(arr2, axis=0, return_index=True)

    # Make a contiguous copy of the array
    contiguous_array = np.ascontiguousarray(arr1)
    contiguous_array2 = np.ascontiguousarray(arr2)

    # Change the dtype of the contiguous array
    arr1 = contiguous_array.astype(arr1.dtype)
    arr2 = contiguous_array.astype(arr2.dtype)

    # Find the intersection of the unique rows
    intersection_rows, idx_arr1, idx_arr2 = np.intersect1d(
        arr1.view([('', arr1.dtype)] * arr1.shape[1]),
        arr2.view([('', arr2.dtype)] * arr2.shape[1]),
        return_indices=True
    )

    # Get the respective indices in the original arrays
    intersection_indices_arr1 = idx_arr1
    intersection_indices_arr2 = idx_arr2

    # Convert the intersection back to a 2D array
    # intersection = intersection_rows.view(arr1_unique.dtype).reshape(-1, arr1_unique.shape[1])

    return None, intersection_indices_arr1, intersection_indices_arr2



def commonsublists(x, y):
    x = x[1:-1, :]
    y = y[1:-1, :]
    _, ia, ib = row_wise_intersection_with_indices_perf(x, y)
    # _, ia2, ib2 = row_wise_intersection_with_indices_perf(x, y)
    # if not np.array_equal(ib,ib2):
    #     f=1
    # if not np.array_equal(ia,ia2):
    #     f=1
    cs = []
    last = -2

    # for i in tqdm(range(100), desc="Processing"):
    for i in range(len(ia) - 1):
        i1 = ia[i]
        i2 = ib[i]
        count = 1
        
        if i1 == ia[i + 1] - 1 and i2 == ib[i + 1] - 1 and last != i - 1:
            last = i
            cs.append([i1, i2, count])
        elif i1 == ia[i + 1] - 1 and i2 == ib[i + 1] - 1 and last == i - 1:
            cs[-1][2] += 1

    return cs


def commonsublists_perf(x, y):
    x = x[1:-1, :]
    y = y[1:-1, :]

    # Convert arrays to tuples of tuples to make them hashable
    x_tuples = tuple(map(tuple, x))
    y_tuples = tuple(map(tuple, y))

    # Find the intersection of the unique rows using set intersection
    intersection_tuples = set(x_tuples).intersection(y_tuples)

    # Convert the intersection back to numpy arrays
    intersection = np.array(list(intersection_tuples))
    
    # Find the indices of the intersection in the original arrays
    indices_x = np.where(np.isin(x_tuples, intersection_tuples))[0]
    indices_y = np.where(np.isin(y_tuples, intersection_tuples))[0]

    # Find the continuous common sublists
    cs = []
    i, j, count = 0, 0, 1
    while i < len(indices_x) - 1 and j < len(indices_y) - 1:
        if indices_x[i] + 1 == indices_x[i + 1] and indices_y[j] + 1 == indices_y[j + 1]:
            count += 1
            i += 1
            j += 1
        elif indices_x[i] + 1 != indices_x[i + 1]:
            i += 1
        elif indices_y[j] + 1 != indices_y[j + 1]:
            j += 1
        else:
            i += 1
            j += 1
            cs.append([indices_x[i], indices_y[j], count])
            count = 1

    return cs


def unionSet(arr1,arr2):
    un=np.unique(
    np.vstack([arr1,arr2]),
    axis=0)
    return un


def readMorpJson(dir, name, realmap):
    paths = []

    if dir.endswith("/"):
        dir = dir[:-1]

    if "PS" in name:
        filename = f"{dir}/{name}.csv"
    else:
        filename = f"{dir}/{name}"

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in tqdm(lines, desc="Processing Json..."):
        data = line.strip().split("||")
        x, y, z = [], [], []

        for value_str in data:
            value = json.loads(value_str)
            if realmap:
                x.append(value['lat'])
                y.append(value['lon'])
                z.append(0)
                # h.append(value['propertyMap']['HEIGHT'])
            else:
                x.append(value['x'])
                y.append(value['y'])
                z.append(value['z'])
                # h.append(value['propertyMap']['HEIGHT'])

        t = np.column_stack((x, y, z))  # Create a 2D numpy array with x, y, and z as columns
        paths.append(t)

    return np.array(paths, dtype=object)  # Return the final result as a numpy ndarray of objects

def readObjectives(dir, name, ext, delimiter=","):
    # If ext is not specified, set it to "csv" as default
    if not ext:
        ext = "csv"

    # Read the data using pandas
    names=['Length','Ascent','Time','Smoothness','Delay']
    filename = f"{dir}/{name}.{ext}"
    gen500 = pd.read_csv(filename, header=None,names=names,delimiter=delimiter)

    # Convert the pandas DataFrame to a numpy ndarray
    gen500 = gen500.to_numpy()

    return gen500

def readRealWorldPath(dir, name, ext, delimiter=","):
    # If ext is not specified, set it to "csv" as default
    if not ext:
        ext = "csv"
    
    # Read the data using pandas
    names=['Lat','Lon','Height']
    allp=[]
    file_pattern = name+"*."+ext  # Change this to the desired pattern
    file_count = count_files_matching_pattern(dir, file_pattern)
    # file_count=6
    for i in range(file_count):
        filename = f"{dir}/{name}{i}.{ext}"
        gen500 = pd.read_csv(filename, header=None,names=names,delimiter=delimiter)
        gen500 = gen500.to_numpy()
        allp.append(gen500)
    # Convert the pandas DataFrame to a numpy ndarray
    
    return np.array(allp, dtype=object)
    # return allp
    
    
def count_files_matching_pattern(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    return len(files)
