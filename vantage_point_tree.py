import numpy as np
# from numba import njit, float32, int32, boolean, prange, gdb_init, gdb_breakpoint
# from numba.core.types import Tuple
# from numba.typed import List
from numba import njit
import numexpr as ne
import os
from math import ceil
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed

os.environ['NUMBA_GDB_BINARY'] = '/usr/bin/gdb'


def calculate_distances(data, vp_idx, point_idc, distances):
    distances[point_idc] = np.linalg.norm(data[point_idc] - data[vp_idx], axis=1)


def vpTreeLevel(data, tree, medians, level_nodes, level_idc, parent_idc,
                curr_max_idx, prev_max_idx, empty_arr, distances):
    level_row_idc = np.arange(level_nodes)
    for curr_node_row_idx in level_row_idc:
        curr_node_idx = prev_max_idx + 1 + curr_node_row_idx
        node_idc = parent_idc[curr_node_row_idx]
        node_idc_len = node_idc.shape[0]
        if node_idc_len > 1:
            vp_idx = node_idc[0]
            point_idc = node_idc[1:]

            calculate_distances(data, vp_idx, point_idc, distances)
            median = np.median(distances[point_idc])
            inner = distances[point_idc] <= median
            inner_idc = point_idc[inner]
            outer_idc = point_idc[~inner]
            level_idc[curr_node_row_idx] = inner_idc
            level_idc[curr_node_row_idx + 1] = outer_idc
            curr_node_child_left = curr_max_idx + 1 + 2 * curr_node_row_idx
            tree[curr_node_idx] = [vp_idx, curr_node_child_left, curr_node_child_left + 1]
        elif node_idc_len == 1:
            level_idc[curr_node_row_idx: curr_node_row_idx + 2] = empty_arr
            tree[curr_node_idx] = [node_idc[0], -1, -1]
            medians[curr_node_idx] = -1
        else:
            level_idc[curr_node_row_idx: curr_node_row_idx + 2] = empty_arr


def vpTree(data: np.ndarray):
    depth = int(np.ceil(np.log2(data.shape[0] + 1) - 1))
    max_nodes = 2 ** (depth + 1) - 1
    tree = -np.ones((max_nodes, 3), dtype=np.int32)
    medians = -np.ones((max_nodes, 1), dtype=np.float32)
    curr_max_idx = 0
    prev_max_idx = 0
    curr_node_row_idx = 0
    curr_node_idx = 0

    node_idc = np.arange(data.shape[0]).astype(np.int32)
    vp_idx = node_idc[0]
    point_idc = node_idc[1:]
    distances = np.zeros((node_idc.shape[0]))
    calculate_distances(data, vp_idx, point_idc, distances)
    median = np.median(distances[point_idc])
    inner = distances[point_idc] <= median
    inner_idc = point_idc[inner]
    outer_idc = point_idc[~inner]
    medians[curr_node_idx] = median
    parent_idc = [inner_idc, outer_idc]

    curr_node_child_left = curr_max_idx + 1 + 2 * curr_node_row_idx
    tree[0] = [vp_idx, curr_node_child_left, curr_node_child_left + 1]

    empty_arr = [np.empty((0,), dtype=np.int32) for _ in range(2)]
    empty_idc = np.empty((0,), dtype=np.int32)

    for d in range(1, depth + 1):
        level_nodes = 2 ** d
        curr_max_idx = prev_max_idx + level_nodes
        level_idc = [empty_idc for x in range(level_nodes * 2)]

        vpTreeLevel(data, tree, medians, level_nodes, level_idc, parent_idc, curr_max_idx, prev_max_idx, empty_arr,
                    distances)

        parent_idc = level_idc
        prev_max_idx = curr_max_idx

    return tree, medians


if __name__ == '__main__':

    # time line of code
    import time

    # benchmark vptree
    for i in range(1, 7):
        start_time = time.time()
        points = np.random.rand(10 ** i, 2).astype(np.float32)
        tree = vpTree(points)
        print(f"--- %s seconds | 10^{i} | {sequential_threshold} ---" % (time.time() - start_time))
