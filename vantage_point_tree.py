import numpy as np
# from numba import njit, float32, int32, boolean, prange, gdb_init, gdb_breakpoint
# from numba.core.types import Tuple
# from numba.typed import List
import numexpr as ne
import os
from math import ceil

os.environ['NUMBA_GDB_BINARY'] = '/usr/bin/gdb'


# @njit(Tuple((int32[::1], int32[::1], float32))(float32[:, :], int32, int32[::1], float32[:]), fastmath=True, parallel=True)
def calculate_distances_parallel(data, vp_idx, point_idc, distances):
    # Calculate the distances between each point and the vantage point

    distances[point_idc] = np.array([np.linalg.norm(data[point_idx] - data[vp_idx]) for point_idx in point_idc])

    median = np.median(distances[point_idc])
    inner = distances[point_idc] <= median
    inner_idc = point_idc[inner]
    outer_idc = point_idc[~inner]
    return inner_idc, outer_idc, median


# @njit(Tuple((int32[::1], int32[::1], float32))(float32[:, :], int32, int32[:], float32[:]), fastmath=True, parallel=False)
# @njit(fastmath=True, parallel=False, cache=False)
def calculate_distances_sequential(data, vp_idx, point_idc, distances):
    # Calculate the distances between each point and the vantage point
    # for i in range(len(point_idc)):
    #     distances[point_idc[i]] = np.linalg.norm(data[point_idc[i]] - data[vp_idx])
    distances[point_idc] = np.linalg.norm(data[point_idc] - data[vp_idx], axis=1)

# @njit('List(i8[:])(f8[:, :], i8[:, :], f8[:, :], i8, List(i8[:]), List(i8[:]), i8, i8, List(i8[:]), f8[:])', fastmath=True)
# @njit(fastmath=True, parallel=True, cache=False)
def vpTreeLevel(data, tree, medians, level_nodes, level_idc, parent_idc,
                curr_max_idx, prev_max_idx, empty_arr, distances):
    for curr_node_row_idx in range(level_nodes):
        curr_node_idx = prev_max_idx + 1 + curr_node_row_idx
        node_idc = parent_idc[curr_node_row_idx]
        node_idc_len = node_idc.shape[0]
        if lnode_idc_len > 1:
            vp_idx = node_idc[0]
            point_idc = node_idc[1:]

            calculate_distances_sequential(data, vp_idx, point_idc, distances)
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


# @njit(fastmath=True, parallel=False, cache=False)
def vpTreeLevel_sequential(data, tree, medians, level_nodes, level_idc, parent_idc,
                curr_max_idx, prev_max_idx, empty_arr, distances):
    for curr_node_row_idx in level_nodes:
        curr_node_idx = prev_max_idx + 1 + curr_node_row_idx
        node_idc = parent_idc[curr_node_row_idx]
        node_idc_len = node_idc.shape[0]
        if node_idc_len > 1:
            vp_idx = node_idc[0]
            point_idc = node_idc[1:]

            calculate_distances_sequential(data, vp_idx, point_idc, distances)
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


# @njit(fastmath=True, parallel=True, cache=False)
def thread_balancer(num_threads, data, tree, medians, level_nodes, level_idc, parent_idc, curr_max_idx, prev_max_idx, empty_arr, distances):
    nodes_per_thread = ceil(level_nodes / num_threads)
    num_threads = ceil(level_nodes / nodes_per_thread)
    level_row_idc = np.arange(level_nodes)

    for thread in range(num_threads):
        thread_start = thread * nodes_per_thread
        thread_end = min((thread + 1) * nodes_per_thread, level_nodes)
        thread_row_idc = level_row_idc[thread_start: thread_end]

        vpTreeLevel_sequential(data, tree, medians, thread_row_idc, level_idc, parent_idc,
                               curr_max_idx, prev_max_idx, empty_arr, distances)


# @njit(Tuple((int32[:, ::1], float32[:, ::1]))(float32[:, :]), fastmath=True, parallel=False)
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
    calculate_distances_sequential(data, vp_idx, point_idc, distances)
    median = np.median(distances[point_idc])
    inner = distances[point_idc] <= median
    inner_idc = point_idc[inner]
    outer_idc = point_idc[~inner]
    medians[curr_node_idx] = median
    # parent_idc = List([inner_idc, outer_idc])
    parent_idc = [inner_idc, outer_idc]

    curr_node_child_left = curr_max_idx + 1 + 2*curr_node_row_idx
    tree[0] = [vp_idx, curr_node_child_left, curr_node_child_left + 1]

    # empty_arr = List([np.empty((0,), dtype=np.int32) for _ in range(2)])
    empty_arr = [np.empty((0,), dtype=np.int32) for _ in range(2)]
    empty_idc = np.empty((0,), dtype=np.int32)

    for d in range(1, depth+1):
        level_nodes = 2**d
        curr_max_idx = prev_max_idx + level_nodes
        # level_idc = List([np.empty((0,), dtype=np.int32) for x in range(level_nodes*2)])
        level_idc = [empty_idc for x in range(level_nodes*2)]
        # return np.zeros((1, 1)).astype(np.int32), np.zeros((1, 1)).astype(np.float32)

        # for curr_node_row_idx in range(level_nodes):
        #     curr_node_idx = prev_max_idx + 1 + curr_node_row_idx
        #     node_idc = parent_idc[curr_node_row_idx]
        #     if len(node_idc) > 1:
        #         vp_idx = node_idc[0]
        #         vp = data[vp_idx]
        #         distances = np.zeros((node_idc.shape[0] - 1))
        #         for d_i in range(1, distances.shape[0]):
        #             distances[d_i] = np.linalg.norm(vp - data[node_idc[d_i]])
        #         median = np.median(distances)
        #         medians[curr_node_idx] = median
        #         inner = distances <= median
        #         inner_idc = node_idc[1:][inner]
        #         outer_idc = node_idc[1:][~inner]
        #         level_idc[curr_node_row_idx: curr_node_row_idx+2] = [inner_idc, outer_idc]
        #         curr_node_child_left = curr_max_idx + 1 + 2*curr_node_row_idx
        #         tree[curr_node_idx] = [vp_idx, curr_node_child_left, curr_node_child_left + 1]
        #     elif len(node_idc) == 1:
        #         level_idc[curr_node_row_idx: curr_node_row_idx+2] = empty_arr
        #         tree[curr_node_idx] = [node_idc[0], -1, -1]
        #         medians[curr_node_idx] = -1
        #     else:
        #         level_idc[curr_node_row_idx: curr_node_row_idx+2] = empty_arr

        theads = 1024
        thread_balancer(theads, data, tree, medians, level_nodes, level_idc, parent_idc,
                        curr_max_idx, prev_max_idx, empty_arr, distances)

        # if level_nodes <= 4096:
        #     vpTreeLevel(data, tree, medians, level_nodes, level_idc, parent_idc, curr_max_idx, prev_max_idx, empty_arr, distances)
        # else:
        #     vpTreeLevel_sequential(data, tree, medians, level_nodes, level_idc, parent_idc, curr_max_idx, prev_max_idx, empty_arr, distances)


        parent_idc = level_idc
        prev_max_idx = curr_max_idx

    return tree, medians




if __name__ == '__main__':

    # time line of code
    import time
    # for i in range(1, 6):
    #     start_time = time.time()
    #     points = np.random.rand(10**i, 2)
    #     tree = VPTree(points)
    #     print(f"--- %s seconds | 10^{i} ---" % (time.time() - start_time))

    # benchmark vptree
    for i in range(1, 8):
        start_time = time.time()
        points = np.random.rand(10**i, 2).astype(np.float32)
        tree = vpTree(points)
        print(f"--- %s seconds | 10^{i} ---" % (time.time() - start_time))
