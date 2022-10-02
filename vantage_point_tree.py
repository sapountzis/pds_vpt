import numpy as np
import pandas as pd
import numexpr as ne
import os
from math import ceil, inf
from multiprocessing import cpu_count, Pool
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import bisect
import collections

os.environ['NUMBA_GDB_BINARY'] = '/usr/bin/gdb'


def calculate_distances(data, vp_idx, point_idc, distances):
    distances[point_idc] = np.linalg.norm(data[point_idc] - data[vp_idx], axis=1)


def vpTreeLevel(data, tree, medians, level_node_idc, level_idc, parent_idc,
                curr_max_idx, prev_max_idx, empty_arr, distances):
    for curr_node_row_idx in level_node_idc:
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
            level_idc[2*curr_node_row_idx] = inner_idc
            level_idc[2*curr_node_row_idx + 1] = outer_idc
            curr_node_child_left = curr_max_idx + 1 + 2 * curr_node_row_idx
            tree[curr_node_idx] = [vp_idx, curr_node_child_left, curr_node_child_left + 1]
        elif node_idc_len == 1:
            level_idc[2*curr_node_row_idx: 2*curr_node_row_idx + 2] = empty_arr
            tree[curr_node_idx] = [node_idc[0], -1, -1]
            medians[curr_node_idx] = 0
        else:
            level_idc[2*curr_node_row_idx: 2*curr_node_row_idx + 2] = empty_arr


def load_balancer(sequential_threshold, data, tree, medians, level_nodes, level_idc, parent_idc, curr_max_idx,
                  prev_max_idx,
                  empty_arr, distances):
    level_node_idc = np.arange(level_nodes)
    if level_nodes < sequential_threshold:
        vpTreeLevel(data, tree, medians, level_node_idc, level_idc, parent_idc, curr_max_idx, prev_max_idx, empty_arr,
                    distances)
    else:
        cores = cpu_count()
        pool = Pool(cores)
        batch_size = ceil(level_nodes / cores)
        for core in range(cores):
            start = core * batch_size
            end = (core + 1) * batch_size
            if end > level_nodes:
                end = level_nodes
            pool.apply_async(vpTreeLevel, args=(data, tree, medians, level_node_idc[start: end], level_idc, parent_idc,
                                                curr_max_idx, prev_max_idx, empty_arr, distances))
        pool.close()
        pool.join()


def vpTree(data: np.ndarray, sequential_threshold):
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

        load_balancer(sequential_threshold, data, tree, medians, level_nodes, level_idc, parent_idc, curr_max_idx,
                      prev_max_idx, empty_arr, distances)

        parent_idc = level_idc
        prev_max_idx = curr_max_idx

    return tree, medians


def is_leaf(node):
    return node[1] == -1 and node[2] == -1


def get_n_nearest_neighbors(data, tree, medians, query, n_neighbors):
    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError('n_neighbors must be strictly positive integer')
    tree = np.concatenate((tree, medians), axis=1)
    neighbors = _AutoSortingList(max_size=n_neighbors)
    queue = collections.deque([tree[0]])
    furthest_d = inf
    need_neighbors = True

    while queue:
        node = queue.popleft()
        if node[0] == -1:
            continue
        d = np.linalg.norm(query - data[int(node[0])])

        if d < furthest_d or need_neighbors:
            neighbors.append((d, int(node[0])))
            furthest_d = neighbors[-1][0]
            if need_neighbors:
                need_neighbors = len(neighbors) < n_neighbors

        if is_leaf(node):
            continue

        left = int(node[1]) if node[1] != -1 else None
        right = int(node[2]) if node[2] != -1 else None

        if d < node[-1]:
            if (left is not None) and d - furthest_d <= node[-1]:
                queue.append(tree[left])
            if (right is not None) and d + furthest_d >= node[-1]:
                queue.append(tree[right])
        else:
            if (right is not None) and d + furthest_d >= node[-1]:
                queue.append(tree[right])
            if (left is not None) and d - furthest_d <= node[-1]:
                queue.append(tree[left])

    return list(neighbors)


class _AutoSortingList(list):
    def __init__(self, max_size=None, *args):
        super(_AutoSortingList, self).__init__(*args)
        self.max_size = max_size

    def append(self, item):
        self.insert(bisect.bisect_left(self, item), item)
        if self.max_size is not None and len(self) > self.max_size:
            self.pop()


if __name__ == '__main__':
    import time

    thresholds = [4, 16, 64, 256, 1024, 4096]
    if 'benchmark_data.csv' not in os.listdir():
        benchmark_data = []
        # benchmark vptree
        for i in range(1, 6):
            points = np.random.rand(10 ** i, 2).astype(np.float32)
            for threshold in thresholds:
                print(f'points shape: {points.shape} | threshold: {threshold}')
                for iteration in range(10):
                    start_time = time.time()
                    tree = vpTree(points, threshold)
                    time_diff = time.time() - start_time
                    benchmark_data.append([points.shape[0], threshold, time_diff, iteration])

        # save benchmark data
        benchmark_data = pd.DataFrame(benchmark_data, columns=['points', 'threshold', 'time', 'iteration'])
        benchmark_data.to_csv('benchmark_data.csv', index=False)
    else:
        benchmark_data = pd.read_csv('benchmark_data.csv')


    benchmark_data_means = benchmark_data.groupby(['points', 'threshold'])['time'].mean().reset_index()
    benchmark_data_std = benchmark_data.groupby(['points', 'threshold'])['time'].std().reset_index()
    fig, axes = plt.subplots(2, 3)

    for i, threshold in enumerate(thresholds):
        ax = axes[i // 3, i % 3]

        benchmark_data_means_threshold = benchmark_data_means[benchmark_data_means['threshold'] == threshold]
        benchmark_data_std_threshold = benchmark_data_std[benchmark_data_std['threshold'] == threshold]

        ax.errorbar(benchmark_data_means_threshold['points'], benchmark_data_means_threshold['time'],
                    yerr=benchmark_data_std_threshold['time'])

        ax.set_title(f'threshold: {threshold}')
        ax.set_xlabel('points')
        ax.set_ylabel('time')
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig('benchmark.png', dpi=300)

    points = np.random.rand(100, 2).astype(np.float32)
    tree, medians = vpTree(points, 256)
    query = np.random.rand(2).astype(np.float32)
    k = 10
    k_nearest = get_n_nearest_neighbors(points, tree, medians, query, k)
    print(k_nearest)
