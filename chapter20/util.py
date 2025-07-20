import multiprocessing as mp
import time
import timeit
from functools import partial
from typing import Callable, Tuple

import numpy as np
import pandas as pd


def get_linear_parts(num_atoms, num_threads):
    """
    (i, j) for 1<=j<=i인 경우처럼 각 row에 대해서 계산의 불균형이 존재할때 해당 불균형을 고려하여 task를 분할하는 함수
    """
    parts = np.linspace(0, num_atoms, num_threads + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def get_nested_parts(num_atoms, num_threads, upper_triangle=False):
    """
    (i, j) for 1<=j<=i인 경우처럼 각 row에 대해서 계산의 불균형이 존재할때 해당 불균형을 고려하여 task를 분할하는 함수
    """
    parts, cur_num_threads = [0], min(num_threads, num_atoms)
    for _ in range(cur_num_threads):
        part = 1 + 4 * (
            parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1) / num_threads
        )
        part = (-1 + part**0.5) / 2
        parts.append(part)
    parts = np.round(parts).astype(int)
    if (
        upper_triangle
    ):  # first row가 가장 heavy한 경우, 첫 번째가 가장 가중치가 높도록 조절
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def multi_thread_barrier_touch(r: np.ndarray, width=0.5, num_threads=8, mode="linear"):
    if mode not in ["linear", "upper_nest", "lower_nest"]:
        raise ValueError("mode must be either 'linear' or 'upper_nest' or 'lower_nest'")
    if mode == "linear":
        parts = np.linspace(0, r.shape[0], min(num_threads, r.shape[0]) + 1)
        parts = np.ceil(parts)
    else:

        def get_nested_parts(num_atoms, num_threads, upper_triangle=False):
            """
            (i, j) for 1<=j<=i인 경우처럼 각 row에 대해서 계산의 불균형이 존재할때 해당 불균형을 고려하여 task를 분할하는 함수
            """
            parts, cur_num_threads = [0], min(num_threads, num_atoms)
            for _ in range(cur_num_threads):
                part = 1 + 4 * (
                    parts[-1] ** 2
                    + parts[-1]
                    + num_atoms * (num_atoms + 1) / cur_num_threads
                )
                part = (-1 + part**0.5) / 2
                parts.append(part)
            parts = np.round(parts).astype(int)
            if (
                upper_triangle
            ):  # first row가 가장 heavy한 경우, 첫 번째가 가장 가중치가 높도록 조절
                parts = np.cumsum(np.diff(parts)[::-1])
                parts = np.append(np.array([0]), parts)
            # print(parts)
            return parts

        parts = get_nested_parts(
            r.shape[0], num_threads, upper_triangle=mode == "upper_nest"
        )
    price = np.log(
        (1 + r).cumprod(axis=0)
    )  # cumulative product of returns to get price
    jobs = []
    for i in range(1, len(parts)):
        start = int(parts[i - 1])
        end = int(parts[i])
        jobs.append(price[start:end, :].copy())  # slice the array for each job
    pool, out = mp.Pool(processes=num_threads), []
    outputs = pool.imap_unordered(get_barrier_touch, jobs)
    for out_ in outputs:
        out.append(out_)
    pool.close()
    pool.join()
    return out


def timeit_snippet(snippet, batches=10, number=100):
    elapsed_times = [[] for _ in range(batches)]
    for i in range(batches):
        timer = timeit.Timer(snippet)
        for _ in range(number):
            elapsed_time = timer.timeit(number=1)
            elapsed_times[i].append(elapsed_time)
    return elapsed_times


def func_wrapper(func, job):
    return func(**job)


def mpPandasObj(
    func: Callable, pd_obj: Tuple, num_threads=8, linear_mols=True, **kwargs
):
    """
    Parallelize a function that operates on pandas objects using multiprocessing.
    Parameters
    ----------
    func : Callable
        The function to apply to each pandas object.
    pd_obj : Tuple
        A tuple containing the pandas objects to process. (0: Name of argument, 1: pandas object)
    num_threads : int, optional
        The number of threads to use for processing (default is 8).
    mp_batchs : int, optional
        The number of batches to use for multiprocessing (default is 1).
    linear_mols : bool, optional
        Whether to use linear molecules (default is True).
    **kwargs
        Additional keyword arguments to pass to the function.
    """

    wrapped_func = partial(func_wrapper, func)

    if linear_mols:
        parts = get_linear_parts(pd_obj[1].shape[0], num_threads)
    else:
        parts = get_nested_parts(pd_obj[1].shape[0], num_threads, upper_triangle=True)
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1] : parts[i], :].copy()}
        job.update(kwargs)
        jobs.append(job)  # slice the array for each job
    if num_threads == 1:  # single thread
        ret = []
        for job in jobs:
            out = wrapped_func(job)
            ret.append(out)
    else:
        pool, ret = mp.Pool(processes=num_threads), []
        outputs = pool.imap_unordered(wrapped_func, jobs)
        for out_ in outputs:
            ret.append(out_)
        pool.close()
        pool.join()

    if isinstance(ret[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(ret[0], pd.Series):
        df0 = pd.Series()
    else:
        return ret  # if not pandas object, return list
    for i in range(len(ret)):
        if isinstance(ret[i], pd.DataFrame) or isinstance(ret[i], pd.Series):
            df0 = pd.concat([df0, ret[i]], axis=0)
        else:
            raise ValueError("Output must be a pandas DataFrame or Series")
    df0 = df0.sort_index()
    return df0


def get_barrier_touch(price: np.ndarray, width=0.5):
    # find the index of the earliest barrier touch
    touch_time = {}
    for j in range(price.shape[1]):
        for i in range(price.shape[0]):
            if price[i, j] > width or price[i, j] < -width:
                touch_time[j] = i
                break  # stop at the first touch
    return touch_time


# if __name__ == "__main__":

#     r = np.random.normal(0, 0.01, (1000, 10000))
#     price = np.log(
#         (1 + r).cumprod(axis=0)
#     )  # cumulative product of returns to get price
#     num_threads = 8

#     timer = timeit.Timer(
#         lambda: mpPandasObj(
#             get_barrier_touch, ("price", price), num_threads=num_threads, width=0.5
#         )
#     )
#     elapsed_time = timer.timeit(number=1)
#     print(f"Elapsed time for mpPandasObj: {elapsed_time:.4f} seconds")
