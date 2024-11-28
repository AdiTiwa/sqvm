import numpy as np
import itertools

def cartesian_product(*arrays):
    return np.array(list(itertools.product(*arrays)))

def algorithm_time(func, num = 100):
    from timeit import timeit

    execution_time = timeit(func, number=num)
    
    return execution_time / num

def overlap(l1, l2):
    return len(np.union1d(l1, l2))
