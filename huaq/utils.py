import numpy as np
import itertools

def cartesian_product(*arrays):
    return np.array(list(itertools.product(*arrays)))
