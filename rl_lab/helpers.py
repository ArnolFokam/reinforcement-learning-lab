import numpy as np


def stable_softmax(x):
    """
    Computer the stable softmax of a one-dimensional vector

    Parameters
    ----------

    x: Union[List[number], np.ndarray]

    Returns
    ----------

    Softmax Distribution

    """
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator

    return softmax
