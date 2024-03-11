import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

import numpy as np

def set_value_to_one(array, value):
    """
    Set the specified value in a 2D numpy array to 1, and all other values to 0.

    Parameters:
    array (np.array): A 2D numpy array.
    value (int/float): The value to be set to 1.

    Returns:
    np.array: A new 2D numpy array with the specified value set to 1 and all other values set to 0.
    """
    return np.where(array == value, 1, 0)

# Example usage
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
value_to_set = 5

new_array = set_value_to_one(array, value_to_set)
print(new_array)


a = np.arange(4)
c = np.ma.masked_where(array == value_to_set, array)
c = c.mask.astype(np.uint8)
print(c)
print(np.array_equal(c, new_array))