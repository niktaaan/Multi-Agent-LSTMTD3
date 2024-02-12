"""
Python does not have an efficient built-in dynamic array class like other languages.
For example, C++ has the <vector>.

I code a quick dynamic array class (1 dimensional) that will serve my purposes using numpy here.
Not sure if I can use this in my code anymore, but it seems to work well.

Author:
    Jordan Cramer

Date:
    2023-09-06
"""
import numpy as np
from copy import deepcopy


class DynamicArray:
    """
    A 1D dynamic array class.
    The array will grow larger when needed (double capacity when the capacity limit is reached).
    """
    def __init__(self, size: int, dtype):
        """

        Args:
            size (int):
                The starting size of the array.
            dtype:
                The data type to store for the dynamic array.
                These should be the same as the types that are used by numpy.
                Example: int, float, bool
        """
        # the size should be a valid integer
        if not (size >= 0):
            raise ValueError('DynamicArray: The size should be >=0.')

        # capacity is the maximum number of elements that can be stored
        self.capacity = size

        # the data type of elements being stored in the array
        self.dtype = dtype

        # size is the current number of stored elements
        self.size = size

        # the actual data being stored in the dynamic array
        self.data = np.zeros(size, dtype=dtype)

    def __getitem__(self, key):
        # Check if key is an integer or a slice
        if isinstance(key, int):
            # Handle single index access
            if abs(key) >= self.size:
                raise ValueError('Indexing out of bounds.')
            return self.data[key]
        elif isinstance(key, slice):
            # Handle slicing
            start, stop, step = key.start, key.stop, key.step

            """
            # 4 special cases
            # start < -self.size
            # start > self.size
            # stop < -self.size
            # stop > self.size
            
            if start < -self.size:
                start = -self.size
            elif start > self.size:
                start = self.size

            if stop < -self.size:
                stop = -self.size
            elif stop > self.size:
                stop = self.size
            """

            # smarter than handling the 4 special cases above
            # slice first
            initial_slice = self.data[:self.size]

            # then slice the slice to get the values
            return initial_slice[start:stop:step]
        else:
            raise TypeError("Unsupported key type")

    def __setitem__(self, key, value):
        # Check if key is an integer or a slice
        if isinstance(key, int):
            # Handle single index access
            if abs(key) >= self.size:
                raise ValueError('Indexing out of bounds.')
            self.data[key] = value
        elif isinstance(key, slice):
            # Handle slicing
            start, stop, step = key.start, key.stop, key.step

            # slice first
            initial_slice = self.data[:self.size]

            # then slice the slice and set the values
            initial_slice[start:stop:step] = value

    def __str__(self):
        """
        This is what is returned when trying to print an object of this class.
        Example: print(DynamicArray(size=5, dtype=float))

        """
        return str(self.data[:self.size])

    def push(self, value):
        """
        Adds a value to the end of the array.
        If the array does not have enough memory capacity to store the new element,
        then it will create a new array with double the capacity and copy over the previous elements.
        """
        # check the size
        # if the size would be larger than the capacity, double the memory storage capacity
        if (self.size + 1) > self.capacity:
            if self.capacity == 0:
                self.set_capacity(1)
            else:
                self.set_capacity(self.capacity * 2)  # double the capacity

        # add the value to the end of the array
        self.data[self.size] = value

        # after adding the new value to the end, increment the size by 1
        self.size += 1

    def append(self, value):
        """
        Function Alias for push()
        Does the same thing as push(), but has a different function name.
        """
        self.push(value)

    def sum(self):
        return self.data[:self.size].sum()

    def set_capacity(self, new_capacity: int):
        """
        The capacity of the array will be changed.
        If the array becomes larger, the new elements will be set to zeroes or false values for booleans.
        If the array becomes smaller, then values at the end will be truncated.
        Old values are copied over to the resized array.

        Args:
            new_capacity (int):
                The new capacity for the dynamic array.
        """
        resized_array = np.zeros(new_capacity, dtype=self.dtype)

        # if the new size is smaller than the old size
        # slice the previous elements and copy the data over
        if new_capacity < self.size:
            resized_array = deepcopy(self.data[:new_capacity])
            self.capacity = new_capacity
            self.size = new_capacity
        else:
            resized_array[:self.size] = deepcopy(self.data[:self.size])
            self.capacity = new_capacity

        self.data = resized_array


if __name__ == '__main__':
    """ I perform some small tests for the dynamic array class here. """

    # creating dummy data for the tests
    arr = DynamicArray(size=5, dtype=float)
    arr2 = DynamicArray(size=5, dtype=float)
    for i in range(5):
        arr2[i] = i

    # indexing
    arr[-1] = 50
    arr[2] = 30
    arr[0] = 10
    print('arr', arr)

    # slicing
    print('arr slice all', arr[:])
    print('arr[:3]', arr[:3])  # 0, 1, 2

    # slicing and slicing assignment
    arr[3:] = arr2[0:2]
    print('arr slice assignment', arr)

    # simple push test
    arr.push(10)
    arr.push(10)

    # slicing outside bounds test
    print(arr[-100:100])
    print('arr.size', arr.size)
    print('arr.capacity', arr.capacity)

    # need for more capacity test
    print('pushing 4 more values')
    arr.push(111)
    arr.push(111)
    arr.push(111)
    arr.push(111)
    print('arr.size', arr.size)
    print('arr.capacity', arr.capacity)

    # manually resizing test
    arr.resize(5)
    print('arr', arr)
    print('arr.size', arr.size)
    print('arr.capacity', arr.capacity)
