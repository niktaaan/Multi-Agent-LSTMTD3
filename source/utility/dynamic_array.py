
import numpy as np
from copy import deepcopy


class DynamicArray:

    def __init__(self, size: int, dtype):
        """

        Args:
            size (int):
                The starting size of the array.
            dtype:
                The data type to store for the dynamic array.
                
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

        return str(self.data[:self.size])

    def push(self, value):
        
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
       
        self.push(value)

    def sum(self):
        return self.data[:self.size].sum()

    def set_capacity(self, new_capacity: int):
        """

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
