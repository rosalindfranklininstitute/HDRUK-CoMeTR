import argparse
import sys
import unittest
import numpy as np
import h5py

from src.global_metrics.MSE import calc_mse


def create_h5_files(file_1: str, file_2: str):
    file_1 = h5py.File(file_1, 'w')
    file_2 = h5py.File(file_2, 'w')

    # Create a dataset in the H5 files
    data1 = np.random.rand(200, 200)
    data2 = np.random.rand(200, 200)

    dataset_1 = file_1.create_dataset('entry/data/data', data1.shape, data=data1)
    dataset_2 = file_2.create_dataset('entry/data/data', data2.shape, data=data2)

    # Close the files
    file_1.close()
    file_2.close()


create_h5_files('file5_200', 'file6_200')


# define a test class
class MSETest(unittest.TestCase):

    def test_MSE_assert_error(self) -> None:
        with self.assertRaises(AssertionError):
            result = calc_mse('file5_200', 'file6_200')
            self.assertEqual(result, np.random.rand())

    # def test_value_error(self) -> None:
    #     with self.assertRaises(ValueError):
    #         calc_mse('file1.h5', 'file5_200')


if __name__ == '__main__':
    unittest.main()

# # Create the  H5 files
# file1 = h5py.File('file1.h5', 'w')
# file2 = h5py.File('file2.h5', 'w')
#
# # Create a dataset in the  H5 files
# data1 = np.random.rand(1000, 1000)
# data2 = np.random.rand(1000, 1000)
#
# dataset_1 = file1.create_dataset('entry/data/data', data1.shape, data=data1)
# dataset_2 = file2.create_dataset('entry/data/data', data2.shape, data=data2)
#
# # close the files
# file1.close()
# file2.close()
