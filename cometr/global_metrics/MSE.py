import os
import argparse
import h5py
import numpy as np
import logging

from sklearn.metrics import mean_squared_error


# function that reads two files and returns their mean square error
class MSE:
    """
            Reads two files, calculates their mean squared error (MSE), and optionally saves the result to a text file.

            Args: file1 (str): Path to the first file. Required command-line argument: -f1/--file1. file2 (str): Path
            to the second file. Required command-line argument: -f2/--file2. output_text (str, optional): Path to the
            output text file. Optional command-line argument: -f3/--output_text.

            Returns:
                str: The path to the output text file if `output_text` is provided, otherwise an empty string.

            Raises:
                FileNotFoundError: If either `file1` or `file2` is not found.
                h5py.FileError: If there is an error reading the HDF5 file.
                Exception: If an unexpected error occurs.

            Reads the specified files using the h5py library and accesses the voxel data within them.
            Converts the data from 3D to 2D numpy arrays to calculate the mean squared error (MSE) between the arrays.
            Prints the calculated MSE to the console.

            If `output_text` is provided, saves the MSE value to the specified text file using numpy.savetxt.

            Example usage:
                $ python script.py -f1 path/to/file1.h5 -f2 path/to/file2.h5 -f3 path/to/output.txt

            """

    def __init__(self, file1: str = '', file2: str = '', output_text: str = ''):
        self.file1 = file1
        self.file2 = file2
        self.output_text = output_text

    # check if the file exists
    def check_file_exists(self):
        if not os.path.exists(self.file1) or not os.path.exists(self.file2):
            raise FileNotFoundError("File not found")

    # check if the file is a h5py file
    def is_h5py_file(self):
        if not h5py.is_hdf5(self.file1) or not h5py.is_hdf5(self.file2):
            raise ValueError("One or both files are not in HDF5 format.")

    def verify_output_file(self):
        if not self.output_text.endswith('.txt'):
            raise ValueError("The output file must be in .txt format.")

    # calculate the mean squared error

    def calc_mse(self):

        self.check_file_exists()
        self.is_h5py_file()
        self.verify_output_file()

        read_file1 = h5py.File(self.file1, 'r')
        read_file2 = h5py.File(self.file2, 'r')

        if "/entry/data/data" not in read_file1.keys() or "/entry/data/data" not in read_file2.keys():
            raise KeyError("The /entry/data/data key is not found in the HDF5 file.")

        file_1_data = read_file1['/entry/data/data'][:]
        file_2_data = read_file2['/entry/data/data'][:]

        # display shape of the file data for both files
        if file_1_data.shape != file_2_data.shape:
            raise ValueError('Dimensions do not match')

        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)
        print(f'The shape of the {file1_name} file is {file_1_data.shape}')
        print(f'The shape of the {file2_name} file is {file_2_data.shape}')

        # convert data from 3D  to 2D numpy arrays to use MSE metric
        file1_2d = np.reshape(file_1_data, (file_1_data.shape[0], -1))
        file2_2d = np.reshape(file_2_data, (file_2_data.shape[0], -1))

        # calculate the mean square error
        mse = mean_squared_error(file1_2d, file2_2d)
        print(f"The Mean Squared Error between the {file1_name} and {file2_name} is:\n", mse)

        # close both files
        read_file1.close()
        read_file2.close()

        if self.output_text != '':
            with open(self.output_text, 'w') as output_file:
                np.savetxt(output_file, [mse], fmt='%s', delimiter='', newline='')

        return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the MSE of two numpy arrays')
    parser.add_argument("-f1", '--file1', required=True, help='Path to first file')
    parser.add_argument("-f2", '--file2', required=True, help='Path to second file')
    parser.add_argument("-f3", '--output_text', required=True)
    args = parser.parse_args()
    mse_instance = MSE(args.file1, args.file2, args.output_text)
    call_func = mse_instance.calc_mse()
