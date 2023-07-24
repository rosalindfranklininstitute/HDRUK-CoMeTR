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

    def __init__(
            self,
            file1: str,
            file2: str ,
            file1_key: str = '/entry/data/data',
            file2_key: str = '/entry/data/data',
            output_text: str = 'output.txt'
    ):
        MSE.check_file_exists(file1)
        MSE.is_h5py_file(file1)
        MSE.is_key_valid(file1, file1_key)
        self.file1 = file1
        self.file1_key = file1_key

        MSE.check_file_exists(file2)
        MSE.is_h5py_file(file2)
        MSE.is_key_valid(file2, file2_key)
        self.file2 = file2
        self.file2_key = file2_key

        MSE.is_txt(output_text)
        self.output_text = output_text
    # check if the file exists
    @staticmethod
    def check_file_exists(inp):
        if not os.path.exists(inp):
            raise FileNotFoundError(f"{inp} file cannot be found")

    # check if the file is a h5py file
    @staticmethod
    def is_h5py_file(inp):
        if not h5py.is_hdf5(inp):
            raise TypeError(f"{inp} is a HDF5 file.")

        if inp[-3:] != '.h5':
            raise NameError(f"{inp} does not have a .h5 extension")

    @staticmethod
    def is_txt(inp):
        if inp[-4:] != '.txt':
            raise NameError(f"{inp} does not have a .txt extension")

    @staticmethod
    def is_key_valid(filename: str, test_key: str) -> None:
        """Test if given key exists in the input .h5 filename

            Args:
                filename (string): Input filename of .h5 file

                test_key (string): Key to be tested, if it is included in the filename .h5 file

        """
        def all_keys(obj):
            keys = (obj.name,)
            if isinstance(obj, h5py.Group):
                for key, value in obj.items():
                    if isinstance(value, h5py.Group):
                        keys = keys + all_keys(value)
                    else:
                        keys = keys + (value.name,)
            return keys

        f = h5py.File(filename, "r")
        list_of_keys = all_keys(f)
        f.close()
        if test_key not in list_of_keys:
            raise NameError(f"{test_key} is not a valid key in the {filename} file")
   # calculate the mean squared error
    def calc_mse(self):

        read_file1 = h5py.File(self.file1, 'r')
        read_file2 = h5py.File(self.file2, 'r')


        file_1_data = read_file1[self.file1_key][:]
        file_2_data = read_file2[self.file2_key][:]

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

        np.savetxt(self.output_text, [mse], fmt='%s', delimiter='', newline='')

        return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the MSE of two numpy arrays')
    parser.add_argument("-f1", '--file1', required=True, help='Path to first file')
    parser.add_argument("-f2", '--file2', required=True, help='Path to second file')
    parser.add_argument("-k1", '--filekey1', default='/entry/data/data', help='Key to data in the first file')
    parser.add_argument("-k2", '--filekey2', default='/entry/data/data', help='Key to data in the second file')
    parser.add_argument("-f3", '--output_text', required=True)
    args = parser.parse_args()
    mse_instance = MSE(args.file1, args.file2, args.filekey1, args.filekey2, args.output_text)
    call_func = mse_instance.calc_mse()
