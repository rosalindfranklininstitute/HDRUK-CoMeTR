import argparse
import os

import h5py
import numpy as np
from beartype import beartype
from sklearn.metrics import mean_squared_error


class MSE:
    """Calculates the Mean Squared Error (MSE) between two HDF5 files containing voxel data.

    Args:
        file1 (str): Path to the first HDF5 file. Required command-line argument: -f1/--file1.
        file2 (str): Path to the second HDF5 file. Required command-line argument: -f2/--file2.
        file1_key (str, optional): Key to the voxel data in the first file. Defaults to '/entry/data/data'.
        file2_key (str, optional): Key to the voxel data in the second file. Defaults to '/entry/data/data'.
        output_text (str, optional): Path to the output text file for saving the result. Defaults to 'output.txt'.

    Returns:
        str: The path to the output text file if `output_text` is provided, otherwise an empty string.

    Raises:
        FileNotFoundError: If either `file1` or `file2` is not found.
        h5py.FileError: If there is an error reading the HDF5 file.
        ValueError: If the voxel data in both files do not have matching dimensions.
        Exception: If an unexpected error occurs.

    Example:
        Usage::
            from MSE import MSE
            mse_instance = MSE('file1.h5', 'file2.h5', output_text='output.txt')
            result = mse_instance.calc_mse()
    """

    @beartype
    def __init__(
            self,
            file1: str,
            file2: str,
            file1_key: str = '/entry/data/data',
            file2_key: str = '/entry/data/data',
            output_text: str = 'output.txt'
    ):
        """Initializes the `MSE` class.

        Args:
            file1 (str): Path to the first HDF5 file.
            file2 (str): Path to the second HDF5 file.
            file1_key (str, optional): Key to the voxel data in the first file. Defaults to '/entry/data/data'.
            file2_key (str, optional): Key to the voxel data in the second file. Defaults to '/entry/data/data'.
            output_text (str, optional): Path to the file to store the result. Defaults to 'output.txt'.
        """
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

    @staticmethod
    @beartype
    def check_file_exists(inp: str) -> None:
        """Check if the file exists.

        Args:
            inp (str): The file path to check.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(inp):
            raise FileNotFoundError(f"{inp} file cannot be found")

    @staticmethod
    @beartype
    def is_h5py_file(inp: str) -> None:
        """Check if the file is a valid HDF5 file.

        Args:
            inp (str): The file path to check.

        Raises:
            TypeError: If the file is not a valid HDF5 file.
            NameError: If the file does not have a .h5 extension.
        """
        if not h5py.is_hdf5(inp):
            raise TypeError(f"{inp} is not a HDF5 file.")

        if inp[-3:] != '.h5':
            raise NameError(f"{inp} does not have a .h5 extension")

    @staticmethod
    @beartype
    def is_txt(inp: str) -> None:
        """Check if the output file has a .txt extension.

        Args:
            inp (str): The file path to check.

        Raises:
            NameError: If the output file does not have a .txt extension.
        """
        if inp[-4:] != '.txt':
            raise NameError(f"{inp} does not have a .txt extension")

    @staticmethod
    @beartype
    def is_key_valid(filename: str, test_key: str) -> None:
        """Test if the given key exists in the input .h5 filename.

        Args:
            filename (str): Input filename of the .h5 file.
            test_key (str): Key to be tested, if it is included in the .h5 file.

        Raises:
            NameError: If the key is not a valid key in the .h5 file.
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

    @beartype
    def calc_mse(self) -> float:
        """Calculates the mean squared error of the two numpy arrays and saves the result to the specified text file.

        Returns:
            float: The mean squared error of the two numpy arrays.
        """
        read_file1 = h5py.File(self.file1, 'r')
        read_file2 = h5py.File(self.file2, 'r')

        file_1_data = read_file1[self.file1_key][:]
        file_2_data = read_file2[self.file2_key][:]

        # Display shape of the file data for both files
        if file_1_data.shape != file_2_data.shape:
            raise ValueError('Dimensions do not match')

        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)
        print(f'The shape of the {file1_name} file is {file_1_data.shape}')
        print(f'The shape of the {file2_name} file is {file_2_data.shape}')

        # Convert data from 3D to 2D numpy arrays to use MSE metric
        file1_2d = np.reshape(file_1_data, (file_1_data.shape[0], -1))
        file2_2d = np.reshape(file_2_data, (file_2_data.shape[0], -1))

        # Calculate the mean squared error
        mse = mean_squared_error(file1_2d, file2_2d)
        print(f"The Mean Squared Error between the {file1_name} and {file2_name} is:\n", mse)

        # Close both files
        read_file1.close()
        read_file2.close()

        np.savetxt(self.output_text, [mse], fmt='%s', delimiter='', newline='')

        return float(mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the MSE of two numpy arrays')
    parser.add_argument("-f1", '--file1', required=True, help='Path to the first file')
    parser.add_argument("-f2", '--file2', required=True, help='Path to the second file')
    parser.add_argument("-k1", '--filekey1', default='/entry/data/data', help='Key to data in the first file')
    parser.add_argument("-k2", '--filekey2', default='/entry/data/data', help='Key to data in the second file')
    parser.add_argument("-f3", '--output_text', default='output.txt', help='File to store result')
    args = parser.parse_args()
    mse_instance = MSE(args.file1, args.file2, args.filekey1, args.filekey2, args.output_text)
    call_func = mse_instance.calc_mse()
