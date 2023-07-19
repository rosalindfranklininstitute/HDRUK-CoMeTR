import os
import argparse
import h5py
import numpy as np

from sklearn.metrics import mean_squared_error


# function that reads two files and returns their mean square error
def calc_mse(file1: str = '', file2: str = '', output_text: str = ''):
    """
    Reads two files, calculates their mean squared error (MSE), and optionally saves the result to a text file.

    Args:
        file1 (str): Path to the first file. Required command-line argument: -f1/--file1.
        file2 (str): Path to the second file. Required command-line argument: -f2/--file2.
        output_text (str, optional): Path to the output text file. Optional command-line argument: -f3/--output_text.

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
    try:
        # read the file
        read_file1 = h5py.File(file1, 'r')
        read_file2 = h5py.File(file2, 'r')

        # access the voxels in the file
        file_1_data = read_file1['entry']['data']['data']
        file_2_data = read_file2['entry']['data']['data']

        # Ensure the arrays have the same shape
        assert file_1_data.shape == file_2_data.shape

        # display shape of the file data for both files
        file1_name = os.path.basename(file1)
        file2_name = os.path.basename(file2)
        print(f'The shape of the {file1_name} file is {file_1_data.shape}')
        print(f'The shape of the {file2_name} file is {file_2_data.shape}')

        # convert data from 3D  to 2D numpy arrays to use MSE metric
        file1_2D = np.reshape(file_1_data, (file_1_data.shape[0], -1))
        file2_2D = np.reshape(file_2_data, (file_2_data.shape[0], -1))

        # calculate the mean square error
        mse = mean_squared_error(file1_2D, file2_2D)
        print(f"The Mean Squared Error between the {file1_name} and {file2_name} is:\n", mse)

        # close both files
        read_file1.close()
        read_file2.close()

        if output_text != '':
            with open(output_text, 'a') as output_file:
                np.savetxt(output_file, [mse], fmt='%s', delimiter='', newline='')

        return mse

    except FileNotFoundError as e:
        print(f'File not found {str(e)}')

    except OSError as e:
        print(f'Error reading file {str(e)}')

    except Exception as e:
        print(f'An error occurred: {str(e)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the MSE of two numpy arrays')
    parser.add_argument("-f1", '--file1', required=True, help='Path to first file')
    parser.add_argument("-f2", '--file2', required=True, help='Path to second file')
    parser.add_argument("-f3", '--output_text', required=False)
    args = parser.parse_args()
    call_func = calc_mse(args.file1, args.file2, args.output_text)
