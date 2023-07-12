import os
import sys
import argparse

import h5py
import numpy as np

from sklearn.metrics import mean_squared_error

#function that reads two files and returns their mean square error
def read_file(file1:str=None, file2:str=None, output_text:str=None):
    if file1 is None or file2 is  None:
        raise ValueError("Both file paths must be provided.")
    else:
        #read the file
        read_file1 = h5py.File(file1, 'r')
        read_file2 = h5py.File(file2, 'r')

        #access the voxels in the file
        file_1_data = read_file1['entry']['data']['data']
        file_2_data = read_file2['entry']['data']['data']

        #display shape of the file data for both files
        file1_name = os.path.basename(file1)
        file2_name = os.path.basename(file2)
        print(f'The shape of the {file1_name} file is {file_1_data.shape}')
        print(f'The shape of the {file2_name} file is {file_2_data.shape}')
na
        # convert data from 3D  to 2D to use MSE metric
        file1_2D = np.reshape(file_1_data, (file_1_data.shape[0], -1))
        file2_2D = np.reshape(file_2_data, (file_2_data.shape[0], -1))

        #calculate the mean square error
        mse = mean_squared_error(file1_2D, file2_2D)
        print(f"The Mean Squared Error between the {file1_name} and {file2_name} is:", mse)

        if output_text is not None:
            with open(output_text, 'a') as w:
                w.write(str(f'{mse} is the mean squared error of {file1_name} and {file2_name}\n'))
        return output_text

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", '--file1', required=False)
    parser.add_argument("-f2", '--file2', required=False)
    parser.add_argument("-f3", '--output_text', required=False)
    args = parser.parse_args()
    call_func = read_file(args.file1, args.file2, args.output_text)

