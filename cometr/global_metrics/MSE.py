import argparse
import time

import torch
from torchmetrics.regression import MeanSquaredError
from cometr.global_metrics.Metrics import Metrics


class MSE(Metrics):
    """Calculates the Mean Squared Error (MSE) between two HDF5 files containing voxel data.

        This class provides a convenient way to calculate the MSE between voxel data
        stored in two HDF5 files. It reads the data from both files, checks for validity,
        calculates the MSE, and stores the result in a specified text file.

        """
    def __init__(
            self,
            file1,
            file2,
            file1_key='/entry/data/data',
            file2_key='/entry/data/data',
            output_text='output.txt'
    ):
        super().__init__(file1, file2, file1_key, file2_key, output_text)

    def metric_calc(self, file1_data, file2_data):
        start = time.time()
        """Calculates the mean squared error of the two numpy arrays and saves the result to the specified text file.

        Returns:
            float: The mean squared error of the two numpy arrays.

        """
        file1_data = torch.tensor(file1_data)
        file2_data = torch.tensor(file2_data)

        mean_squared_error = MeanSquaredError()
        mse = mean_squared_error(file1_data, file2_data)
        end = time.time()
        print(f"runtime: {end-start}")
        return float(mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the MSE of two numpy arrays')
    parser.add_argument("-f1", '--file1', required=True, help='Path to the first file')
    parser.add_argument("-f2", '--file2', required=True, help='Path to the second file')
    parser.add_argument("-k1", '--file1_key', default='/entry/data/data', help='Key to data in the first file')
    parser.add_argument("-k2", '--file2_key', default='/entry/data/data', help='Key to data in the second file')
    parser.add_argument("-f3", '--output_text', default='output.txt', help='File to store result')
    args = parser.parse_args()
    call_func = MSE(args.file1, args.file2, args.file1_key, args.file2_key, args.output_text).calc()
