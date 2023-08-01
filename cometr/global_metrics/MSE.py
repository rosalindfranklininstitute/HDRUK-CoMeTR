import argparse
import time
import numpy as np
import torch
from beartype import beartype
from torchmetrics.regression import MeanSquaredError

from cometr.global_metrics.Metrics import Metrics


class MSE(Metrics):
    """Calculates the Mean Squared Error (MSE) between two HDF5 files containing voxel data.
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

    @beartype
    def metric_calc(self, file1_data: torch.Tensor, file2_data: torch.Tensor) -> float:
        """Calculates the mean squared error of the two numpy arrays and saves the result to the specified text file.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The mean squared error of the two numpy arrays.

        """
        start = time.time()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean_squared_error = MeanSquaredError()
        # mean_squared_error = mean_squared_error.to(device)
        mse = mean_squared_error(file1_data, file2_data)
        end = time.time()
        print(f"runtime:{end-start}")
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
