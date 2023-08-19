import argparse
import os
from os.path import dirname, abspath

import h5py
import numpy as np
import torch
from beartype import beartype
from torchmetrics.regression import MeanAbsoluteError

from cometr.Metric import Metric


class SquaredError(Metric):
    """Calculates the Squared Error (SE) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        file1: str,
        file2: str,
        file1_key: str = "/entry/data/data",
        file2_key: str = "/entry/data/data",
        output_text: str = "se_result.txt",
    ) -> None:
        super().__init__(file1, file2, file1_key, file2_key, output_text)

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray):
        """Calculates the mean absolute error of the two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The mean absolute error of the two numpy arrays.

        """
        # Extract names of the  files
        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)

        # Convert data to tensors
        file1_tensor = torch.from_numpy(file1_data)
        file2_tensor = torch.from_numpy(file2_data)

        if torch.cuda.is_available():
            file1_tensor = file1_tensor.cuda()
            file2_tensor = file2_tensor.cuda()

            # Calculate the squared error on GPU
            se = torch.pow(file1_tensor - file2_tensor, 2).cuda()
            result = se

            # detach result from gpu
            final_result = result.cpu().detach().numpy()

            # store result in a h5 file
            store_result = Metric.store_file(
                final_result,
                dirname(abspath(__file__)) + "/../../localmetrics_h5data/se.h5",
                self.file1_key,
                overwrite=True,
            )

        else:
            # Calculate the squared error on CPU
            se = torch.pow(file1_tensor - file2_tensor, 2)
            result = se
            final_result = result.detach().numpy()

            # store result in a h5 file
            store_result = Metric.store_file(
                final_result,
                dirname(abspath(__file__)) + "/../../localmetrics_h5data/se.h5",
                self.file1_key,
                overwrite=True,
            )

        print(
            f"The Squared Error between the {file1_name} and {file2_name} is stored in the se.h5 file",
        )

        return

    def calc(self) -> None:
        # load voxel data arrays of both files
        file1_data = Metric.load_file(self.file1, self.file1_key)
        file2_data = Metric.load_file(self.file2, self.file2_key)

        self.metric_calc(file1_data, file2_data)

        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the SE of two numpy arrays")
    parser.add_argument("-f1", "--file1", required=True, help="Path to the first file")
    parser.add_argument("-f2", "--file2", required=True, help="Path to the second file")
    parser.add_argument(
        "-k1",
        "--file1_key",
        default="/entry/data/data",
        help="Key to data in the first file",
    )
    parser.add_argument(
        "-k2",
        "--file2_key",
        default="/entry/data/data",
        help="Key to data in the second file",
    )
    parser.add_argument(
        "-f3", "--output_text", default="se_result.txt", help="File to store result"
    )
    args = parser.parse_args()
    se_instance = SquaredError(
        args.file1, args.file2, args.file1_key, args.file2_key, args.output_text
    ).calc()


if __name__ == "__main__":
    main()
