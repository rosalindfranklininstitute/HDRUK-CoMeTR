import argparse
import os

import numpy as np
import torch
from beartype import beartype
from torchmetrics.regression import MeanAbsoluteError

from cometr.Metric import Metric


class AE(Metric):
    """Calculates the Absolute Error (AE) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        file1: str,
        file2: str,
        file1_key: str = "/entry/data/data",
        file2_key: str = "/entry/data/data",
        output_text: str = "mae_result.txt",
    ) -> None:
        super().__init__(file1, file2, file1_key, file2_key, output_text)

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray) -> float:
        """Calculates the absolute error of the two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The absolute error of the two numpy arrays.

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

            # Calculate the absolute error on GPU
            ae = torch.abs(file1_tensor - file2_tensor).cuda()
            result = ae

            # convert result to float
            final_result = result.cpu().detach().item()

        else:
            # Calculate the absolute error on CPU
            ae = torch.abs(file1_tensor - file2_tensor)
            result = ae
            final_result = result.detach().item()

        print(
            f"The Absolute Error between the {file1_name} and {file2_name} is:\n",
            final_result,
        )

        return round(final_result, 6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the AE of two numpy arrays"
    )
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
        "-f3", "--output_text", default="mae_result.txt", help="File to store result"
    )
    args = parser.parse_args()
    AE(args.file1, args.file2, args.file1_key, args.file2_key, args.output_text).calc()


if __name__ == "__main__":
    main()
