import argparse
import torch
import os

import numpy as np
from beartype import beartype

from torchmetrics.regression import MeanSquaredError

from cometr.Metric import Metric


class MSE(Metric):
    """Calculates the Mean Squared Error (MSE) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        file1: str,
        file2: str,
        file1_key: str = "/entry/data/data",
        file2_key: str = "/entry/data/data",
        output_text: str = "mse_result.txt",
    ) -> None:
        super().__init__(file1, file2, file1_key, file2_key, output_text)
        self.output_text = "mse_result.txt"

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray) -> float:
        """Calculates the mean squared error of the two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The mean squared error of the two numpy arrays.

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

            # Calculate the mean squared error
            mse = MeanSquaredError().cuda()
            result = mse(file1_tensor, file2_tensor)

            # convert result to float
            final_result = result.cpu().detach().item()

        else:
            mse = MeanSquaredError()
            result = mse(file1_tensor, file2_tensor)
            final_result = result.detach().item()

        print(
            f"The Mean Squared Error between the {file1_name} and {file2_name} is:\n",
            final_result,
        )

        np.savetxt(self.output_text, [final_result], fmt="%s", delimiter="", newline="")

        return final_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the MSE of two numpy arrays"
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
        "-f3", "--output_text", default="mse_result.txt", help="File to store result"
    )
    args = parser.parse_args()
    call_func = MSE(
        args.file1, args.file2, args.file1_key, args.file2_key, args.output_text
    ).calc()


if __name__ == "__main__":
    main()
