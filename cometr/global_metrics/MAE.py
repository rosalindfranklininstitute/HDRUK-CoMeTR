import argparse
import os

import numpy as np
import torch
from beartype import beartype
from torchmetrics.regression import MeanAbsoluteError

from cometr.Metric import Metric


class MAE(Metric):
    """Calculates the Mean Absolute Error (MAE) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        predicted: str,
        ground_truth: str,
        predicted_key: str = "/entry/data/data",
        ground_truth_key: str = "/entry/data/data",
        output_text: str = "mae_result.txt",
    ) -> None:
        super().__init__(
            predicted, ground_truth, predicted_key, ground_truth_key, output_text
        )
        self.output_text = "mae_result.txt"

    @beartype
    def metric_calc(
        self, predicted_data: np.ndarray, ground_truth_data: np.ndarray
    ) -> float:
        """Calculates the mean absolute error of the two numpy arrays.

        Args:
            predicted_data (np.ndarray): The numpy array containing voxel data from the first file.

            ground_truth_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The mean absolute error of the two numpy arrays.

        """
        # Extract names of the  files
        file1_name = os.path.basename(self.predicted)
        file2_name = os.path.basename(self.ground_truth)

        # Convert data to tensors
        file1_tensor = torch.from_numpy(predicted_data)
        file2_tensor = torch.from_numpy(ground_truth_data)

        if torch.cuda.is_available():
            file1_tensor = file1_tensor.cuda()
            file2_tensor = file2_tensor.cuda()

            # Calculate the mean absolute error on GPU
            mae = MeanAbsoluteError().cuda()
            result = mae(file1_tensor, file2_tensor)

            # convert result to float
            final_result = result.cpu().detach().item()

        else:
            # Calculate the mean absolute error on CPU
            mae = MeanAbsoluteError()
            result = mae(file1_tensor, file2_tensor)
            final_result = result.detach().item()

        print(
            f"The Mean Absolute Error between the {file1_name} and {file2_name} is:\n",
            final_result,
        )

        return round(final_result, 6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the MAE of two numpy arrays"
    )
    parser.add_argument(
        "-f1", "--predicted", required=True, help="Path to the first file"
    )
    parser.add_argument(
        "-f2", "--ground_truth", required=True, help="Path to the second file"
    )
    parser.add_argument(
        "-k1",
        "--predicted_key",
        default="/entry/data/data",
        help="Key to data in the first file",
    )
    parser.add_argument(
        "-k2",
        "--ground_truth_key",
        default="/entry/data/data",
        help="Key to data in the second file",
    )
    parser.add_argument(
        "-f3", "--output_text", default="mae_result.txt", help="File to store result"
    )
    args = parser.parse_args()
    MAE(
        args.predicted,
        args.ground_truth,
        args.predicted_key,
        args.ground_truth_key,
        args.output_text,
    ).calc()


if __name__ == "__main__":
    main()
