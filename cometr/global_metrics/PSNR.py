import argparse
import os

import numpy as np
import torch
from beartype import beartype
from torchmetrics.image import PeakSignalNoiseRatio

from cometr.global_metrics.Metric import Metric


class PSNR(Metric):
    """Calculates the Peak Signal-To-Noise Ratio (PSNR) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        predicted: str,
        target: str,
        predicted_key: str = "/entry/data/data",
        target_key: str = "/entry/data/data",
        output_text: str = "psnr_result.txt",
    ):
        super().__init__(predicted, target, predicted_key, target_key, output_text)

    @beartype
    def metric_calc(self, predicted_data: np.ndarray, target_data: np.ndarray) -> float:
        """Calculates the Peak Signal-To-Noise Ratio (PSNR)  of the two numpy arrays.

        Args:
            predicted_data (np.ndarray): The numpy array containing the predicted voxel data.

            target_data (np.ndarray): The numpy array containing the groundtruth voxel data.

        Returns:
            float: The Peak Signal-To-Noise Ratio (PSNR) of the two numpy arrays.

        """
        # Extract names of the  files
        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)

        # Convert data to tensors
        predicted_tensor = torch.from_numpy(predicted_data)
        target_tensor = torch.from_numpy(target_data)

        if torch.cuda.is_available():
            predicted_tensor = predicted_tensor.cuda()
            target_tensor = target_tensor.cuda()

            # Calculate the peak signal-to-noise ratio on GPU
            psnr = PeakSignalNoiseRatio(
                data_range=torch.max(target_tensor) - torch.min(target_tensor)
            ).cuda()
            result = psnr(predicted_tensor, target_tensor)

            # convert result to float
            final_result = result.cpu().detach().item()

        else:
            # Calculate the peak signal-to-noise ratio on CPU
            psnr = PeakSignalNoiseRatio(
                data_range=torch.max(target_tensor) - torch.min(target_tensor)
            )
            result = psnr(predicted_tensor, target_tensor)
            final_result = result.detach().item()

        print(
            f"The Peak Signal-to-Noise Ratio between the {file1_name} and {file2_name} is:\n",
            final_result,
        )

        return round(final_result, 6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the PSNR of two numpy arrays"
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
        "-f3", "--output_text", default="psnr_result.txt", help="File to store result"
    )
    args = parser.parse_args()
    PSNR(
        args.file1, args.file2, args.file1_key, args.file2_key, args.output_text
    ).calc()


if __name__ == "__main__":
    main()
