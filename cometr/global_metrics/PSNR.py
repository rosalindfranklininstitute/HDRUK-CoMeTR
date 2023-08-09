import argparse
import os

import numpy as np
import torch
from beartype import beartype
from torchmetrics.image import PeakSignalNoiseRatio

from cometr.global_metrics.Metric import Metric


class PSNR(Metric):
    """Calculates the Peak Signal-To-Noise Ratio (PSNR) between two HDF5 files containing voxel data.

    """

    def __init__(
        self,
        file1,
        file2,
        file1_key="/entry/data/data",
        file2_key="/entry/data/data",
        output_text="output.txt",
    ):
        super().__init__(file1, file2, file1_key, file2_key, output_text)
        self.output_text = 'psnr_result.txt'

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray) -> float:
        """Calculates the Peak Signal-To-Noise Ratio (PSNR)  of the two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The Peak Signal-To-Noise Ratio (PSNR) of the two numpy arrays.

        """
        # Extract names of the  files
        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)

        # Convert data to tensors
        file1_tensor = torch.from_numpy(file1_data)
        file2_tensor = torch.from_numpy(file2_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file1_tensor = file1_tensor.to(device)
        file2_tensor = file2_tensor.to(device)

        # Calculate the peak signal-to-noise ratio
        mae = PeakSignalNoiseRatio().to(device)
        result = mae(file1_tensor, file2_tensor)

        # convert result to float
        final_result = result.cpu().detach().item()

        print(f"The Peak Signal-to-Noise Ratio between the {file1_name} and {file2_name} is:")

        return final_result


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
        "-f3", "--output_text", default="output.txt", help="File to store result"
    )
    args = parser.parse_args()
    PSNR(
        args.file1, args.file2, args.file1_key, args.file2_key, args.output_text
    ).calc()


if __name__ == "__main__":
    main()
