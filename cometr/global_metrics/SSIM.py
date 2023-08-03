import argparse
import os
import torch

import h5py
import numpy as np
from beartype import beartype

from torch import tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIM:
    """Calculates the Structural Similarity Index (SSIM) between two HDF5 files containing voxel data.

    This class provides a convenient way to calculate the SSIM between voxel data
    stored in two HDF5 files. It reads the data from both files, checks for validity,
    calculates the SSIM, and stores the result in a specified text file.

    """

    @beartype
    def __init__(
        self,
        file1: str,
        file2: str,
        file1_key: str = "/entry/data/data",
        file2_key: str = "/entry/data/data",
        output_text: str = "output.txt",
    ) -> None:
        """Initializes the `SSIM` class.

        Args:
            file1 (str): Path to the first HDF5 file.

            file2 (str): Path to the second HDF5 file.

            file1_key (str, optional): Key to the voxel data in the first file. Defaults to '/entry/data/data'.

            file2_key (str, optional): Key to the voxel data in the second file. Defaults to '/entry/data/data'.

            output_text (str, optional): Path to the file to store the result. Defaults to 'output.txt'.

        """

        self.file1 = file1
        self.file1_key = file1_key

        self.file2 = file2
        self.file2_key = file2_key

        self.output_text = output_text

        # Open the HDF5 files for reading
        self.read_file1 = h5py.File(self.file1, "r")
        self.read_file2 = h5py.File(self.file2, "r")

    @beartype
    def calc(self) -> float:
        """Calculates the mean squared error of the two numpy arrays and saves the result.

        This method reads data from two specified files.The data is then converted into PyTorch tensors.
        If there exists a CUDA-enabled GPU, the computation takes place on the GPU for enhanced performance.

        Returns:
            float: The mean squared error of the two numpy arrays.

        """
        file_1_data = self.read_file1[self.file1_key][:]
        file_2_data = self.read_file2[self.file2_key][:]

        # Close both files
        self.read_file1.close()
        self.read_file2.close()

        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)
        print(f"The shape of the {file1_name} file is {file_1_data.shape}")
        print(f"The shape of the {file2_name} file is {file_2_data.shape}")

        # Convert data to tensors
        file1_tensor = torch.from_numpy(file_1_data)
        file2_tensor = torch.from_numpy(file_2_data)

        # convert data to a 5D tensor
        file1_tensor_5d = torch.unsqueeze(torch.unsqueeze(file1_tensor, 0), 0)
        file2_tensor_5d = torch.unsqueeze(torch.unsqueeze(file2_tensor, 0), 0)

        # calculate the data range
        data_range = torch.max(file1_tensor_5d) - torch.min(file1_tensor_5d)

        if torch.cuda.is_available():
            file1_tensor_5d = file1_tensor_5d.cuda()
            file2_tensor_5d = file2_tensor_5d.cuda()

            ssim = StructuralSimilarityIndexMeasure(data_range=data_range).cuda()
            result = ssim(file1_tensor_5d, file2_tensor_5d).cuda()

            # convert result to float
            final_result = result.cpu().detach().item()

        else:
            ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
            result = ssim(file1_tensor_5d, file2_tensor_5d)
            final_result = result.detach().item()

        print(
            f"The Structural Similarity Index between the {file1_name} and {file2_name} is:\n",
            final_result,
        )

        np.savetxt(self.output_text, [final_result], fmt="%s", delimiter="", newline="")

        return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the SSIM of two numpy arrays"
    )
    parser.add_argument("-f1", "--file1", required=True, help="Path to the first file")
    parser.add_argument("-f2", "--file2", required=True, help="Path to the second file")
    parser.add_argument(
        "-k1",
        "--filekey1",
        default="/entry/data/data",
        help="Key to data in the first file",
    )
    parser.add_argument(
        "-k2",
        "--filekey2",
        default="/entry/data/data",
        help="Key to data in the second file",
    )
    parser.add_argument(
        "-f3", "--output_text", default="output.txt", help="File to store result"
    )
    args = parser.parse_args()
    ssim_instance = SSIM(
        args.file1, args.file2, args.filekey1, args.filekey2, args.output_text
    )
    call_func = ssim_instance.calc()
