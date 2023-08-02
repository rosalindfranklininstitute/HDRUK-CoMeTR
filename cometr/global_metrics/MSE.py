import argparse
import os
import torch

import h5py
import numpy as np
from beartype import beartype

from torch import tensor
from torchmetrics.regression import MeanSquaredError


class MSE:
    """Calculates the Mean Squared Error (MSE) between two HDF5 files containing voxel data.

    This class provides a convenient way to calculate the MSE between voxel data
    stored in two HDF5 files. It reads the data from both files, checks for validity,
    calculates the MSE, and stores the result in a specified text file.

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
        """Initializes the `MSE` class.

        Args:
            file1 (str): Path to the first HDF5 file.

            file2 (str): Path to the second HDF5 file.

            file1_key (str, optional): Key to the voxel data in the first file. Defaults to '/entry/data/data'.

            file2_key (str, optional): Key to the voxel data in the second file. Defaults to '/entry/data/data'.

            output_text (str, optional): Path to the file to store the result. Defaults to 'output.txt'.

        """
        # super().__init__(file1, file2, file1_key, file2_key, output_text)

        self.file1 = file1
        self.file1_key = file1_key

        self.file2 = file2
        self.file2_key = file2_key

        self.output_text = output_text

        # Open the HDF5 files for reading
        self.read_file1 = h5py.File(self.file1, "r")
        self.read_file2 = h5py.File(self.file2, "r")

    @beartype()
    def metric_calc(self):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the MSE of two numpy arrays"
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
    mse_instance = MSE(
        args.file1, args.file2, args.filekey1, args.filekey2, args.output_text
    )
    call_func = mse_instance.metric_calc()
