import argparse
import os
import time

import numpy as np
from beartype import beartype

import toupy

from cometr.global_metrics.Metric import Metric


class FSC(Metric):
    """Calculates the Fourier Shell Correlation between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        file1: str,
        file2: str,
        file1_key: str = "/entry/data/data",
        file2_key: str = "/entry/data/data",
        output_text: str = "fsc_result.txt",
        ring_thick: int = 1,
        apod_width: int = 20,
    ) -> None:
        super().__init__(file1, file2, file1_key, file2_key, output_text)
        self.apod_width = apod_width
        self.n_slice, self.n_row, self.n_col = Metric.load_file(file1)
        self.ring_thick = ring_thick  # ring thickness

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray):
        """Calculates the fourier shell correlation (FSC) bewteen two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The FSC result between the two numpy arrays.

        """

        # reduce 3D dimension to 2D by slicing on a specific z-index
        z_index = 10
        file1_2Ddata = file1_data[:, :, z_index]
        file2_2Ddata = file2_data[:, :, z_index]

        fsc = toupy.resolution.FourierShellCorr(file1_2Ddata, file2_2Ddata)
        return fsc


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
    call_func = FSC(
        args.file1, args.file2, args.file1_key, args.file2_key, args.output_text
    ).calc()


if __name__ == "__main__":
    main()
