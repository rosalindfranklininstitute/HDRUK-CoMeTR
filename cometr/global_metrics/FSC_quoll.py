import argparse
import os

import numpy as np
from beartype import beartype

from cometr.global_metrics.Metric import Metric

from miplib.analysis.resolution import analysis as frc_analysis
from miplib.analysis.resolution.fourier_ring_correlation import FRC
from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection
from miplib.processing import windowing
from miplib.data.containers.image import Image as miplibImage
from quoll.io import reader, tiles


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
            d_bin: int = 10,
            disable_hamming: bool = True,
            pixel_size: float = 1.0,
            z_correction: float = 0.1,
            average: bool = True,
    ) -> None:
        super().__init__(file1, file2, file1_key, file2_key, output_text)
        self.d_bin = d_bin
        self.disable_hamming = disable_hamming
        self.pixel_size = pixel_size
        self.z_correction = z_correction
        self.average = average

    @beartype
    def metric_calc(
            self,
            file1_data: np.ndarray,
            file2_data: np.ndarray,
    ):
        """Calculates the fourier shell correlation (FSC) bewteen two numpy arrays.

        Args:
            file1_data (np.ndarray): The numpy array containing voxel data from the first file.

            file2_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The FSC result between the two numpy arrays.

        """
        # Extract names of the  files
        file1_name = os.path.basename(self.file1)
        file2_name = os.path.basename(self.file2)

        # reduce 3D dimension to 2D by slicing on a specific z-index
        z_index = 10
        file1_2Ddata = file1_data[:, :, z_index]
        file2_2Ddata = file2_data[:, :, z_index]


        miplibImg1 = reader.Image(file1_2Ddata, self.pixel_size)
        miplibImg2 = reader.Image(file2_2Ddata, self.pixel_size)

        # assert isinstance(miplibImg1, miplibImage)
        # assert isinstance(miplibImg2, miplibImage)

        frc_data = FourierCorrelationDataCollection()

        # Apply Hamming windowing if not disabled
        if not self.disable_hamming:
            miplibImg1 = windowing.apply_hamming_window(miplibImg1)
            miplibImg2 = windowing.apply_hamming_window(miplibImg2)

        # Run FRC
        iterator = [self.disable_hamming, self.d_bin]
        frc_task = FRC(miplibImg1, miplibImg2, iterator)
        frc_data[0] = frc_task.execute()

        if self.average:
            frc_task = FRC(miplibImg1, miplibImg2, iterator)

            frc_data[0].correlation["correlation"] *= 0.5
            frc_data[0].correlation["correlation"] += 0.5 * frc_task.execute().correlation["correlation"]

        # Analyze results
        analyzer = frc_analysis.FourierCorrelationAnalysis(
            frc_data,
            self.pixel_size,
            self.d_bin
        )

        result = analyzer.execute(z_correction=self.z_correction)[0]

        return result.correlation['correlation']


def main():
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
    parser.add_argument("-e1", "--d_bin", default=10)
    parser.add_argument("-e2", "--disable_hamming", default=False)
    parser.add_argument("-e3", "--pixel_size", default=1.0)
    parser.add_argument("-e4", "--z_correction", default=0.1)
    parser.add_argument("-e5", "--average", default=True)
    args = parser.parse_args()
    call_func = FSC(
        args.file1, args.file2, args.file1_key, args.file2_key, args.output_text,
        args.d_bin, args.disable_hamming,args.pixel_size, args.z_correction
    ).calc()


if __name__ == "__main__":
    main()
