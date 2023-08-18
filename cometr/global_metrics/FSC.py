import argparse
import os
import time

import numpy as np
from beartype import beartype

# to perform 3d fourier transformation
from scipy.fft import fftshift, # fastfftn

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

    def transverse_apodization(self):
        """
        Compute a tapered Hanning-like window of the size of the data
        for the apodization
        """

        transv_apod = self.apod_width
        n_slice = fftshift(np.arange(self.n_slice))
        n_row = fftshift(np.arange(self.n_row))
        n_col = fftshift(np.arange(self.n_col))
        window1D1 = (
                    1.0
                    + np.cos(2 * np.pi * (n_slice - np.floor((self.n_slice - 2 * transv_apod - 1) / 2))
                    / (1 + 2 * transv_apod))
                    ) / 2.0
        window1D2 = (
                    1.0
                    + np.cos(2 * np.pi * (n_row - np.floor((self.n_row - 2 * transv_apod - 1) / 2))
                    / (1 + 2 * transv_apod))
                    ) / 2.0
        window1D3 = (
                    1.0
                    + np.cos(2 * np.pi * (n_col - np.floor((self.n_col - 2 * transv_apod - 1) / 2))
                    / (1 + 2 * transv_apod))
                    ) / 2.0
        window1D1[transv_apod: -transv_apod] = 1
        window1D2[transv_apod: -transv_apod] = 1
        window1D3[transv_apod: -transv_apod] = 1
        window = [np.outer(window1D1, window1D2), np.outer(window1D1, window1D3)]
        return window

    @beartype
    def metric_calc(self, file1_data: np.ndarray, file2_data: np.ndarray) -> float:
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

        axial_apod = self.apod_width
        Y, X = np.indices((self.n_row, self.n_col))
        Y -= np.round(self.n_row / 2).astype(int)
        X -= np.round(self.n_col / 2).astype(int)
        r = np.sqrt(X ** 2 + Y ** 2)
        rmax = np.round(np.max(r.shape) / 2.0)
        maskout = r < rmax
        circular_region = (maskout * (1 - np.cos(np.pi * (r - rmax - 2 * axial_apod) / axial_apod))/ 2.0)
        circular_region[np.where(r < (rmax - axial_apod))] = 1

        if self.apod_width == 0:
            self.window = 1
        else:
            print("Apodization in 3D. This takes time and memory...")
            p0 = time.time()
            # TODO: find a more efficient way to do this. It know this is not optimum
            window3D = self.transverse_apodization()
            circle3D = np.asarray([circular_region for i in range(self.n_slice)])
            self.window = (np.array([np.squeeze(circle3D[:, :, i]) * window3D[0] for i in range(self.n_col)]).swapaxes(0, 1).swapaxes(1, 2))
            self.window = np.array([np.squeeze(self.window[:, i, :]) * window3D[1] for i in range(self.n_row)]).swapaxes(0, 1)
            print("Done. Time elapsed: {:.02f}s".format(time.time() - p0))
            # sagital slices
        slicenum = np.round(self.n_row / 2).astype("int")
        img1_apod = (self.window * file1_data)[:, slicenum, :]
        img2_apod = (self.window * file2_data)[:, slicenum, :]



        # FSC computation
        print("Calling method fouriercorr from the class FourierShellCorr")
        p1 = time.time()
        F1 = fastfftn(file1_data * self.window)  # FFT of the first image
        F2 = fastfftn(file2_data * self.window)  # FFT of the second image
        index = self.ringthickness()  # index for the ring thickness
        f, fnyquist = self.nyquist()  # Frequency and Nyquist Frequency
        # initializing variables
        print("Initializing...")
        C = np.empty_like(f).astype(np.float)
        C1 = np.empty_like(f).astype(np.float)
        C2 = np.empty_like(f).astype(np.float)
        npts = np.zeros_like(f)
        print("Calculating the correlation...")
        for ii in f:
            strbar = "Normalized frequency: {:.2f}".format((ii + 1) / fnyquist)
            if self.ring_thick == 0 or self.ring_thick == 1:
                auxF1 = F1[np.where(index == ii)]
                auxF2 = F2[np.where(index == ii)]
            else:
                auxF1 = F1[
                    (
                        np.where(
                            (index >= (ii - self.ring_thick / 2))
                            & (index <= (ii + self.ring_thick / 2))
                        )
                    )
                ]
                auxF2 = F2[
                    (
                        np.where(
                            (index >= (ii - self.ring_thick / 2))
                            & (index <= (ii + self.ring_thick / 2))
                        )
                    )
                ]
            C[ii] = np.abs((auxF1 * np.conj(auxF2)).sum())
            C1[ii] = np.abs((auxF1 * np.conj(auxF1)).sum())
            C2[ii] = np.abs((auxF2 * np.conj(auxF2)).sum())
            npts[ii] = auxF1.shape[0]
            progbar(ii + 1, len(f), strbar)
        print("\r")

        # The correlation
        FSC = C / (np.sqrt(C1 * C2))

        # Threshold computation
        Tnum = (
                self.snrt
                + (2 * np.sqrt(self.snrt) / np.sqrt(npts + np.spacing(1)))
                + 1 / np.sqrt(npts)
        )
        Tden = self.snrt + (2 * np.sqrt(self.snrt) / np.sqrt(npts + np.spacing(1))) + 1
        T = Tnum / Tden

        print("Done. Time elapsed: {:.02f}s".format(time.time() - p1))

        return FSC, T

        return


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
