import argparse
import os
from os.path import dirname, abspath

import h5py
import numpy as np
import torch
from beartype import beartype


from cometr.Metric import Metric


class AbsoluteError(Metric):
    """Calculates the Absolute Error (AE) between two HDF5 files containing voxel data."""

    @beartype
    def __init__(
        self,
        predicted: str,
        ground_truth: str,
        predicted_key: str = "/entry/data/data",
        ground_truth_key: str = "/entry/data/data",
        output_text: str = "ae_result.txt",
    ) -> None:
        super().__init__(
            predicted,
            ground_truth,
            predicted_key,
            ground_truth_key,
            output_text="output.txt",
        )

    @beartype
    def metric_calc(
        self, predicted_data: np.ndarray, ground_truth_data: np.ndarray
    ) -> None:
        """Calculates the absolute error of the two numpy arrays.

        Args:
            predicted_data (np.ndarray): The numpy array containing voxel data from the first file.

            ground_truth_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            float: The absolute error of the two numpy arrays.

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

            # Calculate the absolute error on GPU
            ae = torch.abs(file1_tensor - file2_tensor).cuda()
            result = ae

            # convert result to numpy arrays
            final_result = result.cpu().detach().numpy()
            store_result = Metric.store_file(
                final_result,
                dirname(abspath(__file__)) + "/../../localmetrics_h5data/ae.h5",
                self.predicted_key,
                overwrite=True,
            )

        else:
            # Calculate the absolute error on CPU
            ae = torch.abs(file1_tensor - file2_tensor)
            result = ae
            final_result = result.detach().numpy()
            store_result = Metric.store_file(
                final_result,
                dirname(abspath(__file__)) + "/../../localmetrics_h5data/ae.h5",
                self.predicted_key,
                overwrite=True,
            )

        print(
            f"The Absolute Error between the {file1_name} and {file2_name} is stored in the ae.h5 file"
        )

        return

    def calc(self) -> None:
        # load voxel data arrays of both files
        predicted_data = Metric.load_file(self.predicted, self.predicted_key)
        ground_truth_data = Metric.load_file(self.ground_truth, self.ground_truth_key)

        self.metric_calc(predicted_data, ground_truth_data)

        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the AE of two numpy arrays")
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
        help="Key to data in the predicted file",
    )
    parser.add_argument(
        "-k2",
        "--ground_truth_key",
        default="/entry/data/data",
        help="Key to data in the ground truth file",
    )
    parser.add_argument(
        "-f3", "--output_text", default="ae_result.txt", help="File to store result"
    )

    args = parser.parse_args()
    ae_instance = AbsoluteError(
        args.predicted,
        args.ground_truth,
        args.predicted_key,
        args.ground_truth_key,
        args.output_text,
    ).calc()


if __name__ == "__main__":
    main()
