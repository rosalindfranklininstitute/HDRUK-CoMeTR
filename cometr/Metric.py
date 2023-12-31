import argparse
import os
from os.path import dirname, basename
import h5py
import numpy as np
from beartype import beartype


class Metric:
    """Calculates the desired Metric between two HDF5 files containing voxel data.

    This class provides a convenient way to calculate the  between voxel data
    stored in two HDF5 files. It reads the data from both files, checks for validity,
    calculates the metric, and stores the result in a specified text file.

    """

    @beartype
    def __init__(
        self,
        predicted: str,
        ground_truth: str,
        predicted_key: str = "/entry/data/data",
        ground_truth_key: str = "/entry/data/data",
        output_text: str = "output.txt",
    ) -> None:
        """Initializes the Metrics class.

        Args:
            predicted (str): Path to the first HDF5 file.

            ground_truth (str): Path to the second HDF5 file.

            predicted_key (str, optional): Key to the voxel data in the first file. Defaults to '/entry/data/data'.

            ground_truth_key (str, optional): Key to the voxel data in the second file. Defaults to '/entry/data/data'.

            output_text (str, optional): Path to the file to store the result. Defaults to 'output.txt'.

        """
        data1 = Metric.load_file(predicted, predicted_key)
        self.predicted = predicted
        self.predicted_key = predicted_key

        data2 = Metric.load_file(ground_truth, ground_truth_key)
        self.ground_truth = ground_truth
        self.ground_truth_key = ground_truth_key

        # check if both files are of the same dimension
        if data1.shape != data2.shape:
            raise ValueError("Dimensions do not match")

        Metric.is_txt(output_text)
        self.output_text = output_text

    @staticmethod
    @beartype
    def check_file_exists(inp: str) -> None:
        """Check if the file exists.

        Args:
            inp (str): The file path to check.

        Raises:
            FileNotFoundError: If the file does not exist.

        """
        if not os.path.exists(inp):
            raise FileNotFoundError(f"{inp} file cannot be found")

    @staticmethod
    @beartype
    def is_h5py_file(inp: str) -> None:
        """Check if the file is a valid HDF5 file.

        Args:
            inp (str): The file path to check.

        Raises:
            TypeError: If the file is not a valid HDF5 file.

            NameError: If the file does not have a .h5 extension.

        """
        if not h5py.is_hdf5(inp):
            raise TypeError(f"{inp} is not a HDF5 file.")

        if inp[-3:] != ".h5":
            raise NameError(f"{inp} does not have a .h5 extension")

    @staticmethod
    @beartype
    def is_txt(inp: str) -> None:
        """Check if the output file has a .txt extension.

        Args:
            inp (str): The file path to check.

        Raises:
            NameError: If the output file does not have a .txt extension.

        """
        if inp[-4:] != ".txt":
            raise NameError(f"{inp} does not have a .txt extension")

    @staticmethod
    @beartype
    def is_key_valid(filename: str, test_key: str) -> None:
        """Test if the given key exists in the input .h5 filename.

        Args:
            filename (str): Input filename of the .h5 file.

            test_key (str): Key to be tested, if it is included in the .h5 file.

        Raises:
            NameError: If the key is not a valid key in the .h5 file.

        """

        def all_keys(obj):
            """Returns a list of all the keys in the object, recursively."""
            keys = (obj.name,)
            if isinstance(obj, h5py.Group):
                for key, value in obj.items():
                    if isinstance(value, h5py.Group):
                        keys = keys + all_keys(value)
                    else:
                        keys = keys + (value.name,)
            return keys

        f = h5py.File(filename, "r")
        list_of_keys = all_keys(f)
        f.close()
        if test_key not in list_of_keys:
            raise NameError(f"{test_key} is not a valid key in the {filename} file")

    @staticmethod
    @beartype
    def load_file(filename: str, file_key: str) -> np.ndarray:
        """Load data from two HDF5 files and return the corresponding numpy arrays.

        Args:
            filename (string): The input .h5 file filename

            file_key (string): Key to the voxel data in the file

        Returns:
            file_data (np.ndarray): A numpy array with the .h5 file data.

        """
        Metric.check_file_exists(filename)
        Metric.is_h5py_file(filename)
        Metric.is_key_valid(filename, file_key)

        # Load h5 file
        file = h5py.File(filename, "r")

        # Get data location from file
        data = file[file_key][:]

        # close file
        file.close()
        return data

    @staticmethod
    @beartype
    def store_file(
        data: np.ndarray, filename: str, file_key: str, overwrite: bool = False
    ) -> None:
        """Store data to a HDF5 file.

        Args:
            data (np.ndarray): The data to be stored in the output .h5 file

            filename (string): The output .h5 file filename

            file_key (string): Key to the voxel data in the file

            overwrite (bool): If true an existing file will be overwritten

        """
        if os.path.exists(filename) and overwrite is False:
            raise FileExistsError(
                f"This {filename} file already exists and cannot be overwritten"
            )

        # Load h5 file
        file = h5py.File(filename, "w")

        # Store data location from file
        lst = file_key.split("/")
        lst = [i for i in lst if i]

        grp = file
        for i in range(0, len(lst) - 1):
            grp = grp.create_group(lst[i])

        dataset = grp.create_dataset(lst[-1], data.shape, chunks=True)
        dataset[:] = data[:]

        # close file
        file.close()
        return

    @beartype
    def metric_calc(
        self, predicted_data: np.ndarray, ground_truth_data: np.ndarray
    ) -> None:
        """Calculates the loss metrics of the two numpy arrays.

        Args:
            predicted_data (np.ndarray): The numpy array containing voxel data from the first file.

            ground_truth_data (np.ndarray): The numpy array containing voxel data from the second file.

        Returns:
            None

        """
        pass

    @beartype
    def calc(self) -> float:
        """Calculates the specified metric of the two numpy arrays and saves the result to the specified text file.

        Returns:
            float: The specified metric of the two numpy arrays.

        """
        # load voxel data arrays of both files
        file1_data = Metric.load_file(self.predicted, self.predicted_key)
        file2_data = Metric.load_file(self.ground_truth, self.ground_truth_key)

        # calculate the mean squared error
        result = self.metric_calc(file1_data, file2_data)

        # insert the result in an array
        output = np.empty(
            [
                1,
            ],
            dtype=float,
        )
        output[0] = result

        # save result in a text file
        np.savetxt(self.output_text, X=output, fmt="%f", delimiter="", newline="")

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the desired metric between  two numpy arrays"
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
        help="Key to data in the ground_truth file",
    )
    parser.add_argument(
        "-f3", "--output_text", default="output.txt", help="File to store result"
    )
    args = parser.parse_args()
    call_func = Metric(
        args.predicted,
        args.ground_truth,
        args.predicted_key,
        args.ground_truth_key,
        args.output_text,
    ).calc()


if __name__ == "__main__":
    main()
