import unittest
from os.path import dirname, abspath
from cometr.local_metrics.AE import AbsoluteError as AE
from cometr.Metric import Metric


class AETest(unittest.TestCase):
    """Test cases for the Absolute Error (AE) calculation."""

    # Check error if the file does not exist
    def test_file_not_found_error(self) -> None:
        """Test if FileNotFoundError is raised when a file does not exist."""

        with self.assertRaises(FileNotFoundError):
            AE(
                dirname(abspath(__file__)) + "/../data/file3_300.h5",
                dirname(abspath(__file__)) + "/../data/file2_1000.h5",
                "/entry/data/data",
                "/entry/data/data",
                dirname(abspath(__file__)) + "/../data/output.txt",
            )

    # Check the error if one of the files is not in h5py format
    def test_h5pyfile(self) -> None:
        """Test if TypeError and NameError are raised when files are not in h5py format."""

        with self.assertRaises(TypeError):
            AE(
                dirname(abspath(__file__)) + "/../data/output.txt",
                dirname(abspath(__file__)) + "/../data/file2_1000.h5",
                "/entry/data/data",
                "/entry/data/data",
                dirname(abspath(__file__)) + "/../data/output.txt",
            )

        with self.assertRaises(NameError):
            AE(
                dirname(abspath(__file__)) + "/../data/file4_100",
                dirname(abspath(__file__)) + "/../data/file2_1000.h5",
                "/entry/data/data",
                "/entry/data/data",
                dirname(abspath(__file__)) + "/../data/output.txt",
            )

    # Check the error if the output file format is not txt

    # Check the error if the data is not in the standard dictionary format
    def test_key_error(self) -> None:
        """Test if NameError is raised when the data key is not valid in the HDF5 file."""

        with self.assertRaises(NameError):
            Metric(
                dirname(abspath(__file__)) + "/../data/file1_1000.h5",
                dirname(abspath(__file__)) + "/../data/file3_100.h5",
                "/data",
                "/data",
                dirname(abspath(__file__)) + "/../data/output.txt",
            )

    # Check error if dimensions of the data do not match
    def test_dim_error(self) -> None:
        """Test if ValueError is raised when the dimensions of the data do not match."""

        with self.assertRaises(ValueError):
            AE(
                dirname(abspath(__file__)) + "/../data/file1_200.h5",
                dirname(abspath(__file__)) + "/../data/file2_1000.h5",
                "/entry/data/data",
                "/entry/data/data",
                dirname(abspath(__file__)) + "/../data/output.txt",
            )

    # Check that the calc_mse's results are consistent

    def test_AE_result(self) -> None:
        """Test the consistency of the AE results."""

        metric = AE(
            dirname(abspath(__file__)) + "/../data/file1_200.h5",
            dirname(abspath(__file__)) + "/../data/file2_200.h5",
            "/entry/data/data",
            "/entry/data/data",
            dirname(abspath(__file__)) + "/../data/output.txt",
        )
        metric.calc()


if __name__ == "__main__":
    unittest.main()
