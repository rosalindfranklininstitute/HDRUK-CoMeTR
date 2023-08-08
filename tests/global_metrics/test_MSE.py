import numpy as np
import unittest
from os.path import dirname, abspath

from cometr.global_metrics.MSE import MSE
from cometr.global_metrics.Metric import Metric
from cometr.global_metrics.MAE import MAE
from cometr.global_metrics.PSNR import PSNR


class MSETest(unittest.TestCase):
    """Test cases for the Mean Squared Error (MSE) calculation."""

    # Check error if the file does not exist
    def test_file_not_found_error(self) -> None:
        """Test if FileNotFoundError is raised when a file does not exist."""

        with self.assertRaises(FileNotFoundError):
            Metric(
                dirname(abspath(__file__)) + '/../data/file3_300.h5',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # Check the error if one of the files is not in h5py format
    def test_h5pyfile(self) -> None:
        """Test if TypeError and NameError are raised when files are not in h5py format."""

        with self.assertRaises(TypeError):
            Metric(
                dirname(abspath(__file__)) + '/../data/output.txt',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

        with self.assertRaises(NameError):
            Metric(
                dirname(abspath(__file__)) + '/../data/file4_100',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # Check the error if the output file format is not txt
    def test_outputfile(self) -> None:
        """Test if NameError is raised when the output file format is not '.txt'."""

        with self.assertRaises(NameError):
            Metric(
                dirname(abspath(__file__)) + '/../data/file1_1000.h5',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output'
            )

    # Check the error if the data is not in the standard dictionary format
    def test_key_error(self) -> None:
        """Test if NameError is raised when the data key is not valid in the HDF5 file."""

        with self.assertRaises(NameError):
            Metric(
                dirname(abspath(__file__)) + '/../data/file1_1000.h5',
                dirname(abspath(__file__)) + '/../data/file3_100.h5',
                '/data',
                '/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # Check error if dimensions of the data do not match
    def test_dim_error(self) -> None:
        """Test if ValueError is raised when the dimensions of the data do not match."""

        with self.assertRaises(ValueError):
            Metric(
                dirname(abspath(__file__)) + '/../data/file1_200.h5',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # Check that the calc_mse's results are consistent
    def test_MSE_result(self) -> None:
        """Test the consistency of the calc_mse() results."""
        metric = MSE(
            dirname(abspath(__file__)) + '/../data/file1_200.h5',
            dirname(abspath(__file__)) + '/../data/file2_200.h5',
            '/entry/data/data',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/output.txt'
        )
        data1, data2 = metric.load_files()
        self.assertNotAlmostEqual(metric.metric_calc(data1, data2), np.random.normal(), places=6)

    def test_MAE_result(self) -> None:
        """Test the consistency of the calc_mse() results."""
        metric = MAE(
            dirname(abspath(__file__)) + '/../data/file1_200.h5',
            dirname(abspath(__file__)) + '/../data/file2_200.h5',
            '/entry/data/data',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/output.txt'
        )
        data1, data2 = metric.load_files()
        self.assertNotAlmostEqual(metric.metric_calc(data1, data2), np.random.normal(), places=6)

    def test_PSNR_result(self) -> None:
        """Test the consistency of the calc_mse() results."""
        metric = PSNR(
            dirname(abspath(__file__)) + '/../data/file1_200.h5',
            dirname(abspath(__file__)) + '/../data/file2_200.h5',
            '/entry/data/data',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/output.txt'
        )
        data1, data2 = metric.load_files()
        self.assertNotAlmostEqual(metric.metric_calc(data1, data2), np.random.normal(), places=6)


if __name__ == '__main__':
    unittest.main()
