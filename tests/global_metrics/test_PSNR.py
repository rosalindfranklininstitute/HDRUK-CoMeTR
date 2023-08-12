import unittest
from os.path import dirname, abspath

from cometr.global_metrics.PSNR import PSNR

# load in peak signal-to-noise ratio result to verify metric calculation
with open(dirname(abspath(__file__)) + '/../data/psnr_test_result', 'r') as f:
    psnr_result = float(f.read())


class PSNRTest(unittest.TestCase):
    """Test cases for the Peak Signal-To-Noise ratio (PSNR) calculation."""

    # Check error if the file does not exist
    def test_file_not_found_error(self) -> None:
        """Test if FileNotFoundError is raised when a file does not exist."""

        with self.assertRaises(FileNotFoundError):
            PSNR(
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
            PSNR(
                dirname(abspath(__file__)) + '/../data/output.txt',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

        with self.assertRaises(NameError):
            PSNR(
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
            PSNR(
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
            PSNR(
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
            PSNR(
                dirname(abspath(__file__)) + '/../data/file1_200.h5',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # Check that the type of the result is a float
    def test_result_type(self) -> None:
        """Test that the data type of the peak signal-to-noise ratio result is a float.

        """
        metric = PSNR(
            dirname(abspath(__file__)) + '/../data/file1_200.h5',
            dirname(abspath(__file__)) + '/../data/file2_200.h5',
            '/entry/data/data',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/output.txt'
        )
        data1, data2 = metric.load_files()
        self.assertEqual(type(metric.metric_calc(data1, data2)), float)

    # check consistency of the result
    def test_result_consistency(self) -> None:
        """Test consistency of the peak signal-to-noise ratio result."""
        metric = PSNR(
            dirname(abspath(__file__)) + '/../data/file1_1000.h5',
            dirname(abspath(__file__)) + '/../data/file2_1000.h5',
            '/entry/data/data',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/output.txt'
        )

        data1, data2 = metric.load_files()
        self.assertEqual(
            metric.metric_calc(data1, data2), round(psnr_result, 6)
        )


if __name__ == '__main__':
    unittest.main()
