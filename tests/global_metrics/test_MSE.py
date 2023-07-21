import unittest
import numpy as np

# from cometr.global_metrics.MSE import calc_mse
from src.global_metrics.MSE import MSE


class MSETest(unittest.TestCase):

    # check error if the file does not exist
    def test_file_not_found_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as e:
            MSE('../data/file3_300', '../data/file2_1000', './output.txt').check_file_exists()
        self.assertEqual(str(e.exception), "File not found")

    # check the error if one of the files is not in h5py format
    def test_h5pyfile(self) -> None:
        with self.assertRaises(ValueError) as e:
            MSE('../data/output.txt', '../data/file2_1000').is_h5py_file()
        self.assertEqual(str(e.exception), "One or both files are not in HDF5 format.")

    # check the error if the output file format is not txt
    def test_outputfile(self) -> None:
        with self.assertRaises(ValueError) as e:
            MSE('../data/file1_1000', '../data/file2_1000', './output').verify_output_file()
        self.assertEqual(str(e.exception), "The output file must be in .txt format.")

    # check the error if the data is not in the standard dictionary format
    def test_key_error(self) -> None:
        with self.assertRaises(KeyError) as e:
            MSE('../data/file1_1000', '../data/file3_100', './output.txt').calc_mse()
        self.assertEqual(e.exception.args[0], "The /entry/data/data key is not found in the HDF5 file.")

    # check error if dimensions of the data do not match
    def test_dim_error(self) -> None:
        with self.assertRaises(ValueError) as e:
            MSE('../data/file1_200', '../data/file2_1000', './output.txt').calc_mse()
        self.assertEqual(str(e.exception), 'Dimensions do not match')

    # check that the calc_mse's results are consistent
    def test_MSE_result_error(self) -> None:
        with self.assertRaises(ValueError):
            result = MSE('../data/file1_200', '../data/file2_1000').calc_mse()
            self.assertEqual(result, np.random.rand())


if __name__ == '__main__':
    unittest.main()
