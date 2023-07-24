import unittest
import numpy as np
from os.path import dirname, abspath

from cometr.global_metrics.MSE import MSE


class MSETest(unittest.TestCase):

    # check error if the file does not exist
    def test_file_not_found_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as e:
            metric = MSE(
                dirname(abspath(__file__)) + '/../data/file3_300.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # check the error if one of the files is not in h5py format
    def test_h5pyfile(self) -> None:
        with self.assertRaises(TypeError) as e:
            metric = MSE(
                dirname(abspath(__file__)) + '/../data/output.txt',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

        with self.assertRaises(NameError) as e:
            metric = MSE(
                dirname(abspath(__file__)) + '/../data/file4_100',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )

    # check the error if the output file format is not txt
    def test_outputfile(self) -> None:
        with self.assertRaises(ValueError) as e:
            metric = MSE(
                dirname(abspath(__file__)) + '/../data/file1_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output'
            )

    # check the error if the data is not in the standard dictionary format
    def test_key_error(self) -> None:
        with self.assertRaises(KeyError) as e:
            metric = MSE(
                dirname(abspath(__file__)) + '/../data/file1_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/file3_100.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )
            metric.calc_mse()

    # check error if dimensions of the data do not match
    def test_dim_error(self) -> None:
        with self.assertRaises(ValueError) as e:
            metric = MSE(
                dirname(abspath(__file__)) + '/../data/file1_200.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/file2_1000.h5',
                '/entry/data/data',
                dirname(abspath(__file__)) + '/../data/output.txt'
            )
            metric.calc_mse()

    # check that the calc_mse's results are consistent
    def test_MSE_result(self) -> None:
        metric = MSE(
            dirname(abspath(__file__)) + '/../data/file1_200.h5',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/file2_200.h5',
            '/entry/data/data',
            dirname(abspath(__file__)) + '/../data/output.txt'
            )
        metric.calc_mse()


if __name__ == '__main__':
    unittest.main()
