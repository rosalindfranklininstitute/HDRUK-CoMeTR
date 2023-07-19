import unittest
import numpy as np

from src.global_metrics.MSE import calc_mse


class MSETest(unittest.TestCase):

    def test_MSE_assert_error(self) -> None:
        with self.assertRaises(AssertionError):
            result = calc_mse('../data/file1_200', '../data/file2_200')
            self.assertEqual(result, np.random.rand())

    def test_value_error(self) -> None:
        with self.assertRaises(ValueError) as e:
            calc_mse('../data/file1_1000', '../data/file1_200')

        self.assertEqual(str(e.exception), 'Dimensions do not match')


if __name__ == '__main__':
    unittest.main()
