from os.path import dirname, abspath
import numpy as np

from skimage.metrics import peak_signal_noise_ratio

from cometr.global_metrics.PSNR import PSNR

metric = PSNR(
    dirname(abspath(__file__)) + '/../data/file1_1000.h5',
    dirname(abspath(__file__)) + '/../data/file2_1000.h5',
    '/entry/data/data',
    '/entry/data/data',
    dirname(abspath(__file__)) + '/../data/output.txt'
)
data1, data2 = metric.load_files()
result = peak_signal_noise_ratio(data2, data1, data_range=data2.max()-data2.min())

# insert the result in an array
output = np.empty([1], dtype=float)
output[0] = result

# save result in a text file
np.savetxt("psnr_test_result", X=output, fmt="%f", delimiter="", newline="")