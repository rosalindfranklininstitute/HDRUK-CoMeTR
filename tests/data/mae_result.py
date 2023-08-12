from os.path import dirname, abspath
import numpy as np

from sklearn.metrics import mean_absolute_error

from cometr.global_metrics.MAE import MAE

metric = MAE(
    dirname(abspath(__file__)) + '/../data/file1_200.h5',
    dirname(abspath(__file__)) + '/../data/file2_200.h5',
    '/entry/data/data',
    '/entry/data/data',
    dirname(abspath(__file__)) + '/../data/output.txt'
)
data1, data2 = metric.load_files()
result = mean_absolute_error(data1, data2)

# insert the result in an array
output = np.empty([1], dtype=float)
output[0] = result

# save result in a text file
np.savetxt("mae_test_result", X=output, fmt="%f", delimiter="", newline="")

