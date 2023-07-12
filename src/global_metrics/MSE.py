import h5py
import sys
import numpy as np

from sklearn.metrics import mean_squared_error

# print ("Number of arguments:", len(sys.argv), "arguments")
# print ("Argument List:", str(sys.argv))
# under_samp_file = '/ceph/users/yff85561/data/undersampled_original_rec.h5'
# ground_truth_file = '/ceph/users/yff85561/data/ground_truth_rec.h5'
# network_improved_file = '/ceph/users/yff85561/data/network_improved_rec.h5'

file1 = sys.argv[1]
file2 = sys.argv[2]

def read_file(file1, file2):
    File1 = h5py.File(file1, 'r')
    File2 = h5py.File(file2, 'r')

    file_data_1 = File1['entry']['data']['data']
    file_data_2 = File2['entry']['data']['data']

    # print(list(File1.keys()))
    # print(list(File2.values()))
    # print(len(file_data_1[0:1000:10]))
    # print(len(file_data_2[0:1000:10]))
    # print(file_data_1[0:1000:10])
    # print(file_data_2[0:1000:10])
    return file_data_1, file_data_2

file1_data, file2_data =  read_file(file1, file2)

print(file1_data.shape)
print(file2_data.shape)


#convert data to 2D to use MSE metric
file1_2D = np.reshape(file1_data, (file1_data.shape[0],-1))
file2_2D = np.reshape(file2_data, (file2_data.shape[0],-1))

mse = mean_squared_error(file1_2D, file2_2D)
print("Mean Squared Error:", mse)


