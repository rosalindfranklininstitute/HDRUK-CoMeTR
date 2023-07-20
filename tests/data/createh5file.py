import h5py
import numpy as np


def create_h5_files(file_1: str, file_2: str) -> None:
    file_1 = h5py.File(file_1, 'w')
    file_2 = h5py.File(file_2, 'w')

    # Create a dataset in the H5 files
    data1 = np.random.rand(200, 200)
    data2 = np.random.rand(200, 200)

    file_1.create_dataset('entry/data/data', data1.shape, data=data1)
    file_2.create_dataset('entry/data/data', data2.shape, data=data2)

    # Close the files
    file_1.close()
    file_2.close()


create_h5_files('file1_200', 'file2_200')
