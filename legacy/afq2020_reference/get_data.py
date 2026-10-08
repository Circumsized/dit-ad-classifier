import numpy as np
import math
def get_data(input_datas):
    list_data = []
    num_test_data = input_datas[0, 0][0].shape[0]
    for i in range(20):
        di = input_datas[i, 0][0]
        list_data.append(di)
    all_data_list = []

    for k in range(num_test_data):
        new_data = np.zeros((20, 100))
        for j in range(20):
            data = list_data[j]
            one_ = data[k, :]
            if (math.isnan(np.sum(one_))):
                one_[one_ != 0] = 0
            new_data[j, :] = one_

        all_data_list.append(new_data)
    return np.array(all_data_list)