from scipy.io import loadmat
import numpy as np
import random
import os
import random
from data_analyse import data_classes,data_sites,data_age,data_sex

def data_division():
    data = loadmat("MCAD_AFQ_competition.mat", mat_dtype=True)
    train_sites = data['train_sites']
    sites_info = data_sites(train_sites)
    start_index=0
    all_sites_sub_list=[]
    for key in sites_info:
        data_sub_list=[]
        end_index=start_index+sites_info[key]
        data_indexes=[item  for item in range(start_index,end_index)]
        random.shuffle(data_indexes)
        len_sub=len(data_indexes)//5
        for i in range(4):
            data_sub_list.append(data_indexes[i*len_sub:(i+1)*len_sub])
        data_sub_list.append(data_indexes[4*len_sub:])
        all_sites_sub_list.append(data_sub_list)
        start_index=end_index
    return all_sites_sub_list

def get_sub_txt():
    all_sites_sub_list = data_division()
    all_sites_sub_list = np.array(all_sites_sub_list)
    all_sites_sub_list = all_sites_sub_list.transpose((1, 0))
    l1, l2 = all_sites_sub_list.shape
    print(all_sites_sub_list.shape)
    for i in range(l1):
        train_list = []
        val_list = []
        for j in range(l2):
            val_list.append(all_sites_sub_list[i, j])
        list_train = [item for item in range(l1) if item != i]
        for k in list_train:
            for j in range(l2):
                train_list.append(all_sites_sub_list[k, j])
        for item1 in train_list:
            for file in item1:
                with open("./dataset_txt/train_dataset_" + str(i) + ".txt", "a")  as f:
                    f.write(str(file) + "\n")
        for item2 in val_list:
            for file in item2:
                with open("./dataset_txt/val_dataset_" + str(i) + ".txt", "a")  as f:
                    f.write(str(file) + "\n")


if __name__=="__main__":
    get_sub_txt()





