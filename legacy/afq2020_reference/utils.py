from scipy.io import loadmat
import numpy as np
import random
import os
from  data2pca  import getPca_data
from data_analyse import data_classes
import math
from get_data import get_data

def get_data_flatten(data):
    data=get_data(data)
    list_datas = []
    for item in data:
        list_datas.append(item.flatten())
    return list_datas
def get_data_index_list(txt_path):
    list_data_indexes=[]
    with open(txt_path,"r")  as f:
        for line in f.readlines():
            data_index=int(line.strip())
            # print(data_index)
            list_data_indexes.append(int(data_index))
    return list_data_indexes
def get_txt_data(mat_path,txt_path,mode="train"):
    list_data=[]
    list_label=[]

    data=loadmat("MCAD_AFQ_competition.mat",mat_dtype=True)
    train_diagnose=data["train_diagnose"]
    train_site=data["train_sites"]
    all_data= get_norm_addsex_age_data(mat_path)
    data_indexes=get_data_index_list(txt_path)
    random.shuffle(data_indexes)
    rasmple_idata_indexes=[]
    for item in data_indexes:
        label=train_diagnose[item][0]
        if(mode=="train"):
            if(label==1):
                rasmple_idata_indexes.append(item)
                rasmple_idata_indexes.append(item)
                rasmple_idata_indexes.append(item)
            rasmple_idata_indexes.append(item)
        else:
            rasmple_idata_indexes.append(item)
    random.shuffle(rasmple_idata_indexes)
    for item in rasmple_idata_indexes:
        label = train_diagnose[item][0]
        if (label >= 2):
            # continue
            label = 2
        list_data.append(all_data[item])
        list_label.append(label)
    return list_data,list_label
def get_source_data(mat_path):
    data = loadmat(mat_path, mat_dtype=True)
    train_set = data["train_set"]
    datas = get_data_flatten(train_set)
    return datas
def get_RD_data(mat_path):
    return getPca_data(mat_path)

def get_norm_data(data):
    min_array=data.min(axis=1)
    max_array=data.max(axis=1)
    for i in range(data.shape[1]):
        data[:,i]=(data[:,i]-min_array[i])/(max_array[i]-min_array[i])
    # min_data=np.min(data)
    # max_data=np.max(data)
    # data=(data-min_data)/(max_data-min_data)
    return data

def get_data_set_sex_age(mat_path):
    data=loadmat(mat_path,mat_dtype=True)
    train_population=data["train_population"]
    model,set_data=getPca_data(mat_path)
    set_data=np.array(set_data)
    # print("ggggggggggg")
    # print(set_data)
    train_population[:,1]/=150
    data=np.append(set_data,train_population,axis=1)
    return data
def get_norm_addsex_age_data(mat_path):
    data=get_data_set_sex_age(mat_path)
    # model, data = getPca_data(mat_path)
    # data=get_source_data(mat_path)
    data = np.array(data)
    # norm_data=get_norm_data(data)
    # print(norm_data)
    return data

if __name__=="__main__":
    mat_path="MCAD_AFQ_competition.mat"
    norm_data=get_norm_addsex_age_data(mat_path)
    # model,data=get_RD_data(mat_path)
    # data=np.array(data)
    # print(data.shape)
    # hh=get_norm_data(data)
    # print(data.shape)
    # print(data.argmax(axis=0).shape)







