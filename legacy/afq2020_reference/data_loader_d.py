
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_data_index_list
from scipy.io import loadmat
from get_data import get_data
import random
class data_loader(Dataset):

    def __init__(self, mat_path,txt_path,mode="train",transform=None):
        self.transform=transform
        self.data=loadmat(mat_path,mat_dtype=True)
        self.data_set = self.data["train_set"]
        self.data_diagnose = self.data["train_diagnose"]
        self.data_population = self.data["train_population"]
        self.new_data_set=get_data(self.data_set)
        if(mode=="train"):
            all_list=[]
            img_name_list = get_data_index_list(txt_path)
            for item in img_name_list:
                if(self.data_diagnose[item][0]==1):
                    all_list.append(item)
                all_list.append(item)
            random.shuffle(all_list)
            self.img_name_list=all_list
        else:
            self.img_name_list = get_data_index_list(txt_path)

    def __len__(self):
        return len(self.img_name_list)
    def __getitem__(self, idx):
        index=self.img_name_list[idx]
        data=self.new_data_set[index]
        label=self.data_diagnose[index][0]
        if(label>1):
            label=2
        new_data=np.zeros((20,102))
        new_label = np.zeros((2,))
        # print(label.shap
        max_data=np.max(data)
        min_data=np.min(data)
        data=(data-min_data)/(max_data-min_data)
        # new_data[:,:]=data
        for i in range(20):
            new_data[i,:100]=data[i,:]
            new_data[i,100]=self.data_population[i,0]
            new_data[i,101]=self.data_population[i,1]/150
        if (label > 1):
            new_label[1] = 1
        else:
            new_label[0] = 1
        return new_data,new_label

if __name__=="__main__":
    jj=data_loader("MCAD_AFQ_competition.mat","./dataset_txt/val_dataset_0.txt")