from scipy.io import loadmat
import numpy as np
import math
from sklearn.decomposition import PCA
from get_data import get_data
import random
def pca(data):
    list_datas=[]
    for item in data:
        list_datas.append(item.flatten())
    pca = PCA(n_components=100)
    newX = pca.fit_transform(list_datas)
    return pca,newX

def getPca_data(mat_path):
    data = loadmat(mat_path, mat_dtype=True)
    train_set = data["train_set"]
    datas = get_data(train_set)
    return pca(datas)

if __name__=="__main__":
    # data = loadmat("./MCAD_AFQ_competition.mat", mat_dtype=True)
    # train_set = data["train_set"]
    # datas=get_data(train_set)
    # pca(datas)
    mat_path="./MCAD_AFQ_competition.mat"
    model,new_data=getPca_data(mat_path)