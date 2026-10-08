from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from utils import get_txt_data
import numpy as np
import random

def eval(model,val_data,val_label):
    cc = 0
    dict_sites={}
    dict_error_sites={}
    for index, item in enumerate(val_data):

        # print(item)
        pred = model.predict([item])
        label = val_label[index]
        # print(pred[0])
        if (pred[0] == label):
            cc += 1
    # list_rate_sites_error={}
    # for item in dict_sites:
    #     if(item in dict_error_sites):
    #         list_rate_sites_error[item]=dict_error_sites[item]/dict_sites[item]

    # print(list_rate_sites_error)
    # print(dict_sites)
    print(cc / len(val_label))
    return cc / len(val_label)
def train(mat_path):
    acc=0
    for i in range(5):
        train_data,train_label=get_txt_data(mat_path,"./dataset_txt/train_dataset_"+str(i)+".txt")
        train_data=np.array(train_data)
        train_label=np.array(train_label)
        val_data,val_label=get_txt_data(mat_path,"./dataset_txt/val_dataset_"+str(i)+".txt",mode="val")
        val_data=np.array(val_data)
        val_label=np.array(val_label)
        clf = SVC(class_weight="balanced")
        # clf=DecisionTreeClassifier()
        clf.fit(train_data, train_label)
        print("-------------evaling ----------------------")
        print('train_dataset_'+str(i)+".txt")
        acc+=eval(clf,val_data,val_label)
    print(acc/5)
if __name__=="__main__":
    mat_path="MCAD_AFQ_competition.mat"
    train(mat_path)





