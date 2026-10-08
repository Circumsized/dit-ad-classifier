from scipy.io import loadmat
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def data_classes(class_data):
    dict_casses = {}
    for item in class_data:
        if (item[0] not in dict_casses):
            dict_casses[item[0]] = 1
        else:
            dict_casses[item[0]] += 1
    return dict_casses
def data_sites(sites_data):
    dict_sites = {}
    for item in sites_data:
        if (item[0] not in dict_sites):
            dict_sites[item[0]] = 1
        else:
            dict_sites[item[0]] += 1
    return dict_sites
def data_age(age_data):
    dict_age={40:0,50:0,60:0,70:0,80:0}
    for item in age_data:
        if(item >40 and item <50):
            dict_age[40]+=1
        elif(item<60):
            dict_age[50]+=1
        elif(item<70):
            dict_age[60]+=1
        elif(item<80):
            dict_age[70]+=1
        elif(item<90):
            dict_age[80]+=1
        elif(item <100):
            dict_age[90]+=1
        else:
            dict_age[100]+=1
    return dict_age
def data_sex(sex_data):
    dict_sex={}
    for item in sex_data:
        if(item not in dict_sex):
            dict_sex[item]=1
        else:
            dict_sex[item]+=1
    return dict_sex

def draw_bar(name_list,num_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title(u'各年龄段病人数量分布图')
    plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)

    plt.xlabel(u'人数')
    plt.ylabel(u'岁数')
    plt.show()
def pie():
    pass
if  __name__=="__main__":

    data=loadmat("MCAD_AFQ_competition.mat",mat_dtype=True)
    train_set=data["train_set"]
    train_diagnose=data["train_diagnose"]
    train_population=data["train_population"]
    train_sites=data['train_sites']
    classes_info=data_classes(train_diagnose)
    sites_info=data_sites(train_sites)
    age_info=data_age(train_population[:,1])
    age_info_withillness=data_age(train_population[:,1][train_diagnose[:,0]>1])
    sex_info=data_sex(train_population[:,0])
    print(sex_info)
    # list_class=[int(item) for item in age_info.keys()]
    # list_class=["40-50","50-60","60-70","70-80","80-90"]
    # draw_bar(list_class,age_info.values())


    # print(sex_info)
    # print(age_info_withillness)
    # print(train_population.shape)
    # print(train_diagnose.shape)
    # print(train_diagnose[:,0].shape)
    # print(train_population[:,1].shape)
    # print(train_population[:,1][train_diagnose[:,0]>1])


# import math
# for i in range(700):
#     new_data=np.zeros((20,100))
#     for j in range(20):
#         data=np.load(str(j)+".npy")
#         one_=data[i,:]
#         # one_[math.isnan(one_)]=0
#         # print(one_)
#         if(math.isnan(np.sum(one_))):
#             one_[one_!=0]=0
#         print(one_)
#         new_data[j,:]=one_
#     if(not os.path.exists("./new_data/"+str(i)+"/")):
#         os.mkdir("./new_data/"+str(i)+"/")
#     np.save("./new_data/"+str(i)+".npy",new_data)
#
# for i,item in enumerate(train_diagnose):
#     print(item[0])
#     print(type(item[0]))
#     label=np.zeros((3,))
#     label[int(item[0]-1)]=1
#     np.save("./new_label/"+str(i)+".npy",label)
#
#

