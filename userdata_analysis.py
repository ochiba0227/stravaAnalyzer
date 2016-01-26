# encoding: utf-8
import csv
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import winsound
import re
import os
from datetime import datetime

##自作関数
import funcs

##キーに基づいてデータの総和を取る
def sum_for_key(data,keys):
    num = 0
    for key in keys:
        if len(data[key])>0:
            num += int(data[key])
    return num

##データの正規化
def normlize_data(data,keys):
    num = sum_for_key(data,keys)
    for key in keys:
        if len(data[key])>0:
            data[key] = int(data[key])/num
        else:
            data[key] = int(0)
    return data

##キーに基づいて正規化
def normalize_for_key(got_dict,key_season,key_date):
    for data in got_dict:
        data = normlize_data(data,key_season)
        data = normlize_data(data,key_date)
    return got_dict

##辞書からkmeansできるリストへ
def dicttolist_withkeys(dict_data,keys):
    userlist = []
    for data in dict_data:
        temp = []
        for key in keys:
            if key not in data.keys():
                temp.append(0)
            else:
                temp.append(data[key])
        userlist.append(temp)
    return userlist

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
    print(datetime.now().strftime("%Y%m%d%H%M%S"))
    f = open('results\\user_data.csv', 'r')
    reader = csv.DictReader(f)
    got_dict = list(reader)
    f.close()

##    keys = ['userid','morning','noon','night','spring','summer','autumn','winter','averageSpeed']
    key_season = ['spring','summer','autumn','winter']
    key_date = ['morning','noon','night']
    userdata_list = normalize_for_key(got_dict,key_season,key_date)
    keys = ['userid','morning','noon','night','spring','summer','autumn','winter','averageSpeed']
    funcs.write_csv("results\\user_data_normalized.csv",userdata_list,keys)

    userlist = dicttolist_withkeys(userdata_list,keys)
    features = np.array(userlist)
    ##featuresからuseridを除いた値をkmeansに
    ##arrayのスライスhttp://d.hatena.ne.jp/y_n_c/20091117/1258423212
    kmeans_model = KMeans(n_clusters=12, random_state=10).fit(features[1:,1:])
    labels = kmeans_model.labels_

    testdata = features[1:5,1:]
    print(testdata)
    print(labels[0])

    model = RandomForestClassifier()
    model.fit(features[1:,1:], labels)
    output = model.predict(testdata)

    for label in output: print(label)
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
    print(datetime.now().strftime("%Y%m%d%H%M%S"))
