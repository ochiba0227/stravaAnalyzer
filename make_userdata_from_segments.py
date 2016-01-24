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

##ファイル名,データ配列,保存したいキー値
def write_csv(fname,data,keys):
    header = dict([(val,val)for val in keys])
     
    with open(fname, mode='w') as f:
        data.insert(0,header)
        writer = csv.DictWriter(f, keys, extrasaction='ignore', lineterminator="\n")
        writer.writerows(data)

def sum_month(datas):
    month_list = {}
    for data in datas:
        month = data['month']
        if month in month_list:
            month_list[month] += 1
        else:
            month_list[month] = 1
    return month_list

def sum_hour(datas):
    hour_list = {}
    for data in datas:
        hour = data['hour']
        if hour in hour_list:
            hour_list[hour] += 1
        else:
            hour_list[hour] = 1
    return hour_list

def sum_userid(datas):
    userid_list = {}
    for data in datas:
        userid = data['userid']
        if userid in userid_list:
            userid_list[userid] += 1
        else:
            userid_list[userid] = 1
    return userid_list

##雑に時間を定義、季節によって変える必要あり
def update_hour(udict,hour):
    key = ""
    if hour >= 6 and hour <= 9:
        key = "morning"
    elif hour >=10 and hour <= 18 :
        key = "noon"
    else:
        key = "night"

    if key in udict:
        udict[key] += 1
    else:
        udict[key] = 1
    return udict

##季節を定義
def update_month(udict,month):
    key = ""
    if month >= 3 and month <= 5:
        key = "spring"
    elif month >=6 and month <= 8 :
        key = "summer"
    elif month >=9 and month <= 11 :
        key = "autumn"
    else:
        key = "winter"

    if key in udict:
        udict[key] += 1
    else:
        udict[key] = 1
    return udict

def update_averageSpeed(udict,averageSpeed):
    key = "averageSpeed"
    if key in udict:
        udict[key] = (udict[key]+averageSpeed)/2.0
    else:
        udict[key] = averageSpeed
    return udict    
    

##useridが指すデータが存在すればuserdata_listの更新
def update_userdata_list(userdata_list,data):
    userid = data['userid']
    hour = data['hour']
    month = data['month']
    averageSpeed = data['averageSpeed']
    for udict in userdata_list:
        if userid in udict['userid']:
            udict = update_hour(udict,int(hour))
            udict = update_month(udict,int(month))
            udict = update_averageSpeed(udict,float(averageSpeed))
            return userdata_list
    temp = {'userid':userid}
    temp = update_hour(temp,int(hour))
    temp = update_month(temp,int(month))
    temp = update_averageSpeed(temp,float(averageSpeed))
    userdata_list.append(temp)
    return userdata_list

##userid,month,hourからなるユーザデータの作成
def make_userdata(datas):
    userdata_list = []
    for data in datas:
        userdata_list = update_userdata_list(userdata_list,data)
    return userdata_list
    
##辞書からkmeansできるリストへ
def dicttolist_withkeys(dict_data,keys):
    userlist = []
    for data in userdata:
        temp = []
        for key in keys:
            if key not in data.keys():
                temp.append(0)
            else:
                temp.append(data[key])
        userlist.append(temp)
    return userlist

##並列処理の開始
def get_userdata(path):
    if os.path.isfile(path):
        print(path)
        f = open(path, 'r')
        reader = csv.DictReader(f)
        got_dict = list(reader)
        f.close()
        return make_userdata(got_dict)

##userdata_list中にuidがあればtrue
def search_userid(userdata_list,uid):
    for index in range(len(userdata_list)-1):
        if uid == userdata_list[index]['userid']:
            return index
    return -1

##他のセグメントの走行結果を結合
def marge_userdata(userdata_dict,data):
##    uid以外の要素を結合
    del userdata_dict['userid']
    for key in userdata_dict.keys():
        if key == 'averageSpeed':
            userdata_dict[key] = (userdata_dict[key] + data[key])/2.0
        elif key in data.keys():
            userdata_dict[key] = userdata_dict[key] + data[key]
##ここでuid復活
##dataのみにある要素を結合
    for key in data.keys():
        userdata_dict[key] = data[key]
    return userdata_dict

##userdata_listを一つにまとめる
def join_userdata(userdata_list,data):
##useridがある場合
    index = search_userid(userdata_list,data['userid'])
    if index != -1:
        userdata_list[index] = marge_userdata(userdata_list[index],data)
##useridがない場合
    else:
        userdata_list.append(data)
    return userdata_list

##
def join_userdata_list(orig_userdata_list):
    userdata_list = orig_userdata_list[0]
    for index in range(1,len(orig_userdata_list)-1):
        for data in orig_userdata_list[index]:
            userdata_list = join_userdata(userdata_list,data)
    return userdata_list

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
    print(datetime.now().strftime("%Y%m%d%H%M%S"))
    working_path = 'results\\season\\'
    files = os.listdir(working_path)
    files = funcs.add_workingpath(files,working_path)
    p = multiprocessing.Pool()
    p.daemon = True
    userdata_list = funcs.remove_none(p.map(get_userdata,files))
    p.close()
    print('get_userdata finished')
    userdata_list = join_userdata_list(userdata_list)
    keys = ['userid','morning','noon','night','spring','summer','autumn','winter','averageSpeed']
    funcs.write_csv("results.csv",userdata_list,keys)
    ##print(sum_month(got_dict))
    ##print(sum_hour(got_dict))
    ##print(sum_userid(got_dict))
    ##
    ##for k, v in sorted(sum_userid(got_dict).items(), key=lambda x:x[1]):
    ##    print(k, v)
    ##print(make_userdata(got_dict))

##    userdata = make_userdata(got_dict)
##    keys = ['userid','morning','noon','night','spring','summer','autumn','winter','averageSpeed']
##    funcs.write_csv("results.csv",userdata,keys)

    ##userlist = dicttolist_withkeys(userdata,keys)
    ##features = np.array(userlist)
    ####featuresからuseridを除いた値をkmeansに
    ####arrayのスライスhttp://d.hatena.ne.jp/y_n_c/20091117/1258423212
    ##kmeans_model = KMeans(n_clusters=3, random_state=10).fit(features[:,1:])
    ##labels = kmeans_model.labels_
    ##
    ##testdata = [[3, 3 , 3, 3, 3, 3, 3]]
    ##
    ##model = RandomForestClassifier()
    ##model.fit(features[:,1:], labels)
    ##output = model.predict(testdata)
    ##
    ##for label in output: print(label)
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
    print(datetime.now().strftime("%Y%m%d%H%M%S"))
