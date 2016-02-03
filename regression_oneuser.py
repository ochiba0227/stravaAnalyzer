# encoding: utf-8
import time
import random
import os
import csv
import itertools
import numpy as np
from scipy import linalg
from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, k_means
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix,r2_score
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn import mixture
from sklearn.externals import joblib
from sklearn.preprocessing import scale
import pandas as pd
import multiprocessing
import functools
import funcs
import json
from datetime import datetime

####グリッドサーチの結果をファイルに出力
##def write_gridres(name,data):
##    csvfile = open(name+".csv","w")
##    writer = csv.writer(csvfile)
##    writer.writerows(data.grid_scores_)
##    csvfile.close()
##
####辞書からkmeansできるリストへ
##def dicttolist_withkeys(dict_data,keys):
##    userlist = []
##    for data in dict_data:
##        temp = []
##        for key in keys:
##            if key.find("Speed") == 1:
##                temp.append(float(data[key]))
##            else:
##                temp.append(data[key])
##        userlist.append(temp)
##    return userlist
##
##入力データと入力ラベルに基づいたニューラルネットワークの分類器のチューニング
def tune_nn_classifier(features,labels):
    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,5))),                    
                   'hidden_layer_sizes':list(range(50,150,10)),
                   'algorithm':['l-bfgs','adam']}]
    model = MLPRegressor(activation='logistic', random_state=1)
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=-2)
    clf.fit(train_data,train_label)
    print(clf.best_estimator_)
    return clf
##入力データと入力ラベルに基づいたランダムフォレストの予測器のチューニング
def tune_rf_regressor(features,labels):
    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    parameters = [{'max_features':['auto','log2'],
##n_estimatorsは<50で探索すべき
                   'n_estimators':list(range(10,50,10))}]
    model = RandomForestRegressor(random_state=1)
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=-2)
    clf.fit(train_data,train_label)
    print(clf.best_estimator_)
    return clf

####    得たラベルからユーザタイプ分類のためのNNの学習
####    inputとtestデータは元データからそれぞれ半分ずつ
##def learn_nn(features,labels):
##    ##    トレーニングデータとテストデータを半分に割った場合
##    training_data,test_data,training_label,test_label = cross_validation.train_test_split(
##        features,labels,test_size=0.5,random_state=1)
##    clf = MLPClassifier(algorithm='l-bfgs', alpha=0.1, hidden_layer_sizes=83, random_state=1)
##    clf.fit(training_data, training_label)
##    predicted_label = clf.predict(test_data)
##    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
##    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
##    
##    print(classification_report(test_label, predicted_label, digits = 3))
##    print(confusion_matrix(test_label, predicted_label))
####    クロスバリデーション
##    scores = cross_validation.cross_val_score(clf,features,labels,cv=10)
##    print(scores)
##    print((scores.mean(), scores.std() * 2))
##    preds = cross_validation.cross_val_predict(clf,features,labels,cv=10)
##    print(preds)
##    print((preds.mean(), preds.std() * 2))
##
##    得たラベルから速度計算のためのNNの学習
def regression_nn(features,labels,model):
    training_data,test_data,training_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    model.fit(training_data,training_label)
    predicted_label = model.predict(test_data)
    print(model)
    print(r2_score(test_label,predicted_label))
    difference = []
    for d in np.c_[test_label,predicted_label]:
        difference.append(abs(float(d[0])-float(d[1])))
    print(max(difference)*3.6)
    print(min(difference)*3.6)
    print(sum(difference)/len(difference)*3.6)
    print(sum(difference)/len(difference)*60)

##    得たラベルから速度計算のためのRandomForestの学習
def regression_rf(features,labels,model):
    ##    トレーニングデータとテストデータを半分に割った場合
    training_data,test_data,training_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    model.fit(training_data,training_label)
    predicted_label = model.predict(test_data)
    print(model)
    print(r2_score(test_label,predicted_label))
##各木で出た数値の平均を取れないか？
    difference = []
    for d in np.c_[test_label,predicted_label]:
        difference.append(abs(float(d[0])-float(d[1])))
    print(max(difference)*3.6)
    print(min(difference)*3.6)
    print(sum(difference)/len(difference)*3.6)
    print(sum(difference)/len(difference)*60)
##
####重回帰分析
##def learn_ra(all_data):
##    features = all_data[:,1:].astype('float')
##    number = len(features)
##    half = int(number / 2)
##
##    training_data = features[:half,1:].astype('float')
##    training_label = features[:half,0].astype('float')
####    test_data = np.c_[features[number-half:,1:len(keys)-2].astype('float'),labels[number-half:]]
##    test_data = features[number-half:,1:].astype('float')
##    test_label = features[number-half:,0]
####    model = LinearRegression()
####    バギングすればより良い結果がでるらしい。
####    clf = BaggingClassifier(base_estimator=clf, n_estimators=100, max_samples=0.9, max_features=0.2,n_jobs=4)
##    model = MLPRegressor(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,9), random_state=1)
##    model.fit(training_data,training_label)
##    predicted_label = model.predict(test_data)
##    difference = []
##    for d in np.c_[test_label,predicted_label]:
##        difference.append(abs(float(d[0])-float(d[1])))
##    print(max(difference)*3.6)
##    print(min(difference)*3.6)
##    print(sum(difference)/len(difference)*3.6)

##dicを更新
##各クラスタに所属するデータを追加
##target条件を満たしていたら追加しない
def add_labeleddata(dic,data,target_datanum,labelname):
    firsttime = False
    label = data[labelname]
    if label not in dic.keys():
        dic[label] = []
        firsttime = True
    elif len(dic[label][0]) >= target_datanum:
        return
    keys = ['GRADE','ALTITUDE','DISTANCE','VELOCITY']
    for key in keys:
        if firsttime:
            dic[label].append(data[key])
        else:
            index = keys.index(key)
            dic[label][index].extend(data[key])

##filesで与えられたファイルが同じラベルを持つか返すクラス
def search_haslabel(files,target_label,label_str):
    ret_list = []
    for file in files:
        data = funcs.read_myjson(os.path.join(path_labeled,file),1,False)
        for d in data:
            if target_label == d[label_str]:
                ret_list.append(True)
            else:
                ret_list.append(False)
    return ret_list

##dicを更新
##各クラスタに所属するデータを追加
def update_userdata(dic,data):
    climbCategory = data['climbCategory']
    if climbCategory not in dic.keys():
        dic[climbCategory] = []
        
    keys = ['userid','km_label','gm_label','GRADE','ALTITUDE','DISTANCE','VELOCITY']
    temp_dict = {}
    for key in keys:
        temp_dict[key] = data[key]
    dic[climbCategory].append(temp_dict)
    return temp_dict

##filesに指定されたファイルを読み込みユーザデータを取得
def get_userdata(data_users,files,readrows,randflag):
    for file in files:
        if file in data_users.keys():
            continue
        data_users[file] = {}
        data = funcs.read_myjson(os.path.join(path_labeled,file),readrows,randflag)
        for d in data:
            update_userdata(data_users[file],d)
    return data_users

##各ユーザのデータを取得
def get_data_eachuser(data_users,keys,need_num):
    rundata_dict = {}
    for unamekey in data_users.keys():
        rundata_dict[unamekey] = {}
        rundata_dict[unamekey]['datanum'] = {}
        data = data_users[unamekey]
        for catnamekey in data.keys():
            temp_array = [[],[],[],[]]
            data = data_users[unamekey][catnamekey]
            rundata_dict[unamekey]['datanum'][catnamekey] = 0
            for run_data in data:
                for key in keys:
                    index = keys.index(key)
                    temp_array[index].extend(run_data[key])
##                    データ数を数える
                    if key is 'VELOCITY':
                        rundata_dict[unamekey]['datanum'][catnamekey]+=len(run_data[key])
            rundata_dict[unamekey][catnamekey]=temp_array
##            データ数がneed_numに達しているか確認
            if(rundata_dict[unamekey]['datanum'][catnamekey]<need_num):
                del rundata_dict[unamekey]
                break
            
    return rundata_dict

##各ユーザのデータをそろえる
def align_data_eachuser(actdata_dict,keys,need_num):
##個数のカウント
    for unamekey in actdata_dict.keys():
        data = actdata_dict[unamekey]
        for catnamekey in data.keys():
            data = actdata_dict[unamekey][catnamekey]
            if catnamekey is 'datanum':
                ##            記録しているデータ数はlengthで取得している
                for catnamekey_fornum in data.keys():
                    data = actdata_dict[unamekey][catnamekey][catnamekey_fornum]
                    if data < need_num:
                        need_num = data
                break

##データをneed_numにそろえる
    for unamekey in actdata_dict.keys():
        data = actdata_dict[unamekey]
        for catnamekey in data.keys():
            data = actdata_dict[unamekey][catnamekey]
            if catnamekey is not 'datanum':
                length = actdata_dict[unamekey]['datanum'][catnamekey]
##                長さ分のindexを確保
                pickup_indexes = range(length)
##                シャッフル（戻り値None）
                random.shuffle(list(pickup_indexes))
##                所望の長さへ変換
                pickup_indexes = pickup_indexes[:need_num]
                for key in keys:
                    index = keys.index(key)
##                    pickup_indexesがさすindexのデータを取得
                    data = actdata_dict[unamekey][catnamekey][index]
                    temp_list = []
                    for pickup_index in pickup_indexes:
                        temp_list.append(data[pickup_index])
                    actdata_dict[unamekey][catnamekey][index] = temp_list

##データの準備
def preparation_data(label_str):
    ##    学習で使用するデータの準備
    middata_name = 'middata.json'
    filepath = funcs.get_filepath(middata_name,None)
    data_from_file_flag = False
    if data_from_file_flag or os.path.exists(filepath) is False:
        print("Make Data!!!!")
        files = ['614307.myjson']
    ##    data_usersの初期化
        data_users = get_userdata({},files,None,False)

    ##    同一ラベルを持つユーザの探索
        target_label = data_users['614307.myjson']['FLAT'][0][label_str]
        files = os.listdir(path_labeled)
        random.shuffle(files)
        files=np.array(files[:10])
        flags = np.array(search_haslabel(files,target_label,label_str))
        files = files[flags==True]

    ##    data_usersの更新
        get_userdata(data_users,files,None,False)
        
##        ファイルへ書き込み
        try:
            f = funcs.get_fileobj(middata_name,'w',None)
            json.dump(data_users,f)
        finally:
            f.close()
    else:
        print("Data from FILE")
        try:
            f = funcs.get_fileobj(middata_name,'r',None)
            data_users = json.load(f)
        finally:
            f.close()
    return data_users

##[climbCategory][label][data]
##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    starttime = datetime.now()
    funcs.start_program()
    np.random.seed(1)
    random.seed(1)
    
    path_labeled = 'results\\data_for_regression\\20160203_011146\\labeled'
    files = os.listdir(path_labeled)
    random.shuffle(files)

##    climb_cat = ['FLAT', 'CATEGORY1','CATEGORY2',  'CATEGORY3', 'CATEGORY4' ,'HORS_CATEGORIE']
    km_range = np.array(range(9)).astype(np.str)

##    ラベルに基づいてデータを取得
    label_str = 'km_label'
    data_users = preparation_data(label_str)

    keys = ['GRADE','ALTITUDE','DISTANCE','VELOCITY']
    ##最低データ点数を指定
    need_num = 1000
    actdata_dict = get_data_eachuser(data_users,keys,need_num)
    ##欲しいデータ点数を指定
    need_num = 1000
    align_data_eachuser(actdata_dict,keys,need_num)
    print(actdata_dict.keys())
##    カテゴリFLATにてテスト
    try:
##        データ数をそろえる
        
        uid =''
        

        rundata_dict = {}
        for unamekey in data_users.keys():
            rundata_dict[unamekey] = {}
            data = data_users[unamekey]
            for catnamekey in data.keys():
                temp_array = [[],[],[],[]]
                data = data_users[unamekey][catnamekey]
                for run_data in data:
                    for key in keys:
                        index = keys.index(key)
                        temp_array[index].extend(run_data[key])
                rundata_dict[unamekey][catnamekey]=temp_array
                    

##        3データでやった場合
        darray = np.array(darray)
        features = scale(darray[:3].transpose())
        labels = scale(darray[3].transpose())
        training_data,test_data,training_label,test_label = cross_validation.train_test_split(
            features,labels,test_size=0.5,random_state=1)

        
        model = MLPRegressor(activation='logistic', random_state=1)
        model.fit(training_data,training_label)
        pred = model.predict(test_data)
        r2_score(test_label,pred)
    finally:
        pass
    funcs.end_program()
##    gm_range = np.array(range(5)).astype(np.str)
##    gm_labeled = {}
##    gm_ok = False

####　各ユーザから全データの取得
##    for file in list(files):
##        data = funcs.read_myjson(os.path.join(path_labeled,file),None,False)
##        for d in data:
##            add_labeleddata_climbcategory(km_labeled,d,'km_label')
####        全タイプの走行データがあるか確認
##        if len(km_labeled.keys())==8:
##            counter = 0
####            各タイプの各クライムカテゴリについてデータの存在を確認
##            for key in km_labeled.keys():
##                if len(km_labeled[key].keys()) == 6:
##                    counter +=1
####            全部にデータが存在すれば
##            if counter == 8:
##                break
##    try:
##        file = funcs.get_fileobj('data_each.json','w',None)
##        writer = json.dump(km_labeled,file)
##    finally:
##        file.close()

##    try:
##        file = funcs.get_fileobj('datatest.json','r',None)
##        km_labeled = json.load(file)
##    finally:
##        file.close()

##    for label in km_range:
##        for categorie in km_labeled[label].keys():
##            print(str(label)+':'+str(categorie))
            
    
##    for file in files:
####        [{},{}]の形式でデータを取得
##        data = funcs.read_myjson(os.path.join(path_labeled,file),1,True)
##        if km_ok is False:
##            for d in data:
##                add_labeleddata(km_labeled,d,target_datanum,'km_label')
##            if hasdata_eachlabel(km_labeled,km_range,target_datanum) is True:
##                km_ok = True
##                
##        if gm_ok is False:
##            for d in data:
##                add_labeleddata(gm_labeled,d,target_datanum,'gm_label')
##            if hasdata_eachlabel(gm_labeled,gm_range,target_datanum) is True:
##                gm_ok = True
##
##        if km_ok is True and gm_ok is True:
##            break


##    f = open('gm_labeled.json')
##    gm_labeled = json.load(f)
##    f.close()
##        
##    km_nn_models = []
##    km_rf_models = []
##    for index in km_range:
##        darray = np.array(km_labeled[km_range[index]])
##        data = scale(darray[:3].transpose())
##        labels = scale(darray[3].transpose())
##        model = tune_nn_classifier(data,labels)
##        km_nn_models.append(model)
##        model = tune_nn_classifier(data,labels)
##        km_rf_models.append(model)
##        
##    gm_nn_models = []
##    gm_rf_models = []
##    for index in gm_range:
##        darray = np.array(gm_labeled[gm_range[index]])
##        data = scale(darray[:3].transpose())
##        labels = scale(darray[3].transpose())
##        model = tune_nn_classifier(data,labels)
##        gm_nn_models.append(model)
##        model = tune_nn_classifier(data,labels)
##        gm_rf_models.append(model)

    
