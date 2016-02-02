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
####入力データと入力ラベルに基づいたニューラルネットワークの分類器のチューニング
##def tune_nn_classifier(features,labels):
##    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
##        features,labels,test_size=0.5,random_state=1)
##    parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,7))),
##                   'hidden_layer_sizes':list(range(50,150))}]
##    model = MLPClassifier(activation='logistic',algorithm='l-bfgs', random_state=1)
##    clf = GridSearchCV(model, parameters, cv=5, n_jobs=3)
##    clf.fit(train_data,train_label)
##    print(clf.best_estimator_)
##    predicted_label = clf.predict(test_data)
##    print(classification_report(test_label, predicted_label,digits = 3))
##    return clf
##
####入力データと入力ラベルに基づいてモデルに最適なパラメータを探索
##def tune_model(features,labels,model,parameters):
##    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
##        features,labels,test_size=0.5,random_state=1)
##    clf = GridSearchCV(model, parameters, cv=5, n_jobs=3)
##    clf.fit(train_data,train_label)
##    print(clf.best_estimator_)
##    return clf    
##
####入力データと入力ラベルに基づいたランダムフォレストの予測器のチューニング
##def tune_rf_regressor(features,labels):
##    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
##        features,labels,test_size=0.5,random_state=1)
##    parameters = [{'max_features':['auto','log2'],
####n_estimatorsは<50で探索すべき
##                   'n_estimators':list(range(10,50,10))}]
##    model = RandomForestRegressor(random_state=1)
##    clf = GridSearchCV(model, parameters, cv=5, n_jobs=3)
##    clf.fit(train_data,train_label)
##    print(clf.best_estimator_)
##    return clf
##
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
##
####    得たラベルから速度計算のためのRandomForestの学習
##def regression_rf(features,labels,model):
##    ##    トレーニングデータとテストデータを半分に割った場合
##    training_data,test_data,training_label,test_label = cross_validation.train_test_split(
##        features,labels,test_size=0.5,random_state=1)
##    model.fit(training_data,training_label)
##    predicted_label = model.predict(test_data)
##    print(model)
##    print(r2_score(test_label,predicted_label))
####各木で出た数値の平均を取れないか？
##    difference = []
##    for d in np.c_[test_label,predicted_label]:
##        difference.append(abs(float(d[0])-float(d[1])))
##    print(max(difference)*3.6)
##    print(min(difference)*3.6)
##    print(sum(difference)/len(difference)*3.6)
##    print(sum(difference)/len(difference)*60)
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
##    print(sum(difference)/len(difference)*60)
##
####listにuidが存在するか確認
##def find_userid(datalist,uid):
##    for data in datalist:
##        if uid == data[0]:
##            return True
##    return False
##
####指定されたインデックスのデータをユーザごとに平均をとる
####def sum_userdata(all_data,indexes):
####    userdata = []
####    for data in all_data:
####        if find_userid(userdata,data[0]):
##
####各列毎の平均値をとる
##def get_average(tup):
##    uid = tup[0]
##    data = tup[1]
##    speed_array = np.array(data['data']).astype(np.float)[:,1:4]
####    nan要素をマスキングで計算に現れずにする
##    masked_array = np.ma.masked_array(speed_array,np.isnan(speed_array))
##    speed = np.mean(masked_array,axis=0)
##    ##        np.nansumでnanを無視した合計を取ってくれる便利なヤツあり
####    maskした後mean取ると--になっているところを無視して平均取ってくれる
##    return uid, speed,len(data['climbCategory'])
##
####keyに基づいてdicからデータを取得
####keyのなかでデータがないものがあれば省く
##def make_onedata(dic,keys):
##    data = []
##    for key in keys:
##        d = dic[key]
####        データが存在しない場合無視するためにNoneを代入
##        if d == '-1':
##            d = None
##        data.append(d)
##    return data,dic['userid'],dic['climbCategory']
##
####ユーザデータにラベルを付与
####現在は欠損値があるデータを削除している
##def add_labels(tup,uids,labels):
##    uid = tup[0]
##    data = tup[1]
##
##    if uid not in uids:
##        return None
##    data = np.array(data['data']).astype(np.float)
####    欠損値を削除
##    data = data[~np.isnan(data).any(axis=1)]
##    if len(data) == 0:
##        print(uid)
##        return None
##    label = np.array([labels[uids.index(uid)]]*len(data))
##    return np.c_[data,label]
##
####climbCategory毎の回数を記録
####pythonは引数で与えられた変数を直接弄る？
####dictだからメモリが引数として与えられている？
##def update_climbCategory(dic,climbCategory):
##    if climbCategory in dic.keys():
##        dic[climbCategory] += 1
##    else:
##        dic[climbCategory] = 1
##    return dic


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

##各クラスタに所属するデータが要求数あるか
def hasdata_eachlabel(dic,range_,target_datanum):
    datakeys = dic.keys()
    for label in range_:
        try:
            if len(dic[label][0])<target_datanum:
                return False
        except Exception as e:
            return False      
    return True

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    starttime = time.time()
    funcs.start_program()
    np.random.seed(1)
    random.seed(1)
    
    path_labeled = 'results\\data_for_regression\\20160203_011146\\labeled'
    files = os.listdir(path_labeled)
##    random.shuffle(files)

    target_datanum = 100

    km_range = np.array(range(9)).astype(np.str)
    km_labeled = {}
    km_ok = False
    
    gm_range = np.array(range(5)).astype(np.str)
    gm_labeled = {}
    gm_ok = False

    for file in files:
##        [{},{}]の形式でデータを取得
        data = funcs.read_myjson(os.path.join(path_labeled,file),1,True)
        if km_ok is False:
            for d in data:
                add_labeleddata(km_labeled,d,target_datanum,'km_label')
            if hasdata_eachlabel(km_labeled,km_range,target_datanum) is True:
                km_ok = True
                
        if gm_ok is False:
            for d in data:
                add_labeleddata(gm_labeled,d,target_datanum,'gm_label')
            if hasdata_eachlabel(gm_labeled,gm_range,target_datanum) is True:
                gm_ok = True

        if km_ok is True and gm_ok is True:
            break
    darray = np.array(km_labeled[km_range[0]])
    print(darray)
    funcs.end_program()
