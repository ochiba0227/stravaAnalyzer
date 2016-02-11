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
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix,r2_score,explained_variance_score,mean_absolute_error
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn import mixture
from sklearn.externals import joblib
from sklearn.preprocessing import scale, Normalizer,StandardScaler
import pandas as pd
import multiprocessing
import functools
import funcs
import json
from datetime import datetime
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.linear_model import PassiveAggressiveRegressor

def tuneSVM(features,labels):
    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    C = np.logspace(3.5, 4.5, 10)
    gamma = np.logspace(-4, 4, 10)
    parameters = [{'kernel':['rbf'],'C':C,'gamma':gamma}]
    model = svm.SVR()
    clf = GridSearchCV(model, parameters, n_jobs=-2)
    clf.fit(train_data,train_label)
    print(clf.best_estimator_)
    return clf    

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

####filesに指定されたファイルを読み込みユーザデータを取得
##def get_userdata(data_users,files,readrows,randflag,path_labeled,):
##    for file in files:
##        if file in data_users.keys():
##            continue
##        data_users[file] = {}
##        data = funcs.read_myjson(os.path.join(path_labeled,file),readrows,randflag)
##        for d in data:
##            update_userdata(data_users[file],d)
##    return data_users

#データ数のチェック
def check_datanum(data_users,need_num):
##    全カテゴリに対してデータ数のチェックひとつでも少なければダメ
    for catnamekey in data_users.keys():
        data = data_users[catnamekey]
        counter = 0
        for run_data in data:
##                    データ数を数える
            counter+=len(run_data['VELOCITY'])
##            データ数がneed_numに達しているか確認
        if(counter<need_num):
            return False
    return True

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
    return actdata_dict

##分析用データを作成
##nparray形式でデータを返す
def make_dataarray(actdata_dict,keys,catnamekey):
    temp_array = []
    for unamekey in actdata_dict.keys():
        data = actdata_dict[unamekey][catnamekey]
        if catnamekey is not 'datanum':
            for key in keys:
                index = keys.index(key)
                if(len(temp_array)<=index):
                    temp_array.append([])
                temp_array[index].extend(actdata_dict[unamekey][catnamekey][index])
    return np.array(temp_array)

##dicを更新
##各クラスタに所属するデータを追加
def update_userdata(dic,data):
    climbCategory = data['climbCategory']
    if climbCategory not in dic.keys():
        dic[climbCategory] = []               
    dic[climbCategory].append(data)

##dicを更新
##各クラスタに所属するデータを追加
def check_userdata(data):        
    keys = ['userid','km_label','gm_label','GRADE','ALTITUDE','DISTANCE','VELOCITY']
    temp_dict = {}
    for key in keys:
##            キーが存在しない場合を省く
        if key not in data:
            return None
        temp_dict[key] = data[key]
    return temp_dict

##人数を指定してデータの取得
##make_newfile新規ユーザファイル作成フラグ
def preparation_data(label_str,target_label,person_num,min_datanum,make_newfile):
    ##    学習で使用するデータの準備
    middata_name = label_str+'_class'+target_label+'_personnum'+str(person_num)+'_datanum'+str(min_datanum)+'_userdata.json'
    filepath = funcs.get_filepath(middata_name,None)
    if make_newfile or os.path.exists(filepath) is False:
        print("Make Data!!!!")
    ##    同一ラベルを持つユーザの探索
        path_labeled = 'results\\data_for_regression\\20160203_011146\\labeled'
        files = os.listdir(path_labeled)
        random.shuffle(files)
        data_users = {}
        for file in files:
            if file in data_users.keys():
                continue
##            ひとつデータを読み込みラベルを確認
            data = funcs.read_myjson(os.path.join(path_labeled,file),1,False)
            for d in data:
##                ラベルが異なれば，次のファイルを読み込む
                if target_label != d[label_str]:
                    continue
##                ラベルがあっていれば，データの保存作業に入る
            data_users[file] = {}
##            ランダムで最大1000回分の走行データを得る
            data = funcs.read_myjson(os.path.join(path_labeled,file),200,False)
            for d in data:
    ##                データ点数のチェック
                ret_dict = check_userdata(d)
##                書き込めるデータが帰ってきたら
                if ret_dict is not None:
                    update_userdata(data_users[file],d)
##          全クライムカテゴリのデータがなければ
            if len(data_users[file].keys())<6:
                del data_users[file]
##            データ数が不足していたら
            elif check_datanum(data_users[file],min_datanum) is False:
                del data_users[file]
            if len(data_users)>=person_num:
                break

        if data_users is False:
            print("users is NULL!!!!!!!!!!!!!!!!")
            exit(-1)
        
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
        except:
            return preparation_data(label_str,target_label,person_num,min_datanum,True)
        finally:
            f.close()
    return data_users

##kmeansのラベルのとき
##SVR(C=3162.2776601683795, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
##  gamma=0.35938136638046259, kernel='rbf', max_iter=-1, shrinking=True,
##  tol=0.001, verbose=False)

def regresson_category(actdata_dict,keys,label):
##    カテゴリ毎にテスト
    climb_cat = ['FLAT', 'CATEGORY4',  'CATEGORY3', 'CATEGORY2' ,'CATEGORY1','HORS_CATEGORIE']
    print(actdata_dict.keys())
    models = []
    scores = []
    model_str = 'rf'

    f = funcs.get_fileobj(model_str+'.csv','w',model_str)
    writer = csv.writer(f)
    array_a = []
    array_p = []
    errors = []
    for category in climb_cat:
        print('-----------'+category+'-------------')
        darray = make_dataarray(actdata_dict,keys,category)
    ##        k-foldはXを説明変数，yを目的変数とする
        X = darray[:3].transpose()
        y = darray[3]
##        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
        scaler_exp = StandardScaler().fit(X)
        X = scaler_exp.transform(X)
        scaler_ret = StandardScaler().fit(y)
        y = scaler_ret.transform(y)
        joblib.dump(scaler_exp, funcs.get_filepath('exp_'+category+'_'+model_str+'_'+label+'.pkl',os.path.join('scaler',model_str)))
        joblib.dump(scaler_ret, funcs.get_filepath('ret_'+category+'_'+model_str+'_'+label+'.pkl',os.path.join('scaler',model_str)))
##        model = tuneSVM(X,y)
##        kf = KFold(len(y),n_folds=10)        
##        result = [[],[],[]]
##        for train_index, test_index in kf:
##            X_train, X_test = X[train_index], X[test_index]
##            y_train, y_test = y[train_index], y[test_index]
##            model.fit(X_train,y_train)
##            pred = model.predict(X_test)
##            result[0].append(model.score(y_test,pred))
##            result[1].append(explained_variance_score(y_test,pred))
##            result[2].append(mean_absolute_error(y_test,pred))
        
        train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        X,y,test_size=0.5,random_state=1)
        if model_str in 'nn': 
            model = MLPRegressor(hidden_layer_sizes=83,alpha=0.1)
        elif model_str in 'rf': 
            model = RandomForestRegressor(max_features='log2')
        elif model_str in 'svm': 
           model = PassiveAggressiveRegressor()
        else:
            print('model_str', model_str,'ERROR!!!!')
            exit(-1)
            
        model.fit(train_data,train_label)
        joblib.dump(model, funcs.get_filepath(category+'_'+model_str+'_'+label+'.pkl',os.path.join('model',model_str)))
        pred = model.predict(test_data)
        models.append(model)
        print(r2_score(test_label,pred))
        writer.writerow(np.array(test_label).transpose())
        writer.writerow(np.array(pred).transpose())
        array_a.append(np.array(test_label))
        array_p.append(np.array(pred))
##        print("r2:"+str(np.mean(result[0])))
##        print("evs:"+str(np.mean(result[1])))
##        print("mae:"+str(np.mean(result[2])))
        print('-----------------------------------')
    print(np.array(scores))

##    図の描画
    fig = plt.figure()
    for i in range(6):
        fig.add_subplot(2,3,i+1)
        plt.plot(array_a[i][:30], 'r')
        plt.plot(array_p[i][:30], 'k--')
        plt.title(climb_cat[i])
    plt.tight_layout()
    plt.savefig(funcs.get_filepath('image_'+model_str+'_'+label+'.png',model_str))
    f.close()
    return models

##def regresson_category(actdata_dict,keys,label):
####    カテゴリ毎にテスト
##    climb_cat = ['FLAT', 'CATEGORY4',  'CATEGORY3', 'CATEGORY2' ,'CATEGORY1','HORS_CATEGORIE']
##    print(actdata_dict.keys())
##    models = []
##    scores = []
##    
##    array_a = []
##    array_p_nn = []
##    array_p_rf = []
##    for category in climb_cat:
##        print('-----------'+category+'-------------')
##        darray = make_dataarray(actdata_dict,keys,category)
##    ##        k-foldはXを説明変数，yを目的変数とする
##        X = scale(darray[:3].transpose())
##        y = scale(darray[3].transpose())
##        
##        train_data,test_data,train_label,test_label = cross_validation.train_test_split(
##        X,y,test_size=0.5,random_state=1)
##        model = MLPRegressor(hidden_layer_sizes=83,alpha=0.1)
##        model.fit(train_data,train_label)
##        pred = model.predict(test_data)
##        array_a.append(np.array(test_label))
##        array_p_nn.append(np.array(pred))
##        
##        model = RandomForestRegressor(max_features='log2')
##        model.fit(train_data,train_label)
##        pred = model.predict(test_data)
##        array_a.append(np.array(test_label))
##        array_p_rf.append(np.array(pred))
##        print('-----------------------------------')
##
####    図の描画
##    model_str = 'nn'
##    fig = plt.figure()
##    for i in range(6):
##        fig.add_subplot(2,3,i+1)
##        plt.plot(array_a[i][:30], 'k')
##        plt.plot(array_p_nn[i][:30], 'r--')
##        plt.title(climb_cat[i])
##    plt.tight_layout()
##    plt.savefig(funcs.get_filepath('image_'+model_str+'_'+label+'.png',model_str))
##
##    model_str = 'rf'
##    fig = plt.figure()
##    for i in range(6):
##        fig.add_subplot(2,3,i+1)
##        plt.plot(array_a[i][:30], 'k')
##        plt.plot(array_p_rf[i][:30], 'b--')
##        plt.title(climb_cat[i])
##    plt.tight_layout()
##    plt.savefig(funcs.get_filepath('image_'+model_str+'_'+label+'.png',model_str))
##
####    model_str = 'both'
####    fig = plt.figure()
####    for i in range(6):
####        fig.add_subplot(2,3,i+1)
####        plt.plot(array_a[i][:30], 'k')
####        plt.plot(array_p_nn[i][:30], 'r--')
####        plt.plot(array_p_rf[i][:30], 'b--')
####        plt.title(climb_cat[i])
####    plt.tight_layout()
####    plt.savefig(funcs.get_filepath('image_'+model_str+'_'+label+'.png',model_str))
##    
##    return models

##ラベルが同じで学習に使用していないファイル名を一つ取得
def get_a_data(fnames,label,label_str):
    path_labeled = 'results\\data_for_regression\\20160203_011146\\labeled'
    files = os.listdir(path_labeled)
    random.shuffle(files)
    data_users = {}
    for file in files:
        if file in data_users.keys():
            continue
        ##            ひとつデータを読み込みラベルを確認
        data = funcs.read_myjson(os.path.join(path_labeled,file),1,False)
        for d in data:
    ##                ラベルが一致し，ファイル内になければ
            if label == d[label_str]:
                if file not in fnames:
                        return d

##学習に使用したデータセット以外のテストデータが欲しいとき
def a_data_to_testdata(data,keys):
    temp_array = []
    for key in keys:
        index = keys.index(key)
        if(len(temp_array)<=index):
            temp_array.append([])
        temp_array[index].extend(data[key])
    return np.array(temp_array)
        

##[climbCategory][label][data]
##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    starttime = funcs.start_program()
    np.random.seed(1)
    random.seed(1)
    
##    クラスタリング手法とラベルに基づいて最大person_num人分のデータを取得
    km_range = np.array(range(9)).astype(np.str)
    gm_range = np.array(range(5)).astype(np.str)
    fig_list = []
    for label in gm_range:
        data_users=[]
        label_str = 'km_label'
        label_str = 'gm_label'
        person_num = 30
        ##データ点数を指定
        min_datanum = 10000
        data_users = preparation_data(label_str,label,person_num,min_datanum,False)

        ##取得対象のデータ
        keys = ['GRADE','DISTANCE','ALTITUDE','VELOCITY']
        ##最低データ点数を指定
        keyss = data_users.keys()
        actdata_dict = get_data_eachuser(data_users,keys,min_datanum)
        align_data_eachuser(actdata_dict,keys,min_datanum)
        models = regresson_category(actdata_dict,keys,label)

####       モデル作った後のやつ
##        for_pred = get_a_data(list(actdata_dict.keys()),label,label_str)
##
##        d = a_data_to_testdata(for_pred ,keys )
##        X = scale(d[:3].transpose())
##        y = scale(d[3].transpose())
####        pred = model.predict(X)
####        print(r2_score(y,pred))
    funcs.end_program()
