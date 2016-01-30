# encoding: utf-8
import csv
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_score, recall_score
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
import multiprocessing
import functools
import funcs

##辞書からkmeansできるリストへ
def dicttolist_withkeys(dict_data,keys):
    userlist = []
    for data in dict_data:
        temp = []
        for key in keys:
            if key.find("Speed") == 1:
                temp.append(float(data[key]))
            else:
                temp.append(data[key])
        userlist.append(temp)
    return userlist

##    得たラベルからユーザタイプ分類のためのNNの学習
##    inputとtestデータは元データからそれぞれ半分ずつ
def learn_nn(features,labels):
    half = int(len(features) / 2)
    training_data = features[:half]
    training_label = labels[:half]
    test_data = features[half:]
    test_label = labels[half:]
    clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,9), random_state=1)
    clf.fit(training_data, training_label)
    predicted_label = clf.predict(test_data)
    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
    scores = cross_validation.cross_val_score(clf,features,labels,cv=10)
    print((scores.mean(), scores.std() * 2))
    target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']
    print(classification_report(test_label, predicted_label, target_names=target_names,digits = 3))

##    得たラベルから速度計算のためのRandomForestの学習
def learn_rf(features,labels,keys):
    number = len(features)
    half = int(number / 2)
##    np.cで行列を横向きに結合
##    training_data = np.c_[features[:half,1:len(keys)-2].astype('float'),labels[:half]]
    training_data = features[:half,1:len(keys)-2].astype('float')
    training_label = features[:half,0]
##    test_data = np.c_[features[number-half:,1:len(keys)-2].astype('float'),labels[number-half:]]
    test_data = features[number-half:,1:len(keys)-2].astype('float')
    test_label = features[number-half:,0]
    model = RandomForestClassifier(n_estimators=250)
    model.fit(training_data,training_label)
    predicted_label = model.predict(test_data)
##    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
##    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
##各木で出た数値の平均を取れないか？
    difference = []
    for d in np.c_[test_label,predicted_label]:
        difference.append(abs(float(d[0])-float(d[1])))
    print(max(difference)*3.6)
    print(min(difference)*3.6)
    print(sum(difference)/len(difference)*3.6)
    print(sum(difference)/len(difference)*60)

##重回帰分析
def learn_ra(all_data):
    features = all_data[:,1:].astype('float')
    number = len(features)
    half = int(number / 2)

    training_data = features[:half,1:].astype('float')
    training_label = features[:half,0].astype('float')
##    test_data = np.c_[features[number-half:,1:len(keys)-2].astype('float'),labels[number-half:]]
    test_data = features[number-half:,1:].astype('float')
    test_label = features[number-half:,0]
##    model = LinearRegression()
##    バギングすればより良い結果がでるらしい。
##    clf = BaggingClassifier(base_estimator=clf, n_estimators=100, max_samples=0.9, max_features=0.2,n_jobs=4)
    model = MLPRegressor(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,9), random_state=1)
    model.fit(training_data,training_label)
    predicted_label = model.predict(test_data)
    difference = []
    for d in np.c_[test_label,predicted_label]:
        difference.append(abs(float(d[0])-float(d[1])))
    print(max(difference)*3.6)
    print(min(difference)*3.6)
    print(sum(difference)/len(difference)*3.6)
    print(sum(difference)/len(difference)*60)

##listにuidが存在するか確認
def find_userid(datalist,uid):
    for data in datalist:
        if uid == data[0]:
            return True
    return False

##指定されたインデックスのデータをユーザごとに平均をとる
##def sum_userdata(all_data,indexes):
##    userdata = []
##    for data in all_data:
##        if find_userid(userdata,data[0]):

##各列毎の平均値をとる
def get_average(tup):
    uid = tup[0]
    data = tup[1]
    ##        np.nansumでnanを無視した合計を取ってくれる便利なヤツ
    return uid, np.nansum((np.array(data['data']).astype(np.float))[:,1:4],axis=0)/len(data['data']),len(data['climbCategory'])

##keyに基づいてdicからデータを取得
##keyのなかでデータがないものがあれば省く
def make_onedata(dic,keys):
    data = []
    for key in keys:
        d = dic[key]
##        データが存在しない場合無視するためにNoneを代入
        if d == '-1':
            d = None
        data.append(d)
    return data,dic['userid'],dic['climbCategory']

##climbCategory毎の回数を記録
##pythonは引数で与えられた変数を直接弄る？
##dictだからメモリが引数として与えられている？
def update_climbCategory(dic,climbCategory):
    if climbCategory in dic.keys():
        dic[climbCategory] += 1
    else:
        dic[climbCategory] = 1
    return dic

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    funcs.start_program()
    p = multiprocessing.Pool()
    p.daemon = True
    f = open('F:\\study\\analiser\\results\\user_data_all20160131022824.csv', 'r')
    reader = csv.DictReader(f)
    all_data = {}
##    keys通りの並びでデータが帰ってくる
    keys = ['averageSpeed','averageSpeedNeggrade','averageSpeedNograde','averageSpeedPosgrade','distance','averageGrade']
    try:
        for data,userid,climbCategory in p.imap_unordered(functools.partial(make_onedata,keys=keys),reader):
##            データがおかしかったり抜けがある場合
            if data is None:
                continue
####            5km以上のセグメントのみを取り扱う
##            if float(data[len(keys)-2]) < 5000:
##                continue
            if userid not in all_data.keys():
                all_data[userid] = {'data':[],'climbCategory':{}}
            all_data[userid]['data'].append(data)
            update_climbCategory(all_data[userid]['climbCategory'],climbCategory)
##        各列ごとのデータの平均をとる
        uids = []
        data_forkm = []
        for uid,data,climbCategory in p.imap_unordered(get_average,all_data.items()):
##            全カテゴリの走行データがある場合
            if climbCategory == 6:
                uids.append(uid)
                data_forkm.append(data)
        ##featuresからuseridとaveragespeedとlikecatを除いた値をkmeansに
        ##除去した後明示的にstrからfloatへ変換
        ##arrayのスライスhttp://d.hatena.ne.jp/y_n_c/20091117/1258423212
        kmeans_model = KMeans(n_clusters=9, random_state=1).fit(data_forkm)
        labels = kmeans_model.labels_
        learn_nn(data_forkm,labels)
##        learn_rf(features,labels,keys)
##        
##        learn_ra(features)
    finally:
        p.close()
        f.close()
    funcs.end_program()
