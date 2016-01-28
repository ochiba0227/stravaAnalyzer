# encoding: utf-8
import csv
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn import cross_validation
from sklearn.metrics import classification_report

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
##    除去した後明示的にstrからfloatへ変換
def learn_nn(features,labels,keys):
    number = len(features)
    half = int(number / 2)
    input_data = features[:half,1:len(keys)-2].astype('float')
    input_label = labels[:half]
    test_data = features[number-half:,1:len(keys)-2].astype('float')
    test_label = labels[number-half:]
    clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,9), random_state=1)
    clf.fit(input_data, input_label)
    predicted_label = clf.predict(test_data)
    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
    scores = cross_validation.cross_val_score(clf,features[:,1:len(keys)-2].astype('float'),labels,cv=10)
    print((scores.mean(), scores.std() * 2))
    target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']
    print(classification_report(test_label, predicted_label, target_names=target_names,digits = 3))

##    得たラベルから速度計算のためのRandomForestの学習
def learn_rf(features,labels,keys):
    number = len(features)
    half = int(number / 2)
##    np.cで行列を横向きに結合
    input_data = np.c_[features[:half,1:len(keys)-2].astype('float'),labels[:half]]
    input_label = features[:half,0]
    test_data = np.c_[features[number-half:,1:len(keys)-2].astype('float'),labels[number-half:]]
    test_label = features[number-half:,0]
    model = RandomForestClassifier(n_estimators=250)
    model.fit(input_data,input_label)
    predicted_label = model.predict(test_data)
##    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
##    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
##各木で出た数値の平均を取れないか？
    difference = []
    for d in np.c_[test_label,predicted_label]:
        difference.append(abs(float(d[0])-float(d[1])))
    print(sum(difference)/len(difference))
    print(sum(difference)/len(difference)*60)

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    f = open('F:\\study\\analiser\\results\\type_data.csv', 'r')
    reader = csv.DictReader(f)
    got_dict = list(reader)
    f.close()

    keys = np.sort(list(got_dict[0].keys()))
    userlist = dicttolist_withkeys(got_dict,keys)
    features = np.array(userlist)

    ##featuresからuseridとaveragespeedとlikecatを除いた値をkmeansに
##    除去した後明示的にstrからfloatへ変換
    ##arrayのスライスhttp://d.hatena.ne.jp/y_n_c/20091117/1258423212
    all_data = features[:,1:len(keys)-2].astype('float')
    kmeans_model = KMeans(n_clusters=9, random_state=1).fit(all_data)
    labels = kmeans_model.labels_
    
    learn_nn(features,labels,keys)
##    learn_rf(features,labels,keys)
