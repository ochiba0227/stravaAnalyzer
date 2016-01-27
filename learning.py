# encoding: utf-8
import csv
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
##from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score

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
    
##    得たラベルからNNの学習
##    除去した後明示的にstrからfloatへ変換
    number = len(features)
    half = int(number / 2)
    input_data = features[:half,1:len(keys)-2].astype('float')
    input_lavel = labels[:half]
    test_data = features[number-half:,1:len(keys)-2].astype('float')
    test_lavel = labels[number-half:]
    clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(9,9), random_state=1)
    clf.fit(input_data, input_lavel)
    predicted_lavel = clf.predict(test_data)
    print(precision_score(test_lavel, predicted_lavel, average='binary'))
##    testdata = [[3, 3 , 3, 3, 3, 3, 3]]
##
##    model = RandomForestClassifier(n_estimators=100)
##    model.fit(features[:,1:], labels)
##    output = model.predict(testdata)
##    print(len(model.estimators_))
##    for label in output: print(label)
