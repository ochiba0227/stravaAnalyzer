# encoding: utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

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

##入力データと入力ラベルに基づいたニューラルネットワークのチューニング
def tune_nn(features,labels):
    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,7))),
                   'hidden_layer_sizes':list(range(50,150))}]
    nn = MLPClassifier(activation='logistic',algorithm='l-bfgs', random_state=1)
    clf = GridSearchCV(nn, parameters, cv=5, n_jobs=-1)
    clf.fit(train_data,train_label)
    print(clf.best_estimator_)
    predicted_label = clf.predict(test_data)
    print(classification_report(test_label, predicted_label,digits = 3))
    
    
##    得たラベルからユーザタイプ分類のためのNNの学習
##    inputとtestデータは元データからそれぞれ半分ずつ
##    除去した後明示的にstrからfloatへ変換
def learn_nn(features,labels,keys):
    tune_nn(features[:,1:len(keys)-2].astype('float'),labels)
##    number = len(features)
##    half = int(number / 2)
##    input_data = features[:half,1:len(keys)-2].astype('float')
##    input_label = labels[:half]
##    test_data = features[number-half:,1:len(keys)-2].astype('float')
##    test_label = labels[number-half:]
####    alphaはペナルティ項
##    clf = MLPClassifier(activation='logistic',algorithm='l-bfgs', alpha=0.1, hidden_layer_sizes=99, random_state=1)
##    clf.fit(input_data, input_label)
##    for coef in clf.coefs_:
##        print(coef.shape)
##    for intercept in clf.intercepts_:
##        print(intercept.shape)
##    predicted_label = clf.predict(test_data)
##    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
##    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
##    scores = cross_validation.cross_val_score(clf,features[:,1:len(keys)-2].astype('float'),labels,cv=10)
##    print((scores.mean(), scores.std() * 2))
##    target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']
##    print(classification_report(test_label, predicted_label, target_names=target_names,digits = 3))

##    得たラベルから速度計算のためのRandomForestの学習
def learn_rf(features,labels,keys):
    number = len(features)
    half = int(number / 2)
##    np.cで行列を横向きに結合
##    input_data = np.c_[features[:half,1:len(keys)-2].astype('float'),labels[:half]]
    input_data = features[:half,1:len(keys)-2].astype('float')
    input_label = features[:half,0]
##    test_data = np.c_[features[number-half:,1:len(keys)-2].astype('float'),labels[number-half:]]
    test_data = features[number-half:,1:len(keys)-2].astype('float')
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
    print(max(difference)*3.6)
    print(min(difference)*3.6)
    print(sum(difference)/len(difference)*3.6)
    print(sum(difference)/len(difference)*60)

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    f = open('results\\typetetetete.csv', 'r')
    reader = csv.DictReader(f)
    got_dict = list(reader)
    f.close()

    keys = np.sort(list(got_dict[0].keys()))
    userlist = dicttolist_withkeys(got_dict,keys)
    features = np.array(userlist)

    ##featuresからuseridとaveragespeedとlikecatを除いた値をkmeansに
##    除去した後明示的にstrからfloatへ変換
    ##arrayのスライスhttp://d.hatena.ne.jp/y_n_c/20091117/1258423212
    all_data = features[:,1:len(keys)-2].astype(np.float)
    kmeans_model = KMeans(n_clusters=9, random_state=1)
    kmeans_model.fit(all_data)
    labels = kmeans_model.labels_
    centers = kmeans_model.cluster_centers_ 
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(all_data[:,0], all_data[:,1], all_data[:,2], c=labels.astype(np.float))
##    ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=np.array(range(len(centers))).astype(np.float), s = 1000)
    index = 7
    print(centers[index])
    ax.scatter(centers[index,0], centers[index,1], centers[index,2], c='#eeeeee', s = 1000)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Negative Grade')
    ax.set_ylabel('No Grade')
    ax.set_zlabel('Positive Grade')
    plt.show()
##    learn_nn(features,labels,keys)
##    learn_rf(features,labels,keys)

