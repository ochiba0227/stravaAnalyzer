# encoding: utf-8
import csv
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix,r2_score
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
import multiprocessing
import functools
import funcs

##グリッドサーチの結果をファイルに出力
def write_gridres(name,data):
    csvfile = open(name+".csv","w")
    writer = csv.writer(csvfile)
    writer.writerows(data.grid_scores_)
    csvfile.close()

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

##入力データと入力ラベルに基づいたニューラルネットワークの分類器のチューニング
def tune_nn_classifier(features,labels):
    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,7))),
                   'hidden_layer_sizes':list(range(50,150))}]
    model = MLPClassifier(activation='logistic',algorithm='l-bfgs', random_state=1)
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=3)
    clf.fit(train_data,train_label)
    print(clf.best_estimator_)
    predicted_label = clf.predict(test_data)
    print(classification_report(test_label, predicted_label,digits = 3))
    return clf

##入力データと入力ラベルに基づいてモデルに最適なパラメータを探索
def tune_model(features,labels,model,parameters):
    train_data,test_data,train_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=3)
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
    clf = GridSearchCV(model, parameters, cv=5, n_jobs=3)
    clf.fit(train_data,train_label)
    print(clf.best_estimator_)
    return clf

##    得たラベルからユーザタイプ分類のためのNNの学習
##    inputとtestデータは元データからそれぞれ半分ずつ
def learn_nn(features,labels):
    ##    トレーニングデータとテストデータを半分に割った場合
    training_data,test_data,training_label,test_label = cross_validation.train_test_split(
        features,labels,test_size=0.5,random_state=1)
    clf = MLPClassifier(algorithm='l-bfgs', alpha=0.1, hidden_layer_sizes=83, random_state=1)
    clf.fit(training_data, training_label)
    predicted_label = clf.predict(test_data)
    print("precision:"+str(precision_score(test_label, predicted_label, average='binary')))
    print("recall:"+str(recall_score(test_label, predicted_label, average='binary')))
    
    print(classification_report(test_label, predicted_label, digits = 3))
    print(confusion_matrix(test_label, predicted_label))
##    クロスバリデーション
    scores = cross_validation.cross_val_score(clf,features,labels,cv=10)
    print(scores)
    print((scores.mean(), scores.std() * 2))
    preds = cross_validation.cross_val_predict(clf,features,labels,cv=10)
    print(preds)
    print((preds.mean(), preds.std() * 2))

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
    speed_array = np.array(data['data']).astype(np.float)[:,1:5]
##    nan要素をマスキングで計算に現れずにする
    masked_array = np.ma.masked_array(speed_array,np.isnan(speed_array))
    speed = np.mean(masked_array,axis=0)
    ##        np.nansumでnanを無視した合計を取ってくれる便利なヤツあり
##    maskした後mean取ると--になっているところを無視して平均取ってくれる
    return uid, speed,len(data['climbCategory'])

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

##ユーザデータにラベルを付与
##現在は欠損値があるデータを削除している
def add_labels(tup,uids,labels):
    uid = tup[0]
    data = tup[1]

    if uid not in uids:
        return None
    data = np.array(data['data']).astype(np.float)
##    欠損値を削除
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) == 0:
        print(uid)
        return None
    label = np.array([labels[uids.index(uid)]]*len(data))
    return np.c_[data,label]

##climbCategory毎の回数を記録
##pythonは引数で与えられた変数を直接弄る？
##dictだからメモリが引数として与えられている？
def update_climbCategory(dic,climbCategory):
    if climbCategory in dic.keys():
        dic[climbCategory] += 1
    else:
        dic[climbCategory] = 1
    return dic

##dataforkmからaveragespeed_gradeの内容だけを持ち、ほかはダミーなcsvを作成
def output_file_forkm(data_forkm,uids):
    diclist = []
    for index in range(len(data_forkm)):
        dic = {'averageSpeedNeggrade':data_forkm[index][0],
               'averageSpeedPosgrade':data_forkm[index][2],
               'likecategory':"FF",
               'averageSpeed':10000,
               'userid':uids[index],
               'averageSpeedNograde':data_forkm[index][1]} 
        diclist.append(dic)
        
    keys = list(diclist[0].keys())
    header = dict([(val,val)for val in keys])
    diclist.insert(0,header)
    f = open('F:\\study\\analiser\\results\\typetetetete.csv', 'w')
    writer = csv.DictWriter(f, keys, lineterminator="\n")
    writer.writerows(diclist)
    f.close()

##クラスタごとに色分けしてプロット
def plot_data_3D(data,classes):
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(data[:,0], data[:,1], data[:,2], c=classes.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Negative Grade')
    ax.set_ylabel('No Grade')
    ax.set_zlabel('Positive Grade')
    plt.show()

##クラスタごとに色分けしてプロット
def plot_data_2D(data,classes):
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(data[:,0], data[:,1], data[:,2], c=classes.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Negative Grade')
    ax.set_ylabel('No Grade')
    ax.set_zlabel('Positive Grade')
    plt.show()  

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    funcs.start_program()
    p = multiprocessing.Pool()
    p.daemon = True
    f = open('F:\\study\\analiser\\results\\user_data_all20160131_023720.csv', 'r')
    reader = csv.DictReader(f)
    for data in reader:
        print(data.keys())
        break
    all_data = {}
##    keys通りの並びでデータが帰ってくる
    keys = ['averageSpeed','averageSpeedNeggrade','averageSpeedNograde','averageSpeedPosgrade','startDistance','distance']
    try:
        for data,userid,climbCategory in p.imap_unordered(functools.partial(make_onedata,keys=keys),reader):
##            記録開始だけして動いていないデータを削除
            if float(data[len(keys)-1]) < 5:
                continue
            if userid not in all_data.keys():
                all_data[userid] = {'data':[],'climbCategory':{}}
            all_data[userid]['data'].append(data)
            update_climbCategory(all_data[userid]['climbCategory'],climbCategory)
##        各列ごとのデータの平均をとる
        uids = []
        data_forlabel = []
        for uid,data,climbCategory in p.imap_unordered(get_average,all_data.items()):
##            全部nanの行がある場合を排除
            if True in data.mask:
                continue
##            全カテゴリの走行データがある場合
            if climbCategory == 6:
                uids.append(uid)
                data_forlabel.append(data)
        ##featuresからuseridとaveragespeedとlikecatを除いた値をkmeansに
        ##除去した後明示的にstrからfloatへ変換
        ##arrayのスライスhttp://d.hatena.ne.jp/y_n_c/20091117/1258423212
##        後の処理をしやすくするためにnparrayに変換
        clusters = 9
        data_forlabel = np.array(data_forlabel)
        kmeans_model = KMeans(n_clusters=clusters, random_state=1).fit(scale(data_forlabel))
        labels = np.array(kmeans_model.labels_)
        print(kmeans_model.cluster_centers_)

##        クラス0のデータの平均と分散に基づいてデータの検索
        print(np.mean(data_forlabel[labels==0],axis=0))
        plot_data_3D(data_forlabel,labels)
##        datas = p.map(lambda x:(np.mean(data_forlabel[labels==x],axis=0),
##                                np.mean(data_forlabel[labels==x],axis=0)),range(clusters))
##        ##        チューニング用の関数
##        model = MLPClassifier(activation='logistic',algorithm='l-bfgs', random_state=1)
##        parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,5))),
##                   'hidden_layer_sizes':list(range(50,151))}]
##        tuned_nnclassifier = tune_model(data_forlabel,labels,model,parameters)

##        learn_nn(data_forlabel,labels)
##
####        各行にラベルを付与
##        data_forreg = []
##        for data in p.imap_unordered(functools.partial(add_labels,uids=uids,labels=labels),all_data.items()):
##            if data is None:
##                continue
##            data_forreg.extend(data)
##        data_forreg = np.array(data_forreg)
##        features = data_forreg[:,1:]
##        labels = data_forreg[:,0]
####        チューニング用の関数
##        model = MLPRegressor(activation='logistic',algorithm='adam', random_state=1)
##        parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,5))),
##                   'hidden_layer_sizes':list(range(50,150,10))}]
##        tuned_nnregressor = tune_model(features,labels,model,parameters)
##
##        model = RandomForestRegressor(max_features='log2', random_state=1)
####n_estimatorsは<50で探索すべき
##        parameters = [{'n_estimators':list(range(10,50,5))}]
##        tuned_rfregressor = tune_model(features,labels,model,parameters)
##
##        model = RandomForestRegressor(n_estimators=50,max_features='log2', random_state=1)
##        regression_rf(features,labels,model)
##        model = MLPRegressor(activation='logistic',algorithm='adam', random_state=1,alpha=0.0001,hidden_layer_sizes=90)
##        regression_nn(features,labels,model)
##
####        距離，平均勾配，ユーザタイプから平均速度が予測できるか
##        features = data_forreg[:,4:]
####        チューニング用の関数
##        model = MLPRegressor(activation='logistic',algorithm='adam', random_state=1)
##        parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,5))),
##                   'hidden_layer_sizes':list(range(50,150,10))}]
##        tuned_nnregressor2 = tune_model(features,labels,model,parameters)
##
##        model = RandomForestRegressor(max_features='log2', random_state=1)
####n_estimatorsは<50で探索すべき
##        parameters = [{'n_estimators':list(range(10,50,5))}]
##        tuned_rfregressor2 = tune_model(features,labels,model,parameters)
##        
##        model = RandomForestRegressor(n_estimators=40,max_features='log2', random_state=1)
##        regression_rf(features,labels,model)
##        model = MLPRegressor(activation='logistic',algorithm='adam', random_state=1,alpha=0.01,hidden_layer_sizes=90)
##        regression_nn(features,labels,model)
##        
####        勾配に対する平均速度から平均速度が予測できるか
##        features = data_forreg[:,1:4]
##        ##        チューニング用の関数
##        model = MLPRegressor(activation='logistic',algorithm='adam', random_state=1)
##        parameters = [{'alpha':list(map(lambda x:1/(10**x),range(1,5))),
##                   'hidden_layer_sizes':list(range(50,150,10))}]
##        tuned_nnregressor3 = tune_model(features,labels,model,parameters)
##
##        model = RandomForestRegressor(max_features='log2', random_state=1)
####n_estimatorsは<50で探索すべき
##        parameters = [{'n_estimators':list(range(10,50,5))}]
##        tuned_rfregressor3 = tune_model(features,labels,model,parameters)
##
##        model = RandomForestRegressor(n_estimators=40,max_features='log2', random_state=1)
##        regression_rf(features,labels,model)
##        model = MLPRegressor(activation='logistic',algorithm='adam', random_state=1,alpha=0.01,hidden_layer_sizes=90)
##        regression_nn(features,labels,model)
##        
##        learn_ra(features)
##    except Exception as e:
##        print('error:'+str(e))
##        p.terminate()
    finally:
        p.close()
        f.close()
    funcs.end_program()

##a = np.array([[2,None,2],[2,2,2]]).astype(np.float)
##print(a)
##print(np.nansum(a,axis=0)/2)
##print(a.shape)
##print(np.isnan(a))
##mdat = np.ma.masked_array(a,np.isnan(a))
##print(mdat)
##mm = np.mean(mdat,axis=0)
##print(mm)
