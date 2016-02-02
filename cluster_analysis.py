# encoding: utf-8
import csv
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix,r2_score,euclidean_distances
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
from sklearn import mixture
from x_means import XMeans
import multiprocessing
import functools
import funcs
from sklearn.externals import joblib

##listにuidが存在するか確認
def find_userid(datalist,uid):
    for data in datalist:
        if uid == data[0]:
            return True
    return False

##入力データと入力ラベルに基づいてGMMに最適なパラメータを探索
def tune_GMM(data):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, random_state=1)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm

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
def plot_data_2D(data,classes,components_label):
    fig = plt.figure()
    data_len = len(components_label)
    for x in range(data_len):
        for y in range(data_len):
            ax = fig.add_subplot(data_len,data_len,1+y+x*data_len)
            if x == y:
                ax.text(0,0,components_label[x])
            else:
                ax.scatter(data[:,x], data[:,y], c=classes.astype(np.float))
    plt.show()  

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    funcs.start_program()
    p = multiprocessing.Pool()
    p.daemon = True
    f = open('F:\\study\\analiser\\results\\user_data_all20160131_023720.csv', 'r')
    reader = csv.DictReader(f)
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
        data_forlabel = np.array(data_forlabel)

        cluster_num = 9
        components_label = ['averageSpeedNeggrade','averageSpeedNograde','averageSpeedPosgrade']
        data = scale(data_forlabel[:,:-1])

##kmeansでクラスタリング
        kmeans_model = KMeans(n_clusters=cluster_num, random_state=1).fit(data)
        km_labels = np.array(kmeans_model.labels_)
##        plot_data_2D(data,labels,components_label)

##GMMでクラスタリング
##        GMMのチューニング
        gmm = tune_GMM(data)
        gm_labels = gmm.predict(data)
        outfile = open('results\\labels.csv', 'w')
        writer = csv.writer(outfile, lineterminator='\n')
        writer.writerow(['uid','9-means','GMM'])
        writer.writerows(np.c_[uids,km_labels,gm_labels])
        outfile.close()
##        kmeansの重心のプロット
##        centers = kmeans_model.cluster_centers_
##        plot_data_2D(centers,np.array(range(cluster_num)),components_label)

    finally:
        p.close()
        f.close()
    funcs.end_program()
