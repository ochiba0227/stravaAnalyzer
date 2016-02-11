# encoding: utf-8
import csv
import numpy as np
import os
import pandas as pd
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix,r2_score,euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
from sklearn import mixture
from x_means import XMeans
import multiprocessing
import functools
import funcs
import make_movie
from sklearn.externals import joblib

##listにuidが存在するか確認
def find_userid(datalist,uid):
    for data in datalist:
        if uid == data[0]:
            return True
    return False

##入力データと入力ラベルに基づいてMSに最適なパラメータを探索
def tune_MS(data):
    mss = []
    range_max = 20
    bandwidth_range = 0.35 + (np.array(range(0, range_max+1))-(range_max/2))*0.001
    for bandwidth in bandwidth_range:
        ms = MeanShift(bandwidth=bandwidth, n_jobs=-2)
        ms.fit(data)
        mss.append(ms)
        
    metrics=['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    distance_max = 0
    retms = mss[0]
    for metric in metrics:
        print(metric)
        print("-------------")
        for ms in mss:
            distance = pairwise_distances(ms.cluster_centers_,metric=metric,n_jobs=-2)
            print(np.mean(distance,axis=0))
            distance = np.mean(distance)
            if distance > distance_max:
                distance_max = distance
                retms = ms
        print("-------------")
    return retms

##入力データと入力ラベルに基づいてGMMに最適なパラメータを探索
def tune_GMM(data):
    lowest_bic = np.infty
    bic = np.array([['bic']], dtype=object)
    aic = np.array([['aic']], dtype=object)
    n_components_range = range(2, 50)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        bic = np.r_[bic,[[cv_type]]]
        aic = np.r_[aic,[[cv_type]]]
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, random_state=1)
            gmm.fit(data)
            bic = np.r_[bic,np.array([[gmm.bic(data)]])]
            aic = np.r_[aic,np.array([[gmm.aic(data)]])]
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    write_rows = np.c_[bic,aic]
    try:
        file = funcs.get_fileobj('gmm_result.csv','w',None)
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(write_rows)
    finally:
        file.close()
    return best_gmm

##各列毎の平均値をとる
def get_average(tup):
    uid = tup[0]
    data = tup[1]
    speed_array = np.array(data['data']).astype(np.float)
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

##クラスタごとに色分けしてアニメーション
##コマンドプロンプトから実行しないとmendoderにパスが通っていないから動かない
def animate_data_3D(data,classes):
    fig = plt.figure()
    fig.set_size_inches(800,800)
    fig.set_dpi(800)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=classes.astype(np.float))
    ax.set_axis_off()
##        ax.w_xaxis.set_ticklabels([])
##        ax.w_yaxis.set_ticklabels([])
##        ax.w_zaxis.set_ticklabels([])
##        ax.set_xlabel('Negative Grade')
##        ax.set_ylabel('No Grade')
##        ax.set_zlabel('Positive Grade')
    angles = np.linspace(0,720)[:-1]*0.5
##    make_movie.rotanimate(ax, angles,'F:\\study\\analiser\\movie.gif',delay=20)
    outpath = funcs.get_filepath('movie.mp4',None)
    make_movie.rotanimate(ax, angles,outpath,fps=10,bitrate=100000000)
##    anime = animation.ArtistAnimation(fig,axs, interval=1, repeat_delay=1000)
##    anime.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

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

##ユーザデータをファイルから作成
def makeuserdata_from_file(path):
    try:
        f = open(path, 'r')
        reader = csv.DictReader(f)
        all_data = {}
    ##    keys通りの並びでデータが帰ってくる
        keys = ['averageSpeed','averageSpeedNeggrade','averageSpeedNograde','averageSpeedPosgrade','startDistance','distance']
        for row in reader:
            data,userid,climbCategory = make_onedata(row,keys)
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
        for items in all_data.items():
            uid,data,climbCategory = get_average(items)
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
        file = funcs.get_fileobj('middata.csv','w',None)
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(np.r_[['uid'],keys])
        writer.writerows(np.c_[uids,data_forlabel])
        return(data_forlabel)
    finally:
        f.close()
    return None


##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    funcs.start_program()
    filepath = funcs.get_filepath('middata.csv',None)
    if os.path.exists(filepath) is False:
        print("DATA FROM FILE!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data = makeuserdata_from_file('F:\\study\\analiser\\results\\user_data_all20160131_023720.csv')
    else:
        df = pd.read_csv(filepath)
        data = np.array(df).astype(np.float)

##    クラスタリング開始
    uids = data[:,0]
    data = data[:,2:5]
    scaled_data = scale(data)
##kmeansでクラスタリング
    kmeans_model = KMeans(n_clusters=9, random_state=1)
    kmeans_model.fit(scale(scaled_data))
    km_labels = np.array(kmeans_model.labels_)
##        plot_data_2D(data,labels,components_label)

##GMMでクラスタリング
##        GMMのチューニング
##    gmm = tune_GMM(scaled_data)
    gmm = mixture.GMM(covariance_type='full', init_params='wmc', min_covar=0.001,
  n_components=5, n_init=1, n_iter=100, params='wmc', random_state=1,
  tol=0.001, verbose=0)
    gmm.fit(scaled_data)
    gm_labels = gmm.predict(scaled_data)
    joblib.dump(gmm, funcs.get_filepath('gmmpred.pkl','gmm_model'))
##    animate_data_3D(data,gm_labels)
##    plot_data_2D(scaled_data,gm_labels,['Negative Grade','No Grade','Positive Grade'])

##    ms_model = tune_MS(scaled_data)
    ms_model = MeanShift(bandwidth=0.35299999999999998)
    ms_model.fit(scaled_data)
    ms_labels = ms_model.labels_
    
    try:
        file = funcs.get_fileobj('labels.csv','w',None)
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['uid','9-means','GMM','MeanShift'])
        writer.writerows(np.c_[uids,km_labels,gm_labels,ms_labels])
    finally:
        file.close()
##        kmeansの重心のプロット
##        centers = kmeans_model.cluster_centers_
##        plot_data_2D(centers,np.array(range(cluster_num)),components_label)
    funcs.end_program()
