# encoding: utf-8
import time
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

##ファイル名がjsonかどうか確認する
def is_json(file):
    root,ext = os.path.splitext(file)
    if ext == ".json":
        return file
    return None

##与えられたpath以下のjsonファイルのパスを取得
def get_filepath(path,working_path):
    return os.path.join(working_path,path), os.listdir(os.path.join(working_path,path))

##与えられたパスからjsonデータの取得
def get_data(path):
    data = funcs.decode_json(path)
    if data:
        effort_data = data['segmentEffortData']
##        記録だけして走行していない場合
        if effort_data['distance'] < 5:
            return None
        stream_data = data['segmentStreamData']
        temp = {'userid':effort_data['athlete']['id'],
                'segmentid':effort_data['segment']['id'],
                'climbCategory':effort_data['segment']['climbCategory'],
                'startDate':effort_data['startDate']
                }
        for data in stream_data:
            datastr = data['type']
            if datastr == 'VELOCITY':
                temp[datastr] = data['data']
                temp['averageSpeed'] = np.mean(np.array(data['data']))
            if datastr == 'DISTANCE':
                temp[datastr] = data['data']
            if datastr == 'GRADE':
                temp[datastr] = data['data']
            if datastr == 'ALTITUDE':
                temp[datastr] = data['data']
            if datastr == 'MAPPOINT':
                temp[datastr] = data['mapPoints']
            if datastr == 'HEARTRATE':
                temp[datastr] = data['data']
            if datastr == 'CADENCE':
                temp[datastr] = data['data']
        return temp
    return None

##与えられたパスからユーザidを取得
def get_userid(path):
    data = funcs.decode_json(path)
    if data:
        return data['segmentEffortData']['athlete']['id']
    return None

##与えられたパスからjsonデータを読み込み走行速度の平均をとる
##使ってない
def get_avespeed(path):
    data = funcs.decode_json(path)
    if data:
        effort_data = data['segmentEffortData']
        stream_data = data['segmentStreamData']
        for data in stream_data:
            datastr = data['type']
            if datastr == 'VELOCITY':
                return(np.mean(np.array(data['data'])))
    return None

##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
    starttime = funcs.start_program()
    np.random.seed(0)
    df = pd.read_csv('results\\labels.csv')
    colnames = df.columns
    labeled_users = list(df[colnames[0]])
    km_label = list(df[colnames[1]])
    gmm_label = list(df[colnames[2]])
    try:
        working_path = 'F:\\study\\strava\\finished'
        save_path = 'results\\data_for_regression\\'+starttime.strftime("%Y%m%d_%H%M%S")
        dirs = os.listdir(working_path)
        dirsnum = len(dirs)
        write_counter = 0
        for d in dirs:
            path,filepaths=get_filepath(d,working_path)
            for filepath in filepaths:
                file = is_json(filepath)
                if file is None:
                    continue
                data = get_data(os.path.join(path,file))
                if data is None:
                    continue
                uid = data['userid']
                ##                ラベルがあればlabeledディレクトリへ，なければnotlabeledへ
                save_to = ''
                if uid in labeled_users:
                    save_to = os.path.join(save_path,'labeled')
                    index = labeled_users.index(uid)
                    data['km_label']=str(km_label[index])
                    data['gm_label']=str(gmm_label[index])
                else:
                    save_to = os.path.join(save_path,'notlabeled')

##                データをユーザごとにファイルに分割して保存
                funcs.write_myjsonfile(os.path.join(save_to,str(uid)+'.myjson'),data)
                write_counter+=1
                if write_counter % 1000 == 0:
                    endtime = datetime.now() - starttime
                    print(str(write_counter)+'/'+str(endtime))
    finally:
        pass
    funcs.end_program()

##    data = json.load(open('results\\data_for_regression\\20160202_233521\\0.json'))
##    if isinstance(data, dict):
##        data = [data]
##    for d in data:
##        pass
