# encoding: utf-8
import csv
import json
import os
import sys
import multiprocessing
import winsound
import re
import funcs
from datetime import datetime

##各ユーザの['averageSpeedPosgrade','averageSpeedNeggrade','averageSpeedNograde','averageSpeed']の平均を取得
def get_userdata(path):
    try:
        dirs = os.listdir(path)
        averageSpeeds = {}
        categoryNum = {}
        keys = ['averageSpeedPosgrade','averageSpeedNeggrade','averageSpeedNograde','averageSpeed']
    ##    averagespeedsの初期化
        for key in keys:
            averageSpeeds[key] = []
            
        for d in dirs:
            files = funcs.get_abspath(os.path.join(path,d))
            for file in files:
                root,ext = os.path.splitext(file)
    ##            ファイルがjsonなら
                if funcs.is_json(ext):
                    data = funcs.decode_json(file)
    ##                jsonファイルの内容があれば
                    if data:
    ##                    ['averageSpeedPosgrade','averageSpeedNeggrade','averageSpeedNograde','averageSpeed']の内容を取得
                        for key in keys:
                            if data[key] != -1:
                                averageSpeeds[key].append(data[key])
            categoryNum[d] = len(files)
        ret_dict = {'userid' : data['userid']}
    ##    ['averageSpeedPosgrade','averageSpeedNeggrade','averageSpeedNograde','averageSpeed']の平均をとる
        for key in keys:
            ret_dict[key] = sum(averageSpeeds[key])/len(averageSpeeds[key])
        ret_dict['likecategory'] = funcs.get_maxkey_dict(categoryNum)
##        print('finished:'+path)
        return ret_dict
    except Exception as e:
        print("Error:"+path)
        print(e)

##全クライムカテゴリの走行データがあるユーザのディレクトリを取得
def check_have_allcategories(d):
    dirnum = len(os.listdir(d))
    if dirnum == 6:
        return d
    
##--------------プログラム開始------------------
##マルチスレッド処理の為にこれが必要 http://matsulib.hatenablog.jp/entry/2014/06/19/001050
if __name__ == '__main__':
##    working_path = 'F:\\study\\strava\\finished\\'
##    dirs = os.listdir(working_path)
##    dirs = add_workingpath(dirs,working_path)
##    p = multiprocessing.Pool()
##    dirs = remove_none(p.map(get_dirs, dirs))
##    results = remove_none(p.map(get_onedata,dirs))
##    keys = ['segmentid','name','distance','file_num']
##    write_csv(".\\results\\"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",results,keys)
##    print("finished")
 
##get_with_userdataな場合
    funcs.start_program()
    working_path = 'F:\\study\\analiser\\results\\data_for_each_category_4\\'
    dirs = os.listdir(working_path)
    dirs = funcs.add_workingpath(dirs,working_path)
    p = multiprocessing.Pool()
    p.daemon = True
    dirs = funcs.remove_none(p.map(check_have_allcategories,dirs))
    users = funcs.remove_none(p.map(get_userdata,dirs))
    p.close()
    keys = users[0].keys()
    funcs.write_csv('results\\type_data'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv',users,keys)
    funcs.end_program()
