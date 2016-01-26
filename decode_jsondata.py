# encoding: utf-8
import csv
import json
import os
import sys
import multiprocessing
import winsound
from datetime import datetime

##ファイル名,データ配列,保存したいキー値
def write_csv(fname,data,keys):
    header = dict([(val,val)for val in keys])
     
    with open(fname, mode='w') as f:
        data.insert(0,header)
        writer = csv.DictWriter(f, keys, extrasaction='ignore', lineterminator="\n")
        writer.writerows(data)

def write_tofile(fname,data):
    with open(fname, mode='w') as f:
        for d in data:
            f.write(d+','+str(data[d])+'\n') # 引数の文字列をファイルに書き込む
        f.close() # ファイルを閉じる

##jsonのデコード
def decode_json(file):
    ##f = open("7417_132941303.json")
    ##weather_dict = json.loads(f)
    ##weather_format_json = json.dumps(weather_dict, indent=4, separators=(',', ': '))
    ##print(weather_format_json)
##    空のファイルを無視
    root,ext = os.path.splitext(file)
    if os.path.getsize(file) != 0 and is_json(ext):
        with open(file, 'r') as f:
            json_dic = json.load(f)
            effort_data = json_dic['segmentEffortData']
        ## ケイデンス等取得データは10種類ある。以下のように確認
    ##        print('CADENCE' in test[5].values())
    ##        距離は以下のように取得、単位はm
    ##        print(fenrifja_dic['segmentEffortData']['distance'])
            f.close()
    ##    segmentidキーを追加
        effort_data['segmentid']=effort_data['segment']['id']
        effort_data['userid']=effort_data['athlete']['id']
        return effort_data
        

##拡張子がjsonかどうか確認する
def is_json(ext):
    if ext == ".json":
        return True
    return False

##再帰的にディレクトリの取得
##def get_dirs_rec(path):
##    dirs=[]
##    for item in os.listdir(path):
##        item = os.path.join(path,item)
##        if os.path.isdir(item):
##            nextdir = get_dirs(item)
##            for gotdir in nextdir:
##                dirs.append(gotdir)
##            dirs.append(item)
##            print(item)
##    return dirs

##ディレクトリの取得
def get_dirs(path):
    if os.path.isdir(path):
        return path

##取得したディレクトリからjsonファイルを一つだけ取得し、セグメントの距離などをreturn
def get_onedata(path):
    files = os.listdir(path)
    for file in files:
        root,ext = os.path.splitext(file)
        if is_json(ext):
##            print(file)
            data = decode_json(os.path.join(path,file))
            if data:
                data['file_num'] = len(files)
            return data

##usersに存在しなければ追加、存在すれば出現回数を加算
def add_user(users,uid):
    if uid in users:
        users[uid] = users[uid] + 1
    else:
        users[uid] = 1
    return users

##取得したディレクトリからユーザ数を付加して取得
def get_with_userdata(path):
    files = os.listdir(path)
    users = {}
    data = {}
    for file in files:
        root,ext = os.path.splitext(file)
        if is_json(ext):
            data = decode_json(os.path.join(path,file))
            if data:
                users = add_user(users,data['athlete']['id'])
    if data:
        data['users'] = users
        data['file_num'] = len(files)
    return data

##listからnoneを省く
def remove_none(listdata):
    return [item for item in listdata if item is not None]

##ワーキングパスを付加する
def add_workingpath(listdata,path):
    return_data = []
    for data in listdata:
        return_data.append(path+data)
    return return_data

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
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
    print(datetime.now().strftime("%Y%m%d%H%M%S"))
    working_path = 'F:\\study\\strava\\finished\\1660077\\'
    dirs = os.listdir(working_path)
    dirs = add_workingpath(dirs,working_path)
    print(dirs[0])
    p = multiprocessing.Pool()
    results = remove_none(p.map(decode_json,dirs))
    p.close()

    season_count = {}
    time_count = {}
    season_and_time_count = {}
    segment_id = ""
    for result in results:
        month = result['startDateLocal']['date']['month']
        hour = result['startDateLocal']['time']['hour']
        month_hour = str(month) + "and" + str(hour)
        if month in season_count:
            season_count[month] += 1
        else:
            season_count[month] = 1
        if hour in time_count:
            time_count[hour] += 1
        else:
            time_count[hour] = 1
        if month_hour in season_and_time_count:
            season_and_time_count[month_hour] += 1
        else:
            season_and_time_count[month_hour] = 1
        segment_id = result['segmentid']
        
    print(season_count)
    print(time_count)
    print(season_and_time_count)
##    write_tofile(".\\results\\season_"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",season_and_time_count)
    f = open(".\\results\\season_"+str(segment_id)+".csv", 'w') # 書き込みモードで開く
    f.write('month\n')
    for d in season_count:
        f.write(str(d)+','+str(season_count[d])+'\n')
    f.write('hour\n')
    for d in time_count:
        f.write(str(d)+','+str(time_count[d])+'\n')
    f.write('month_and_hour\n')
    for d in season_and_time_count:
        f.write(str(d)+','+str(season_and_time_count[d])+'\n')
    f.close() # ファイルを閉じる
##    keys = ['segmentid','name','userid','startDate']
##    write_csv(".\\results\\season_"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",results,keys)
##    print("finished:"+datetime.now().strftime("%Y%m%d%H%M%S"))
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
