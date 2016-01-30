# encoding: utf-8
import csv
import json
import os
import sys
import multiprocessing
import winsound
import re
from datetime import datetime
import numpy as np
import inspect

##ファイル名,データ配列,保存したいキー値
def write_csv(fname,data,keys):
    header = dict([(val,val)for val in keys])
     
    with open(fname, mode='w') as f:
        data.insert(0,header)
        writer = csv.DictWriter(f, keys, extrasaction='ignore', lineterminator="\n")
        writer.writerows(data)

##ファイルオブジェクト,データdict
##keysをソートしてから出力しないと、でたらめなファイルになる。
def write_csv_onedata(f,data):
    keys = sorted(list(data.keys()))
    if f.tell() == 0:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(keys)
    writer = csv.DictWriter(f, keys, extrasaction='ignore', lineterminator="\n")        
    writer.writerows([data])

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
        ## ケイデンス等取得データは10種類ある。以下のように確認
    ##        print('CADENCE' in test[5].values())
    ##        距離は以下のように取得、単位はm
    ##        print(fenrifja_dic['segmentEffortData']['distance'])
            f.close()
            return json_dic        

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

##ファイルの取得
def get_files(path):
    if os.path.isfile(path):
        return path

##辞書に鍵が存在するか確認
def compare_dictkeys(key,data):
    if key in data.keys():
        return data[key]
    return None

##mappointsの最初と最後を取得
def get_start_end_latlng(stream_data):
    for data in stream_data:
        if data['type'] == 'MAPPOINT':
            points = data['mapPoints']
            startlatlng = points[0]
            endlatlng = points[len(points)-1]
            return [startlatlng,endlatlng]
    return None

##listからnoneを省く
def remove_none(listdata):
    return [item for item in listdata if item is not None]

##ワーキングパスを付加する
def add_workingpath(listdata,path):
    return_data = []
    for data in listdata:
        return_data.append(path+data)
    return return_data

##指定されたパスに含まれるファイル，ディレクトリを絶対パスで取得
def get_abspath(path):
    curpath = os.getcwd()
    os.chdir(path)
    dirs = os.listdir(path)
    ret_dirs = []
    for d in dirs:
        ret_dirs.append(os.path.abspath(d))
    os.chdir(curpath)
    return ret_dirs

##ログファイルの書き込み
def write_log(text):
    # 追記モードで出力
    f = open( 'runlog.txt', 'a' )
    try:
        f.write(text+'\n')
    finally:
        f.close()

##プログラムの開始時SE
def start_program():
    winsound.PlaySound('se\\se_moa01.wav',winsound.SND_FILENAME)
    filename = inspect.currentframe().f_back.f_code.co_filename
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = 'start:'+filename+' on:'+time
    print(line)
    write_log(line)

##プログラムの終了時SE
def end_program():
    winsound.PlaySound('se\\se_moc07.wav',winsound.SND_FILENAME)
    filename = inspect.currentframe().f_back.f_code.co_filename
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = 'end:'+filename+' on:'+time
    print(line)
    write_log(line)

##ディレクトリの作成
def make_dir(path):
    try:
        os.makedirs(path)
    except:
        pass

##辞書から最大値を持つキーを返す
##dict.values()とdict.keys()は対応している
def get_maxkey_dict(d):
    indexes = np.argsort(list(d.values()))
    index = indexes[len(indexes)-1]
    return list(d.keys())[index]
