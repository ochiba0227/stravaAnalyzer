# encoding: utf-8
import csv
import json
import os
from datetime import datetime

##ファイル名,データ配列,保存したいキー値
def write_csv(fname,data,keys):
    header = dict([(val,val)for val in keys])
     
    with open(fname, mode='w') as f:
        data.insert(0,header)
        writer = csv.DictWriter(f, keys, extrasaction='ignore')
        writer.writerows(data)

##jsonのデコード
def decode_json(file):
    ##f = open("7417_132941303.json")
    ##weather_dict = json.loads(f)
    ##weather_format_json = json.dumps(weather_dict, indent=4, separators=(',', ': '))
    ##print(weather_format_json)
    with open(file, 'r') as f:
        json_dic = json.load(f)
        effort_data = json_dic['segmentEffortData']
    ## ケイデンス等取得データは10種類ある。以下のように確認
##        print('CADENCE' in test[5].values())
##        距離は以下のように取得、単位はm
##        print(fenrifja_dic['segmentEffortData']['distance'])
        f.close()
    return effort_data
        

##拡張子がjsonかどうか確認する
def is_json(ext):
    if ext == ".json":
        return True
    return False

##再帰的にディレクトリの取得
def get_dirs(path):
    dirs=[]
    for item in os.listdir(path):
        item = os.path.join(path,item)
        if os.path.isdir(item):
            nextdir = get_dirs(item)
            for gotdir in nextdir:
                dirs.append(gotdir)
            dirs.append(item)
    return dirs

##取得したディレクトリからjsonファイルを一つだけ取得し、セグメントの距離などをreturn
def get_onedata(path):
    files = os.listdir(path)
    for file in files:
        root,ext = os.path.splitext(file)
        if is_json(ext):
            return decode_json(os.path.join(path,file))

##--------------プログラム開始------------------
datas=[]
for dirs in get_dirs('.'):
    data = get_onedata(dirs)
    if data:
        datas.append(data)

keys = ['id','name','distance']
write_csv(".\\results\\"+datetime.now().strftime("%Y%m%d%H%M%S")+".txt",datas,keys)
