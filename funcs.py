# encoding: utf-8
import csv
import json
import os
import sys
import multiprocessing
import winsound
import re
import random
from datetime import datetime
import numpy as np
import inspect
from sklearn.externals import joblib
import linecache

##名前を指定してモデルを保存
def save_model(model,name):
    save_dir = 'results\\models\\'+name
    make_dir(save_dir)
    joblib.dump(model, os.path.join(save_dir,name+'.pkl')) 

##myjsonファイルの読み込み
##numが指定されていれば，num個のdictを含むlistを返す
##randflagがTrueで与えられていればrowsをランダムに並び替え
##指定されていなければ全部返す
##一行読み込みlinecache
##http://qiita.com/Kodaira_/items/eb5cdef4c4e299794be3
def read_myjson(fname,num,randflag):
    random.seed(1)
    diclist = []
    rows = list(range(1,1+sum(1 for line in open(fname))))
    if randflag is True:
        random.shuffle(rows)
    if num is not None:
        rows = rows[:num]
    for row in rows:
        diclist.append(json.loads(linecache.getline(fname, row)))
        linecache.clearcache()
    return diclist

##myjsonファイルの書き込み
def write_myjsonfile(fname,data):
##    ディレクトリの作成
    make_dir(os.path.dirname(fname))
    f = open(fname,'a')
    f.write(json.dumps(data)+'\n')
    f.close()

##jsonファイルの更新
##不要
def update_jsonfile(fname,data):
    print("avoid method update_jsonfile!!!!!!!!!!")
    return
    temp = []
##    存在する場合読み込んで追記
    if os.path.exists(fname):
        f = open(fname)
        temp = json.load(f)
        f.close()
##存在しない場合リスト型に変換してから書き込み
    temp.append(data)
    write_json(fname,temp)

##jsonファイルの保存
def write_json(fname,data):
##    ディレクトリがなければ作成
    make_dir(os.path.dirname(fname))
    with open(fname, mode='w') as f:
        json.dump(data, f)

##ファイル名,データ配列,保存したいキー値
def write_csv(fname,data,keys):
##    ディレクトリがなければ作成
    make_dir(os.path.dirname(fname))
    
    header = dict([(val,val)for val in keys])
     
    with open(fname, mode='a') as f:
        data.insert(0,header)
        writer = csv.DictWriter(f, keys, extrasaction='ignore', lineterminator="\n")
        writer.writerows(data)

##ファイルオブジェクト,データdict
##keysをソートしてから出力しないと、でたらめなファイルになる。
def write_csv_onedata(f,data):
    keys = sorted(data.keys())
    if f.tell() == 0:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(keys)
    writer = csv.DictWriter(f, keys, lineterminator="\n")        
    writer.writerow(data)

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
    time = datetime.now()
    line = 'start:'+filename+' on:'+time.strftime("%Y/%m/%d %H:%M:%S")
    print(line)
    write_log(line)
    return time

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

##ディレクトリにあるファイルオブジェクトの取得
##mainとして実行されいているファイル名のディレクトリにあるデータを参照する
##fnameファイル名のみ，param:rwa，joinpathを指定するとファイル名の次の階層のディレクトリを作成できる
def get_fileobj(fname,param,joinpath):
    root,ext = os.path.splitext(os.path.basename(inspect.currentframe().f_back.f_code.co_filename))
    opath = os.path.join('results',root)
    if joinpath is not None:
        opath = os.path.join(opath,joinpath)
    if param in ['w','a']:
        make_dir(opath)
    try:
        return open(os.path.join(opath,fname),param)
    except Exception as e:
        print(e)
        return None

##ディレクトリにあるファイルパスの取得
##mainとして実行されいているファイル名のディレクトリにあるデータを参照する
##fnameファイル名のみ，param:rwa，joinpathを指定するとファイル名の次の階層のディレクトリを作成できる
def get_filepath(fname,joinpath):
    root,ext = os.path.splitext(os.path.basename(inspect.currentframe().f_back.f_code.co_filename))
    opath = os.path.join('results',root)
    if joinpath is not None:
        opath = os.path.join(opath,joinpath)
    return os.path.join(opath,fname)

##辞書から最大値を持つキーを返す
##dict.values()とdict.keys()は対応している
def get_maxkey_dict(d):
    indexes = np.argsort(list(d.values()))
    index = indexes[len(indexes)-1]
    return list(d.keys())[index]
