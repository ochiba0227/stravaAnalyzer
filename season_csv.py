# encoding: utf-8
import csv
import json
import os
import sys
import multiprocessing
import winsound
import re
from datetime import datetime

##ファイル名,データ配列,保存したいキー値
def write_csv(fname,data,keys):
    header = dict([(val,val)for val in keys])
     
    with open(fname, mode='w') as f:
        data.insert(0,header)
        writer = csv.DictWriter(f, keys, extrasaction='ignore', lineterminator="\n")
        writer.writerows(data)

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

##平均、最高速度、最高速度を記録した地点をdictで取得
def get_speeds(stream_data):
    moving = []
    speed = []
    points = []
    for data in stream_data:
        if data['type'] == 'VELOCITY':
            speed = data['data']
        if data['type'] == 'MOVING':
            moving = data['moving']
        if data['type'] == 'MAPPOINT':
            points = data['mapPoints']
##    moving=falseでもvelocity=0ではない
    counter = 0
    speed_list = []
    for m in moving:
        if m is True:
            speed_list.append(speed[counter])
        counter += 1;

    return_dict = {}
    key = 0
    return_dict['averageSpeed'] = sum(speed)/len(speed)
    return_dict['maxSpeed'] = max(speed,key = speed.index)
    return_dict['maxSpeedLat'] = points[key]['latitude']
    return_dict['maxSpeedLng'] = points[key]['longitude']
    if len(speed_list) > 0:
        key_m = speed_list.index(max(speed_list))
##        key_m = 0
        return_dict['averageSpeedMoving'] = sum(speed_list)/len(speed_list)
        return_dict['maxSpeedMoving'] = max(speed_list)
        return_dict['maxSpeedLatMoving'] = points[key_m]['latitude']
        return_dict['maxSpeedLngMoving'] = points[key_m]['longitude']
    return return_dict

##取得したディレクトリからjsonファイルを全て読み込みcsv出力
def get_files(path):
##    fname = re.split( r'\\', path )
##    if os.path.exists(".\\results\\season_"+fname[len(fname)-1]+".csv"):
##        print("exists:"+str(fname))
##        return

    files = os.listdir(path)
    data_list = []
    segment_id = ""
    for file in files:
        root,ext = os.path.splitext(file)
        if is_json(ext):
            data = decode_json(os.path.join(path,file))
            if data:
                effort_data = data['segmentEffortData']
                stream_data = data['segmentStreamData']
                effort_data['segmentid']=effort_data['segment']['id']
                effort_data['userid']=effort_data['athlete']['id']
                start_latlng,end_latlng = get_start_end_latlng(stream_data)
                speeds_dict = get_speeds(stream_data)
                temp = {'userid':effort_data['userid'],
                        'startDate':effort_data['startDate'],
                        'elapsedTime':effort_data['elapsedTime'],
                        'year':effort_data['startDateLocal']['date']['year'],
                        'month':effort_data['startDateLocal']['date']['month'],
                        'day':effort_data['startDateLocal']['date']['day'],
                        'hour':effort_data['startDateLocal']['time']['hour'],
                        'minute':effort_data['startDateLocal']['time']['minute'],
                        'distance':effort_data['distance'],
                        'startIndex':effort_data['startIndex'],
                        'endIndex':effort_data['endIndex'],
                        'averageGrade':effort_data['segment']['averageGrade'],
                        'maximumGrade':effort_data['segment']['maximumGrade'],
                        'elevationHigh':effort_data['segment']['elevationHigh'],
                        'elevationLow':effort_data['segment']['elevationLow'],
                        'segmentstartLat':effort_data['segment']['startLatlng']['latitude'],
                        'segmentstartlng':effort_data['segment']['startLatlng']['longitude'],
                        'segmentendLat':effort_data['segment']['endLatlng']['latitude'],
                        'segmentendlng':effort_data['segment']['endLatlng']['longitude'],
                        'startLat':start_latlng['latitude'],
                        'startLng':start_latlng['longitude'],
                        'endLat':end_latlng['latitude'],
                        'endLng':end_latlng['longitude']
                        }
                temp.update(speeds_dict)
##                ケイデンスと心拍数は装置がある人のみ取得可能
                keys = ['averageCadence','averageHeartrate','maxHeartrate','averageWatts']
                for key in keys:
                    temp[key] = compare_dictkeys(key,effort_data)
                data_list.append(temp);
                segment_id = effort_data['segmentid']
##    csvに書き出す要素をkeyで指定
##    keys = ['userid','month','hour']
    if len(data_list) > 0:
        keys = data_list[0].keys()
        write_csv(".\\results\\season_"+str(segment_id)+".csv",data_list,keys)
        print("finished:"+path)
    
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
    working_path = 'F:\\study\\strava\\finished\\'
    dirs = os.listdir(working_path)
    dirs = add_workingpath(dirs,working_path)
    p = multiprocessing.Pool()
    p.daemon = True
    remove_none(p.map(get_files,dirs))
##    for d in dirs:
##        get_files(d)
    p.close()
    print("finished:"+datetime.now().strftime("%Y%m%d%H%M%S"))
    winsound.PlaySound('se_moa01.wav',winsound.SND_FILENAME)
