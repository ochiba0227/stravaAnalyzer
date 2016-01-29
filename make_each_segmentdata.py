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

##mappointsの最初と最後を取得
def get_start_end_latlng(stream_data):
    for data in stream_data:
        if data['type'] == 'MAPPOINT':
            points = data['mapPoints']
            startlatlng = points[0]
            endlatlng = points[len(points)-1]
            return [startlatlng,endlatlng]
    return None

##listの平均をとる．listが空なら-1
def get_average_speed(speed_list):
    length = len(speed_list)
    if length > 0:
        return sum(speed_list)/length
    return -1

##平地、上り(3%以上)、下り(3%以下)それぞれの速度の平均値
def get_speeds_grade(speed,grade):
    averageSpeedPosgrade = []
    averageSpeedNeggrade = []
    averageSpeedNograde = []

    counter = 0
    for g in grade:
        if g >= 3.0:
            averageSpeedPosgrade.append(speed[counter])
        elif g <= -3.0:
            averageSpeedNeggrade.append(speed[counter])
        else:
            averageSpeedNograde.append(speed[counter])
        counter += 1;

    return_dict = {}
    return_dict['averageSpeedPosgrade'] = get_average_speed(averageSpeedPosgrade)
    return_dict['averageSpeedNeggrade'] = get_average_speed(averageSpeedNeggrade)
    return_dict['averageSpeedNograde'] = get_average_speed(averageSpeedNograde)
    return return_dict

##平均、最高速度、最高速度を記録した地点をdictで取得
def get_speeds(stream_data):
    moving = []
    speed = []
    points = []
    grade = []
    for data in stream_data:
        if data['type'] == 'VELOCITY':
            speed = data['data']
        if data['type'] == 'MOVING':
            moving = data['moving']
        if data['type'] == 'MAPPOINT':
            points = data['mapPoints']
        if data['type'] == 'GRADE':
            grade = data['data']
##    moving=falseでもvelocity=0ではない
    counter = 0
    speed_list = []
    for m in moving:
        if m is True:
            speed_list.append(speed[counter])
        counter += 1;

    return_dict = {}

    maxspeed = max(speed)
    key = speed.index(maxspeed)
    return_dict['averageSpeed'] = sum(speed)/len(speed)
    return_dict['maxSpeed'] = maxspeed
    return_dict['maxSpeedLat'] = points[key]['latitude']
    return_dict['maxSpeedLng'] = points[key]['longitude']
    if len(speed_list) > 0:
        maxspeed = max(speed_list)
        key_m = speed_list.index(maxspeed)
        return_dict['averageSpeedMoving'] = sum(speed_list)/len(speed_list)
        return_dict['maxSpeedMoving'] = maxspeed
        return_dict['maxSpeedLatMoving'] = points[key_m]['latitude']
        return_dict['maxSpeedLngMoving'] = points[key_m]['longitude']

    return_dict.update(get_speeds_grade(speed,grade))
    return return_dict

##平均勾配、最高勾配、最低勾配をdictで取得
def get_grades(stream_data):
    for data in stream_data:
        if data['type'] == 'GRADE':
            grade = data['data']
            break
    return_dict = { 'maxGrade' : max(grade),
                    'minGrade' : min(grade),
                    'averageGrade' : sum(grade)/len(grade)}
    return return_dict

##セグメント開始、終了時の走行距離をdictで取得
def get_distances(stream_data):
    for data in stream_data:
        if data['type'] == 'DISTANCE':
            distance = data['data']
            break
    return_dict = { 'startDistance' : distance[0],
                    'endDistance' : distance[len(distance)-1]}
    return return_dict

##取得したディレクトリからjsonファイルを全て読み込みcsv出力
def get_files(path):
    files = os.listdir(path)
    for file in files:
        root,ext = os.path.splitext(file)
        if funcs.is_json(ext):
            data = funcs.decode_json(os.path.join(path,file))
            if data:
                effort_data = data['segmentEffortData']
                stream_data = data['segmentStreamData']
                effort_data['segmentid']=effort_data['segment']['id']
                effort_data['userid']=effort_data['athlete']['id']
                start_latlng,end_latlng = get_start_end_latlng(stream_data)
                speeds_dict = get_speeds(stream_data)
                grades_dict = get_grades(stream_data)
                distances_dict = get_distances(stream_data)
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
                temp.update(grades_dict)
                temp.update(distances_dict)
##                ケイデンスと心拍数は装置がある人のみ取得可能
                keys = ['averageCadence','averageHeartrate','maxHeartrate','averageWatts']
                for key in keys:
                    temp[key] = funcs.compare_dictkeys(key,effort_data)
                segment_id = str(effort_data['segmentid'])
                user_id = str(effort_data['userid'])
                effort_id = str(effort_data['id'])
                climb_category = str(effort_data['segment']['climbCategory'])
                out_path = '.\\results\\data_for_each_category2\\'+user_id+'\\'+climb_category+'\\'
                funcs.make_dir(out_path)
##                keys = temp.keys()
##                funcs.write_csv(path+segment_id + '_' + effort_id + '.csv',temp,keys)
                with open(out_path+segment_id + '_' + effort_id + '.json', 'w') as f:
                    json.dump(temp, f, sort_keys=True, indent=4)
                
##    csvに書き出す要素をkeyで指定
##    keys = ['userid','month','hour']
##    if len(data_list) > 0:
##        keys = data_list[0].keys()
##        funcs.write_csv(".\\results\\data_for_each_category\\"+root+".csv",data_list,keys)
##        print(root)
    print("finished:"+path)

##与えられたpath以下のjsonファイルのパスを取得
def get_filepath(path):
    files = os.listdir(path)
    jsonfiles = []
    for file in files:
        root,ext = os.path.splitext(file)
        if funcs.is_json(ext):
            jsonfiles.append(os.path.join(path,file))
    return jsonfiles

##pathで与えられたjsonファイルのデータを分析用ファイルにdumpし、dumpした辞書を返す
def get_data(path):
    data = funcs.decode_json(path)
    if data:
        effort_data = data['segmentEffortData']
        stream_data = data['segmentStreamData']
        effort_data['segmentid']=effort_data['segment']['id']
        effort_data['userid']=effort_data['athlete']['id']
        start_latlng,end_latlng = get_start_end_latlng(stream_data)
        speeds_dict = get_speeds(stream_data)
        grades_dict = get_grades(stream_data)
        distances_dict = get_distances(stream_data)
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
        temp.update(grades_dict)
        temp.update(distances_dict)
##                ケイデンスと心拍数は装置がある人のみ取得可能
        keys = ['averageCadence','averageHeartrate','maxHeartrate','averageWatts']
        for key in keys:
            temp[key] = funcs.compare_dictkeys(key,effort_data)
        segment_id = str(effort_data['segmentid'])
        user_id = str(effort_data['userid'])
        effort_id = str(effort_data['id'])
        climb_category = str(effort_data['segment']['climbCategory'])
        out_path = '.\\results\\data_for_each_category_3\\'+user_id+'\\'+climb_category+'\\'
        funcs.make_dir(out_path)
##                keys = temp.keys()
##                funcs.write_csv(path+segment_id + '_' + effort_id + '.csv',temp,keys)
        with open(out_path+segment_id + '_' + effort_id + '.json', 'w') as f:
            json.dump(temp, f, sort_keys=True, indent=4)
        return temp
    
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
    working_path = 'F:\\study\\strava\\finished\\'
    dirs = os.listdir(working_path)
    dirs = funcs.add_workingpath(dirs,working_path)
    p = multiprocessing.Pool()
    p.daemon = True
##    funcs.remove_none(p.map(get_files,dirs))
    data_list = []
    for paths in p.imap(get_filepath,dirs):
        for data in p.imap(get_data,paths):
            data_list.append(data)
    p.close()
##    データ取得終了したら全データをcsvに書き出し
    keys = data_list[0].keys()
    funcs.write_csv(".\\results\\user_data_all.csv",data_list,keys)
    funcs.end_program()
