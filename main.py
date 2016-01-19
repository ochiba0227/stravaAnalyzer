# encoding: utf-8
import json
import os

def write_file(fname,text):
    f = open(fname, 'w') # 書き込みモードで開く
    f.write(text) # 引数の文字列をファイルに書き込む
    f.close()

##jsonのデコード
def decode_json(file):
    ##f = open("7417_132941303.json")
    ##weather_dict = json.loads(f)
    ##weather_format_json = json.dumps(weather_dict, indent=4, separators=(',', ': '))
    ##print(weather_format_json)
    with open(file, 'r') as f:
        fenrifja_dic = json.load(f)
        mappoint = fenrifja_dic.keys()
    ##    print(fenrifja_dic['segmentStreamData'][0]['originalSize'])
        print(len(fenrifja_dic['segmentStreamData']))
        test = fenrifja_dic['segmentStreamData']
    ## ケイデンス等取得データは10種類ある。以下のように確認
##        print('CADENCE' in test[5].values())
##        距離は以下のように取得、単位はm
##        print(fenrifja_dic['segmentEffortData']['distance'])

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

##取得したディレクトリからjsonファイルを一つだけ取得し、セグメントの距離などをファイルに記録
def make_filelist(path):
    files = os.listdir(path)
    for file in files:
        root,ext = os.path.splitext(file)
        if is_json(ext):
            print(os.path.join(path,file))
            decode_json(os.path.join(path,file))
            break
##    ファイルに書き込み
    write_file()

##--------------プログラム開始------------------
for dirs in get_dirs('.'):
    make_filelist(dirs)
