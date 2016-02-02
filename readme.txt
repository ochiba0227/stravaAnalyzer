#.myjsonの仕様
一行に一つのjsonデータ
読み込み：funcs.readmyjson(filename)
書き込み：funcs.writemyjson(filename,data)

#プログラムの勉強用に作成
decode_jsondata.py:jsonデータのデコードのテスト用
get_usernum.py:各セグメントごとのユーザ数を取得
make_segmentdata.py:各セグメント毎にユーザの走行結果をまとめて作成
|- make_userdata_from_segments.py:make_segmentdata.pyで作成したファイルからデータを作成
userdata_analysis.py:機械学習の勉強用

#実験用(ダメなヤツ)
funcs.py:共通利用する関数
make_each_segmentdata.py:各ユーザごとのデータをクライムカテゴリごとに分類して作成
|- make_type_data.py:make_each_segmentdata.pyで作成したデータから各ユーザごとのデータを作成
 |- learning.py:機械学習用プログラム
 
#実験用
funcs.py:共通利用する関数
cluster_analysis.py:ユーザをクラスタに分割する
makedata_for_regression.py:ユーザにラベルを付けて必要なデータのみに整形
regression.py:クラスタに基づいてユーザを分割し，速度を予測