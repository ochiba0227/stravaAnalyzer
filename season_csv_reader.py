# encoding: utf-8
import csv

def sum_month(datas):
    month_list = {}
    for data in datas:
        month = data['month']
        if month in month_list:
            month_list[month] += 1
        else:
            month_list[month] = 1
    return month_list

def sum_hour(datas):
    hour_list = {}
    for data in datas:
        hour = data['hour']
        if hour in hour_list:
            hour_list[hour] += 1
        else:
            hour_list[hour] = 1
    return hour_list

def sum_userid(datas):
    userid_list = {}
    for data in datas:
        userid = data['userid']
        if userid in userid_list:
            userid_list[userid] += 1
        else:
            userid_list[userid] = 1
    return userid_list

f = open('F:\\study\\analiser\\results\\season_2554959.csv', 'r')
reader = csv.DictReader(f)
got_dict = list(reader)
print(sum_month(got_dict))
print(sum_hour(got_dict))
print(sum_userid(got_dict))

for k, v in sorted(sum_userid(got_dict).items(), key=lambda x:x[1]):
    print(k, v)
