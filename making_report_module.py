import sqlite3
import datetime
from threading import Thread
import time
from reporting_module import making_report
import pymysql


def reporting(idx, datetime, line_no, date_start, date_end):

    db_connect = pymysql.connect(user='zabbix', passwd='zabbix', host='210.121.218.5', db='test', port=3306, charset='euckr')
    # curs = db_connect.cursor(pymysql.cursors.DictCursor)
    # db_connect = sqlite3.connect('report.db')
    curs = db_connect.cursor()
    # db state 0 -> 1로 update
    curs.execute('UPDATE report SET state="{}" WHERE idx="{}"'.format(1, idx))
    db_connect.commit()

    try:
        url = making_report(datetime, line_no, date_start, date_end)   # 수정
        print('report 생성 완료: {}'.format(url))   # 수정

        curs.execute('UPDATE report SET state="{}", url="{}" WHERE idx="{}"'.format(2, str(url), idx))
        db_connect.commit()
        db_connect.close()

    except:
        print('report 생성 오류 발생!')   # 수정
        curs.execute('UPDATE report SET state="{}", url="{}" WHERE idx="{}"'.format(9, 'error', idx))
        db_connect.commit()
        db_connect.close()


if __name__ == '__main__':
    loop_count = 0

    while True:
        db_connect = pymysql.connect(user='zabbix', passwd='zabbix', host='210.121.218.5', db='test', port=3306, charset='euckr')
        # curs = db_connect.cursor(pymysql.cursors.DictCursor)
        # db_connect = sqlite3.connect('report.db')
        curs = db_connect.cursor()
        curs.execute('SELECT * FROM report')
        db_connect.commit()
        rows = [list(row) for row in curs.fetchall()]   # 수정

        for row in rows:
            # row[6] ==0 이면 reporting 실행
            if row[6] == 0:
                my_thread = Thread(target=reporting, args=(row[0], row[2], row[3], row[4], row[5],))
                my_thread.start()
                print('report 생성 요청')
                break
            else:
                pass

        time.sleep(3)
        loop_count += 1
        # loop_count 횟수만큼 db screening 완료
        if loop_count % 3 == 0:
            print('Running...' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))