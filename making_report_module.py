import sqlite3
import datetime
import threading
from threading import Thread, currentThread
import time
import pymysql
from reporting import Reporting

if __name__ == '__main__':

    loop_count = 0
    while True:
        db_connect = pymysql.connect(user='zabbix', passwd='zabbix', host='210.121.218.5', db='test', port=3306,
                                     charset='euckr')
        curs = db_connect.cursor()
        curs.execute('SELECT * FROM report')
        db_connect.commit()

        # report table을 행 별로 읽어오기
        rows = [list(row) for row in curs.fetchall()]   # 수정
        for row in rows:
            # row[6] ==0 이면 report 발행 요청
            if row[6] == 0:
                report = Reporting(row)
                my_thread = Thread(target=report.request())
                my_thread.start()
                print(row[3], threading.currentThread().getName())


        time.sleep(3)
        loop_count += 1
        # loop_count 횟수만큼 db screening 완료
        if loop_count % 20 == 0:
            print('Running...' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))