import pandas as pd
import numpy as np
import pymysql


class Zabbix:
    def __init__(self):
        self.zabbix_conn = pymysql.connect(user='zabbix',
                                          passwd='zabbix',
                                          host='210.121.218.5',
                                          db='ZABBIXDB',
                                          port=3306,
                                          charset='euckr')
        self.zabbix_curs = self.zabbix_conn.cursor()
        self.test_conn = pymysql.connect(user='zabbix',
                                          passwd='zabbix',
                                          host='210.121.218.5',
                                          db='test',
                                          port=3306,
                                          charset='euckr')
        self.test_curs = self.test_conn.cursor()

    def update(self, *args):
        """test의 report table update

        """
        if len(args) == 2:
            sql = 'UPDATE report SET state="{}" WHERE idx="{}"'.format(args[0], args[1])
        else:
            sql = 'UPDATE report SET state="{}", url="{}" WHERE idx="{}"'.format(args[0], args[1], args[2])
        self.test_curs.execute(sql)
        self.test_conn.commit()

    def read(self, sql):
        """ zabbix db 읽어오기

        Args:
            - sql_command
        Return:
            - df: 데이터프레임
        """
        self.zabbix_curs.execute(sql)
        df = pd.DataFrame(self.zabbix_curs.fetchall())
        df = df.replace('', np.nan)
        df.dropna(axis=0, inplace=True)

        return df

    def read_one(self, sql):
        """ zabbix db 특정 데이터 읽어오기

        Args:
            - sql_command
        Return:
            - one: 특정 데이터
        """
        self.zabbix_curs.execute(sql)
        one = self.zabbix_curs.fetchone()
        return one

    def commit_db(self, sql):
        """sql명령문을 받아 실행

        Args:
            - sql_command
        """
        self.zabbix_curs.execute(sql)
        self.zabbix_conn.commit()

    def delete(self, itemid):
        """ 해당 itemid가 포함된 row 삭제

        Args:
            - itemid
        """
        sql = "DELETE FROM test_ys WHERE itemid='{}'".format(itemid)
        self.zabbix_curs.execute(sql)
        self.zabbix_conn.commit()

    def insert(self, df):
        """ 예측 데이터(type: df) 삽입

        Args:
            - df: AI 예측 데이터프레임
        """
        for row in range(len(df)):
            sql = "INSERT INTO test_ys (itemid, dates, traffic) VALUES ('{}', '{}', '{}')".format(
                df.iloc[row][0],
                df.iloc[row][1],
                df.iloc[row][2])
            self.zabbix_curs.execute(sql)
            self.zabbix_conn.commit()
