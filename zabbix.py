import pandas as pd
import numpy as np
import pymysql

class Zabbix():
    def __init__(self, user, passwd, host, db, port, charset):
        """ Zabbix class attribute 정의

        zabbix db 연결을 위한 attribute를 정의하는 함수.
        
        Args:
            - user
            - passwd
            - host
            - db
            - port
            - charset
        """
        self.db_connect = pymysql.connect(user=user, passwd=passwd, host=host, db=db, port=port, charset=charset)
        self.curs = self.db_connect.cursor()

    def read_db(self, sql_command):
        """ zabbix db 읽어오기

        Args:
            - sql_command
        
        Return:
            - df: 데이터프레임
        """
        self.curs.execute(sql_command)
        df = pd.DataFrame(self.curs.fetchall())
        df = df.replace('', np.nan)
        df.dropna(axis=0, inplace=True)

        return df

    def read_one(self, sql_command):
        """ zabbix db 특정 데이터 읽어오기

        Args:
            - sql_command

        Return:
            - one: 특정 데이터
        """
        self.curs.execute(sql_command)
        one = self.curs.fetchone()
        return one

    def delete_db(self, itemid):
        """ 해당 itemid가 포함된 row 삭제

        Args:
            - itemid
        """
        sql_command = "DELETE FROM test_ys WHERE itemid='{}'".format(itemid)
        self.curs.execute(sql_command)
        self.db_connect.commit()

    def insert_db(self, df):
        """ 예측 데이터(type: df) 삽입

        Args:
            - df: AI 예측 데이터프레임
        """
        for row in range(len(df)):
            sql_command = "INSERT INTO test_ys (itemid, dates, traffic) VALUES ('{}', '{}', '{}')".format(
                df.iloc[row][0],
                df.iloc[row][1],
                df.iloc[row][2])
            self.curs.execute(sql_command)
            self.db_connect.commit()