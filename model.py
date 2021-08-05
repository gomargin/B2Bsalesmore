from pptx import Presentation
from pptx.util import Inches
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import sys
import os
import io
import re
import pymysql
import pandas as pd
import numpy as np
import datetime

from zabbix import Zabbix
from dateutil import rrule
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


def reframe_df(df):
    """
    :param db에서 로드한 데이터프레임
    :return: 재구성한 데이터프레임(index: date, column: 'dt2sin', 'traffic')
    """
    week2sec = 7 * 24 * 60 * 60   # 일주일을 초 단위로 변환
    dt2ts = df['datetime'].map(datetime.datetime.timestamp)   # datetime to timestamp
    df['dt2sin'] = np.sin(dt2ts*(2*np.pi/week2sec))   # datetime to sin
    df['date'] = df['datetime'].dt.date   # date column 추가
    df.drop(['itemid', 'datetime'], axis=1, inplace=True)
    df = df[['date', 'dt2sin', 'traffic']]
    dp_df = df.groupby(['date'], as_index=True).max()   # day peak dataframe
    dp_df = dp_df.dropna()
    return dp_df


class Scaler():
    def __init__(self, df, feature):
        self.df = df
        self.feature = feature
        self.max = max(self.df[self.feature])
        self.min = min(self.df[self.feature])

    def normalization(self):
        self.df[self.feature] = list((self.df[self.feature]-self.min)/(self.max-self.min))
        return self.df

    def rev_normalization(self, array):
        pred_array = array*(self.max-self.min)+self.min
        return pred_array


class Train():
    def __init__(self, itemid, in_steps, out_steps, valid_per, epochs, batch_size, unit, drop_per):
        self.itemid = itemid
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.valid_per = valid_per
        self.epochs = epochs
        self.batch_size = batch_size
        self.unit = unit
        self.drop_per = drop_per

    def split_data(self, df):
        train_x = []
        train_y = []
        for i in range(len(df) - (self.in_steps + self.out_steps)):
            train_x.append(df.values[i: i + self.in_steps, :])
            train_y.append(df.values[i + self.in_steps: i + (self.in_steps + self.out_steps), -1])
        train_x, train_y = np.array(train_x), np.array(train_y)

        return train_x, train_y

    def generate_lstm(self, data):
        # data -> train_x
        model = Sequential([
            LSTM(self.unit, return_sequences=True, input_shape=(data.shape[1], data.shape[2])),
            Dropout(self.drop_per),
            LSTM(self.unit, return_sequences=False),
            Dropout(self.drop_per),
            Dense(self.out_steps)
        ])
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        return model

    def predict(self, df):
        train_x, train_y = self.split_data(df)
        model = self.generate_lstm(train_x)
        # verbose 1: 언제 training 멈추었는지 확인 가능
        early_stopping = EarlyStopping(monitor='loss', mode='min', patience=10, verbose=1)
        model_name = '{}.h5'.format(self.itemid)
        model_path = os.getcwd() + '/models/' + model_name
        model_check = ModelCheckpoint(filepath=model_path, monitor='loss', mode='min', save_best_only=True)
        # verbose 0: silence, 1: progress bar, 2: one line per each
        hist = model.fit(train_x, train_y, self.batch_size, self.epochs, validation_split=self.valid_per,
                         callbacks=[early_stopping, model_check], verbose=0)
        test_data = df.values[-self.in_steps:]
        test_x = test_data.reshape(1, self.in_steps, df.shape[1])
        # model = load_model('./models/{}.h5'.format(self.itemid))
        prediction = model.predict(test_x)
        return prediction