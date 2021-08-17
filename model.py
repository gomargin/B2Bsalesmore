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
    :return: 재구성한 데이터프레임(index: date, column: 'day2sin', 'traffic')
    """
    df['day'] = df['datetime'].dt.weekday
    df['day2sin'] = np.sin((df['day']/7)*2*np.pi)   # datetime to sin
    df['date'] = df['datetime'].dt.date   # date column 추가
    df = df[['date', 'day2sin', 'traffic']]
    dp_df = df.groupby(['date'], as_index=True).max()   # day peak dataframe
    dp_df.dropna(axis=0, inplace=True)
    return dp_df


class Scaler:
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


class Train:
    def __init__(self, itemid, pred_len, in_steps, out_steps, valid_per, epochs, batch_size, unit, drop_per):
        self.itemid = itemid
        self.pred_len = pred_len
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

    def train_model(self, df):
        train_x, train_y = self.split_data(df)
        model = self.generate_lstm(train_x)
        # verbose 1: 언제 training 멈추었는지 확인 가능
        early_stopping = EarlyStopping(monitor='loss', mode='min', patience=10, verbose=1)
        model_name = '{}.h5'.format(self.itemid)
        model_path = os.getcwd() + '/models/' + model_name
        model_check = ModelCheckpoint(filepath=model_path, monitor='loss', mode='min', save_best_only=True)
        # verbose 0: silence, 1: progress bar, 2: one line per each
        model.fit(train_x, train_y, self.batch_size, self.epochs, validation_split=self.valid_per,
                  callbacks=[early_stopping, model_check], verbose=0)

    def predict_model(self, df):
        model = load_model('./models/{}.h5'.format(self.itemid))
        input_data = df.values[-self.in_steps:]
        day2sin_list = [np.sin((i / 7) * 2 * np.pi) for i in range(7)]
        for _ in range(self.pred_len):
            input_x = input_data[-3:].reshape(1, 3, 2)
            prediction = model.predict(input_x)[0][0]
            last_day = input_data[-1][0]
            last_idx = day2sin_list.index(last_day)
            if last_idx != 6:
                next_ = day2sin_list[last_idx + 1]
            else:
                next_ = 0
            new_row = [next_, prediction]
            input_data = np.append(input_data, [new_row], axis=0)

        return input_data[3:, -1]
